import os
import sys
import time
import hashlib
import logging
import threading
import random
from abc import ABC, abstractmethod
from typing import List, Optional

user_site = os.path.expanduser(
    "~/AppData/Roaming/Python/Python313/site-packages"
)
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import httpx
import numpy as np

from app.config import settings

_LAZY_ST_SENTINEL = object()
SentenceTransformer = _LAZY_ST_SENTINEL

def _get_sentence_transformer_cls():
    global SentenceTransformer
    if SentenceTransformer is None:
        return None
    if SentenceTransformer is not _LAZY_ST_SENTINEL:
        return SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
        return SentenceTransformer
    except ImportError:
        SentenceTransformer = None
        return None

logger = logging.getLogger("repolens.ai.embeddings")


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


class MockEmbedding(EmbeddingProvider):
    """Deterministic pseudo-embedding provider for testing/demo."""

    def __init__(self, dim: Optional[int] = None, model_name: Optional[str] = None):
        self._dim = (
            dim
            if dim is not None
            else getattr(settings, "EMBEDDING_DIMENSION", 384)
        )
        self._model_name = (
            model_name
            if model_name is not None
            else f"mock-embedding-{self._dim}d"
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _text_to_vector(
        self, text: str, dim: Optional[int] = None
    ) -> List[float]:
        target_dim = dim if dim is not None else self._dim
        vec = np.zeros(target_dim, dtype=np.float32)

        for idx, word in enumerate(text.lower().split()):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            vec[h % target_dim] += 1.0 / (idx + 1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)


_cached_local_model = None
_local_model_lock = threading.Lock()


class LocalEmbedding(EmbeddingProvider):
    """Semantic embeddings generated locally using Sentence Transformers with lazy singleton caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: Optional[int] = None,
    ):
        self._model_name = model_name
        self._dim = 384
        if batch_size is None or not (isinstance(batch_size, int) and batch_size > 0):
            raw_batch = os.environ.get("EMBEDDING_BATCH_SIZE")
            safe_batch = 16
            if raw_batch is not None:
                try:
                    parsed = int(str(raw_batch).strip())
                    if parsed > 0:
                        safe_batch = parsed
                except (ValueError, TypeError):
                    pass
            batch_size = safe_batch
        self._batch_size = batch_size

    def _load_model(self):
        global _cached_local_model
        if _cached_local_model is not None:
            return _cached_local_model

        with _local_model_lock:
            if _cached_local_model is not None:
                return _cached_local_model

            ST = _get_sentence_transformer_cls()
            if ST is None:
                raise RuntimeError(
                    "sentence-transformers package is not installed. "
                    "Install sentence-transformers via `pip install sentence-transformers` to use local embeddings."
                )

            # Restrict PyTorch CPU thread pool to 1 thread to prevent multi-core memory blowups on 512 MB instances
            try:
                import torch
                try:
                    torch.set_num_threads(1)
                except (RuntimeError, AttributeError) as thread_exc:
                    logger.warning("Could not set PyTorch intra-op threads to 1: %s", thread_exc)

                try:
                    torch.set_num_interop_threads(1)
                except (RuntimeError, AttributeError) as interop_exc:
                    logger.warning("Could not set PyTorch inter-op threads to 1: %s", interop_exc)
            except ImportError:
                pass

            logger.info("Initializing local embedding model lazily: %s", self._model_name)
            try:
                _cached_local_model = ST(self._model_name)
                logger.info("Local embedding model loaded successfully: %s", self._model_name)
            except Exception as exc:
                logger.error("Failed to load local embedding model '%s': %s", self._model_name, exc, exc_info=True)
                raise RuntimeError(
                    f"Failed to load or download local embedding model '{self._model_name}': {exc}. "
                    "Ensure internet access is available for the initial model download (~80MB), "
                    "or pre-cache the model files in your Hugging Face cache directory."
                ) from exc

            return _cached_local_model

    @property
    def model(self):
        """Lazy-loaded singleton model instance for backward compatibility."""
        return self._load_model()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self._load_model()
        try:
            import torch
            inference_ctx = torch.inference_mode()
        except (ImportError, AttributeError):
            import contextlib
            inference_ctx = contextlib.nullcontext()

        try:
            with inference_ctx:
                embeddings = model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Local embedding document inference error: %s", exc, exc_info=True)
            raise RuntimeError(f"Local embedding generation failed for model '{self._model_name}': {exc}") from exc

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim

        model = self._load_model()
        try:
            import torch
            inference_ctx = torch.inference_mode()
        except (ImportError, AttributeError):
            import contextlib
            inference_ctx = contextlib.nullcontext()

        try:
            with inference_ctx:
                embedding = model.encode(
                    text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return embedding.tolist()
        except Exception as exc:
            logger.error("Local embedding query inference error: %s", exc, exc_info=True)
            raise RuntimeError(f"Local query embedding generation failed for model '{self._model_name}': {exc}") from exc


_cached_onnx_session = None
_cached_onnx_tokenizer = None
_onnx_model_lock = threading.Lock()


class ONNXLocalEmbedding(EmbeddingProvider):
    """
    Lightweight, CPU-only semantic embeddings using ONNX Runtime with quantized all-MiniLM-L6-v2.
    Designed for memory-constrained environments (e.g., 512 MB containers).
    Guarantees zero PyTorch or sentence-transformers imports in its runtime path.
    """

    MODEL_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2-qint8",
        model_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: int = 256,
    ):
        self._model_name = model_name
        self._model_dir = model_dir
        self._dim = 384
        self._max_length = max_length

        if batch_size is None or not (isinstance(batch_size, int) and batch_size > 0):
            raw_batch = os.environ.get("ONNX_EMBEDDING_BATCH_SIZE") or os.environ.get("EMBEDDING_BATCH_SIZE")
            safe_batch = 4
            if raw_batch is not None:
                try:
                    parsed = int(str(raw_batch).strip())
                    if parsed > 0:
                        safe_batch = parsed
                except (ValueError, TypeError):
                    pass
            batch_size = safe_batch
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _resolve_model_paths(self) -> tuple[str, str]:
        import urllib.request

        candidate_dirs = []
        if self._model_dir:
            candidate_dirs.append(self._model_dir)
        setting_dir = getattr(settings, "ONNX_MODEL_DIR", None)
        if setting_dir:
            candidate_dirs.append(setting_dir)
        env_dir = os.environ.get("ONNX_MODEL_DIR")
        if env_dir:
            candidate_dirs.append(env_dir)

        cache_dir = os.path.expanduser("~/.cache/repolens/all-MiniLM-L6-v2-onnx")
        candidate_dirs.append(cache_dir)

        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project_models_dir = os.path.join(backend_dir, "models", "all-MiniLM-L6-v2-onnx")
        candidate_dirs.append(project_models_dir)

        for cdir in candidate_dirs:
            m_path = os.path.join(cdir, "model_quantized.onnx")
            if not os.path.exists(m_path):
                m_path = os.path.join(cdir, "model.onnx")
            t_path = os.path.join(cdir, "tokenizer.json")
            if os.path.exists(m_path) and os.path.exists(t_path):
                return m_path, t_path

        os.makedirs(cache_dir, exist_ok=True)
        m_dest = os.path.join(cache_dir, "model_quantized.onnx")
        t_dest = os.path.join(cache_dir, "tokenizer.json")

        def _download_file(url: str, dest: str, label: str):
            if not os.path.exists(dest):
                logger.info("Downloading lightweight ONNX asset %s from %s...", label, url)
                tmp_dest = f"{dest}.tmp"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "RepoLensAI"})
                    with urllib.request.urlopen(req, timeout=60.0) as resp, open(tmp_dest, "wb") as f:
                        f.write(resp.read())
                    os.replace(tmp_dest, dest)
                    logger.info("Successfully cached %s (%.2f MB).", label, os.path.getsize(dest) / (1024 * 1024))
                except Exception as dl_err:
                    if os.path.exists(tmp_dest):
                        try:
                            os.remove(tmp_dest)
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Failed to download ONNX embedding asset '{label}' from {url}: {dl_err}. "
                        "Pre-place 'model_quantized.onnx' and 'tokenizer.json' in ~/.cache/repolens/all-MiniLM-L6-v2-onnx/ "
                        "or specify ONNX_MODEL_DIR."
                    ) from dl_err

        _download_file(self.TOKENIZER_URL, t_dest, "tokenizer.json")
        _download_file(self.MODEL_URL, m_dest, "model_quantized.onnx")

        return m_dest, t_dest

    def _load_model(self):
        global _cached_onnx_session, _cached_onnx_tokenizer
        if _cached_onnx_session is not None and _cached_onnx_tokenizer is not None:
            return _cached_onnx_session, _cached_onnx_tokenizer

        with _onnx_model_lock:
            if _cached_onnx_session is not None and _cached_onnx_tokenizer is not None:
                return _cached_onnx_session, _cached_onnx_tokenizer

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as imp_err:
                raise RuntimeError(
                    "onnxruntime and tokenizers packages are required for ONNX embeddings. "
                    "Install via `pip install onnxruntime tokenizers`."
                ) from imp_err

            model_path, tokenizer_path = self._resolve_model_paths()

            logger.info("Initializing ONNX embedding session from: %s", model_path)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            try:
                _cached_onnx_session = ort.InferenceSession(
                    model_path,
                    sess_options,
                    providers=["CPUExecutionProvider"],
                )
                tok = Tokenizer.from_file(tokenizer_path)
                tok.enable_truncation(max_length=self._max_length)
                tok.enable_padding(pad_to_multiple_of=8, pad_id=0, pad_token="[PAD]")
                _cached_onnx_tokenizer = tok
                logger.info("ONNX embedding session and tokenizer initialized successfully (384-d).")
            except Exception as load_err:
                logger.error("Failed to initialize ONNX embedding session: %s", load_err, exc_info=True)
                raise RuntimeError(f"Failed to load ONNX embedding model from '{model_path}': {load_err}") from load_err

            return _cached_onnx_session, _cached_onnx_tokenizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        session, tokenizer = self._load_model()
        all_embeddings: List[List[float]] = []

        try:
            for start in range(0, len(texts), self._batch_size):
                batch_slice = texts[start : start + self._batch_size]
                clean_texts = [str(t) if (t is not None and str(t).strip()) else " " for t in batch_slice]

                encoded = tokenizer.encode_batch(clean_texts)
                input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
                attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
                token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

                feed = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                input_names = [inp.name for inp in session.get_inputs()]
                if "token_type_ids" in input_names:
                    feed["token_type_ids"] = token_type_ids

                outputs = session.run(None, feed)
                last_hidden = outputs[0]  # Shape: (batch, seq_len, 384)

                # Mean pooling over attention mask
                mask_expanded = np.broadcast_to(
                    np.expand_dims(attention_mask, -1), last_hidden.shape
                ).astype(float)
                sum_embs = np.sum(last_hidden * mask_expanded, axis=1)
                sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                pooled = sum_embs / sum_mask

                # L2 normalize
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms = np.clip(norms, a_min=1e-12, a_max=None)
                batch_embeddings = (pooled / norms).tolist()

                for emb in batch_embeddings:
                    if len(emb) != self._dim:
                        raise ValueError(
                            f"ONNX embedding dimension mismatch: expected {self._dim}, got {len(emb)}"
                        )

                all_embeddings.extend(batch_embeddings)

                del clean_texts, encoded, input_ids, attention_mask, token_type_ids, feed, outputs, last_hidden, mask_expanded, sum_embs, sum_mask, pooled, norms, batch_embeddings
        except Exception as exc:
            logger.error("ONNX embedding document inference error: %s", exc, exc_info=True)
            raise RuntimeError(f"ONNX embedding generation failed: {exc}") from exc

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text or not str(text).strip():
            return [0.0] * self._dim
        results = self.embed_documents([str(text)])
        if not results:
            return [0.0] * self._dim
        return results[0]


ONNXEmbedding = ONNXLocalEmbedding


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self.api_key = api_key
        self._model = model
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self._model,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)

            if resp.status_code >= 400:
                raise RuntimeError(
                    "OpenAI embedding API error "
                    f"(HTTP {resp.status_code}): {resp.text[:1500]}"
                )

            data = resp.json()
            return [
                item["embedding"]
                for item in data["data"]
            ]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class GeminiEmbedding(EmbeddingProvider):
    """Gemini embeddings with smaller batches and paced requests."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
    ):
        self.api_key = api_key
        self._model = model
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        batch_size = 5
        batch_delay_seconds = 2
        all_embeddings: List[List[float]] = []

        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), batch_size):
                # Pace requests to reduce bursts against the API quota.
                if start > 0:
                    time.sleep(batch_delay_seconds)

                batch = texts[start : start + batch_size]

                requests = [
                    {
                        "model": f"models/{self._model}",
                        "content": {
                            "parts": [{"text": text}]
                        },
                        "outputDimensionality": self._dim,
                        "taskType": "RETRIEVAL_DOCUMENT",
                    }
                    for text in batch
                ]

                resp = None
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        resp = client.post(
                            url,
                            headers=headers,
                            json={"requests": requests},
                        )
                    except httpx.HTTPError as exc:
                        if attempt < max_retries:
                            time.sleep((2 ** attempt) + random.uniform(0.2, 0.5))
                            continue
                        raise RuntimeError(
                            "Unable to reach Gemini embedding API: "
                            f"{type(exc).__name__}"
                        ) from None

                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit() and int(retry_after) <= 5 and attempt < max_retries:
                            time.sleep(int(retry_after) + random.uniform(0.1, 0.5))
                            continue
                        raise RuntimeError(
                            "Gemini embedding API quota exceeded (HTTP 429: RESOURCE_EXHAUSTED). "
                            "The Gemini embedding free-tier rate limit (30K TPM / daily quota) has been reached. "
                            "To index repositories completely free without token quotas, configure "
                            "EMBEDDING_PROVIDER=local and EMBEDDING_DIMENSION=384 in your environment."
                        )

                    if resp.status_code in (408, 500, 502, 503, 504) and attempt < max_retries:
                        time.sleep((2 ** attempt) + random.uniform(0.2, 0.5))
                        continue

                    break

                if resp.status_code >= 400:
                    # Never put the API key in the URL or log it.
                    error_detail = resp.text[:1500].replace(self.api_key, "[REDACTED]")
                    raise RuntimeError(
                        "Gemini embedding API error "
                        f"(HTTP {resp.status_code}): {error_detail}"
                    )

                try:
                    data = resp.json()
                    batch_embeddings = [
                        item["values"]
                        for item in data.get("embeddings", [])
                    ]
                except (ValueError, TypeError, KeyError) as exc:
                    raise RuntimeError(
                        "Gemini returned an invalid embedding response: "
                        f"{type(exc).__name__}"
                    ) from None

                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        "Gemini embedding count mismatch: "
                        f"expected {len(batch)}, got {len(batch_embeddings)}"
                    )

                for idx, embedding in enumerate(batch_embeddings):
                    if len(embedding) != self._dim:
                        raise ValueError(
                            "Gemini embedding dimension mismatch: "
                            f"expected {self._dim}, got {len(embedding)} "
                            f"in batch item {idx}"
                        )

                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_documents([text])

        if not results:
            raise ValueError(
                "Gemini embedding returned no vector for query."
            )

        return results[0]


def get_active_embedding_model_name() -> str:
    """
    Returns the active embedding model name based on current settings and provider.
    Guaranteed to return a non-'unknown' active model name for standard providers.
    """
    provider_type = str(getattr(settings, "EMBEDDING_PROVIDER", "")).strip().lower()
    if provider_type == "local":
        return (
            getattr(settings, "LOCAL_EMBEDDING_MODEL", None)
            or getattr(settings, "SENTENCE_TRANSFORMERS_MODEL", None)
            or "all-MiniLM-L6-v2"
        )
    elif provider_type == "gemini":
        return (
            getattr(settings, "GEMINI_EMBEDDING_MODEL", None)
            or "gemini-embedding-001"
        )
    elif provider_type == "openai":
        return (
            getattr(settings, "OPENAI_EMBEDDING_MODEL", None)
            or "text-embedding-3-small"
        )
    elif provider_type in ("onnx_local", "onnx"):
        return (
            getattr(settings, "ONNX_EMBEDDING_MODEL", None)
            or "all-MiniLM-L6-v2-qint8"
        )
    elif provider_type == "mock":
        dim = getattr(settings, "EMBEDDING_DIMENSION", 384)
        return getattr(settings, "MOCK_EMBEDDING_MODEL", None) or f"mock-embedding-{dim}d"
    return getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_embedding_provider() -> EmbeddingProvider:
    provider = str(settings.EMBEDDING_PROVIDER).strip().lower()
    is_production = (
        str(getattr(settings, "APP_ENV", "development"))
        .strip()
        .lower()
        == "production"
    )

    if provider == "local":
        if is_production and _get_sentence_transformer_cls() is None:
            raise RuntimeError(
                "Production embedding configuration error: "
                "EMBEDDING_PROVIDER is 'local', but sentence-transformers is not installed. "
                "Configure EMBEDDING_PROVIDER='gemini' with GEMINI_API_KEY in production."
            )

        model_name = (
            getattr(settings, "LOCAL_EMBEDDING_MODEL", None)
            or getattr(settings, "SENTENCE_TRANSFORMERS_MODEL", None)
            or "all-MiniLM-L6-v2"
        )
        try:
            return LocalEmbedding(model_name=model_name)
        except Exception as exc:
            if is_production:
                raise RuntimeError(
                    "Production embedding configuration error: "
                    f"LocalEmbedding failed to load: {exc}. "
                    "Sentence Transformers must be available."
                ) from exc

            logger.warning(
                "Failed to load LocalEmbedding: %s. "
                "Falling back to MockEmbedding (dim=%s).",
                exc,
                settings.EMBEDDING_DIMENSION,
            )
            return MockEmbedding(
                dim=settings.EMBEDDING_DIMENSION,
                model_name=f"mock-fallback-{settings.EMBEDDING_DIMENSION}d",
            )

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key missing: OPENAI_API_KEY must be provided "
                "when EMBEDDING_PROVIDER='openai'."
            )
        model_name = getattr(settings, "OPENAI_EMBEDDING_MODEL", None) or "text-embedding-3-small"
        return OpenAIEmbedding(settings.OPENAI_API_KEY, model=model_name)

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY must be provided in production."
            )

        dimension = getattr(settings, "EMBEDDING_DIMENSION", 768)
        if dimension != 768:
            logger.warning(
                "Gemini embedding dimension configured as %s; using 768.",
                dimension,
            )
            dimension = 768

        model_name = getattr(settings, "GEMINI_EMBEDDING_MODEL", None) or "gemini-embedding-001"
        return GeminiEmbedding(
            settings.GEMINI_API_KEY,
            model=model_name,
            dimension=dimension,
        )

    if provider in ("onnx_local", "onnx"):
        model_name = (
            getattr(settings, "ONNX_EMBEDDING_MODEL", None)
            or "all-MiniLM-L6-v2-qint8"
        )
        model_dir = getattr(settings, "ONNX_MODEL_DIR", None)
        try:
            return ONNXLocalEmbedding(model_name=model_name, model_dir=model_dir)
        except Exception as exc:
            if is_production:
                raise RuntimeError(
                    "Production embedding configuration error: "
                    f"ONNXLocalEmbedding failed to initialize: {exc}."
                ) from exc
            logger.warning(
                "Failed to load ONNXLocalEmbedding: %s. "
                "Falling back to MockEmbedding (dim=%s).",
                exc,
                settings.EMBEDDING_DIMENSION,
            )
            return MockEmbedding(
                dim=settings.EMBEDDING_DIMENSION,
                model_name=f"mock-fallback-{settings.EMBEDDING_DIMENSION}d",
            )

    if provider != "mock":
        if is_production:
            raise ValueError(
                "Production configuration error: unsupported embedding "
                f"provider '{provider}'."
            )

        logger.warning(
            "Embedding provider '%s' is unsupported. "
            "Falling back to MockEmbedding.",
            provider,
        )

    return MockEmbedding(
        dim=settings.EMBEDDING_DIMENSION,
        model_name=getattr(settings, "MOCK_EMBEDDING_MODEL", None),
    )
