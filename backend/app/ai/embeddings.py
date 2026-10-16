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

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

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
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._dim = 384
        self._batch_size = batch_size

    def _load_model(self):
        global _cached_local_model
        if _cached_local_model is not None:
            return _cached_local_model

        with _local_model_lock:
            if _cached_local_model is not None:
                return _cached_local_model

            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers package is not installed. "
                    "Install sentence-transformers via `pip install sentence-transformers` to use local embeddings."
                )

            logger.info("Initializing local embedding model lazily: %s", self._model_name)
            try:
                _cached_local_model = SentenceTransformer(self._model_name)
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
            embedding = model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as exc:
            logger.error("Local embedding query inference error: %s", exc, exc_info=True)
            raise RuntimeError(f"Local query embedding generation failed for model '{self._model_name}': {exc}") from exc


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
