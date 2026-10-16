import os
import sys

user_site = os.path.expanduser(
    "~/AppData/Roaming/Python/Python313/site-packages"
)
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

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

    def __init__(self, dim: Optional[int] = None):
        self._dim = (
            dim
            if dim is not None
            else getattr(settings, "EMBEDDING_DIMENSION", 384)
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"mock-embedding-{self._dim}d"

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


class LocalEmbedding(EmbeddingProvider):
    """Semantic embeddings generated locally using Sentence Transformers."""

    def __init__(self):
        self._model_name = "all-MiniLM-L6-v2"
        self._dim = 384

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers package is not installed."
            )

        logger.info("Loading local embedding model: %s", self._model_name)
        self.model = SentenceTransformer(self._model_name)
        logger.info("Local embedding model loaded: %s", self._model_name)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._model = "text-embedding-3-small"
        self._dim = 1536

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
    """Gemini embeddings with bounded request batches and safe error reporting."""

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

        batch_size = 20
        all_embeddings: List[List[float]] = []

        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), batch_size):
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

                try:
                    resp = client.post(
                        url,
                        headers=headers,
                        json={"requests": requests},
                    )
                except httpx.HTTPError as exc:
                    raise RuntimeError(
                        "Unable to reach Gemini embedding API: "
                        f"{type(exc).__name__}"
                    ) from None

                if resp.status_code >= 400:
                    # The key is not included in the URL or error message.
                    error_detail = resp.text[:1500]
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


def get_embedding_provider() -> EmbeddingProvider:
    provider = str(settings.EMBEDDING_PROVIDER).strip().lower()
    is_production = (
        str(getattr(settings, "APP_ENV", "development"))
        .strip()
        .lower()
        == "production"
    )

    if provider == "local":
        try:
            return LocalEmbedding()
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
            return MockEmbedding(dim=settings.EMBEDDING_DIMENSION)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key missing: OPENAI_API_KEY must be provided "
                "when EMBEDDING_PROVIDER='openai'."
            )
        return OpenAIEmbedding(settings.OPENAI_API_KEY)

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key missing: GEMINI_API_KEY must be provided "
                "when EMBEDDING_PROVIDER='gemini'."
            )

        dimension = getattr(settings, "EMBEDDING_DIMENSION", 768)
        if dimension != 768:
            logger.warning(
                "Gemini embedding dimension configured as %s; using 768.",
                dimension,
            )
            dimension = 768

        return GeminiEmbedding(
            settings.GEMINI_API_KEY,
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

    return MockEmbedding(dim=settings.EMBEDDING_DIMENSION)