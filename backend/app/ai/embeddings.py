import os
import sys

user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import hashlib
import numpy as np
from typing import List
from abc import ABC, abstractmethod
import httpx
from app.config import settings
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger("repolens.ai.embeddings")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


class MockEmbedding(EmbeddingProvider):
    """
    Deterministic pseudo-embedding provider for testing/demo.
    """

    def _text_to_vector(self, text: str, dim: int = 128) -> List[float]:
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()

        for idx, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0 / (idx + 1.0)

        norm = np.linalg.norm(vec)

        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)


class LocalEmbedding(EmbeddingProvider):
    """
    Real semantic embeddings generated locally using Sentence Transformers (all-MiniLM-L6-v2).
    Runs completely locally without external API keys.
    """

    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"

        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers package is not installed.")

        logger.info(f"Loading local embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Local embedding model loaded: {self.model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "text-embedding-3-small"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": texts,
            "model": self.model
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents?key={self.api_key}"
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]}
            }
            for t in texts
        ]
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"requests": requests})
            resp.raise_for_status()
            data = resp.json()
            return [e["values"] for e in data.get("embeddings", [])]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider() -> EmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        try:
            return LocalEmbedding()
        except Exception as e:
            logger.warning(f"Failed to load LocalEmbedding: {e}. Falling back to MockEmbedding.")
            return MockEmbedding()

    if provider == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbedding(settings.OPENAI_API_KEY)

    if provider == "gemini" and settings.GEMINI_API_KEY:
        return GeminiEmbedding(settings.GEMINI_API_KEY)

    if provider != "mock":
        logger.warning(f"Embedding provider '{provider}' fallback to MockEmbedding.")

    return MockEmbedding()