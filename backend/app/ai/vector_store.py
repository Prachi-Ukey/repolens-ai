import os
import logging
from typing import List, Dict, Any, Optional

import chromadb

from app.config import settings
from app.ai.embeddings import get_embedding_provider


logger = logging.getLogger("repolens.ai.vector_store")


class ChromaManager:
    """
    Manages ChromaDB vector collections with repository isolation.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = (
            persist_dir or settings.CHROMA_PERSIST_DIRECTORY
        )

        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        self.embedding_provider = get_embedding_provider()

    def _get_collection_name(self, repository_id: str) -> str:
        """
        Converts repository ID into a valid Chroma collection name.
        """

        # Chroma collection names must be 3-63 characters.
        # Replace hyphens with underscores.
        sanitized_id = repository_id.replace("-", "_")

        return f"repo_{sanitized_id}"

    def get_or_create_collection(self, repository_id: str):
        """
        Gets an existing repository collection or creates it.
        """

        col_name = self._get_collection_name(repository_id)

        return self.client.get_or_create_collection(
            name=col_name
        )

    def delete_collection(self, repository_id: str):
        """
        Deletes the Chroma collection for a repository.
        """

        col_name = self._get_collection_name(repository_id)

        try:
            self.client.delete_collection(
                name=col_name
            )

            logger.info(
                f"Deleted Chroma collection: {col_name}"
            )

        except Exception as e:
            logger.warning(
                f"Could not delete collection {col_name}: {e}"
            )

    def upsert_chunks(
        self,
        repository_id: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Generates embeddings and stores repository chunks in ChromaDB.
        """

        if not chunks:
            return

        collection = self.get_or_create_collection(
            repository_id
        )

        documents = [
            chunk["content"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "file_path": chunk["file_path"],
                "language": chunk["language"],
                "start_line": int(chunk["start_line"]),
                "end_line": int(chunk["end_line"]),
                "symbol_name": (
                    chunk.get("symbol_name") or ""
                ),
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        # Generate embeddings.
        embeddings = (
            self.embedding_provider.embed_documents(
                documents
            )
        )

        # Batch upsert in groups of 100.
        batch_size = 100

        for i in range(
            0,
            len(chunks),
            batch_size
        ):
            collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[
                    i:i + batch_size
                ],
                documents=documents[
                    i:i + batch_size
                ],
                metadatas=metadatas[
                    i:i + batch_size
                ],
            )

        logger.info(
            f"Upserted {len(chunks)} chunks to collection "
            f"'{self._get_collection_name(repository_id)}'"
        )

    def query_similar_chunks(
        self,
        repository_id: str,
        query_text: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Queries the repository-isolated Chroma collection
        and returns semantic candidate chunks with distances.
        """

        collection = self.get_or_create_collection(
            repository_id
        )

        if collection.count() == 0:
            return []

        query_embedding = (
            self.embedding_provider.embed_query(
                query_text
            )
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                collection.count()
            ),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            distances = results.get(
                "distances",
                [[]]
            )[0]

            for idx, (doc, meta) in enumerate(
                zip(docs, metas)
            ):
                dist = (
                    distances[idx]
                    if idx < len(distances)
                    else 1.0
                )

                retrieved_chunks.append(
                    {
                        "file_path": meta["file_path"],
                        "language": meta["language"],
                        "start_line": meta["start_line"],
                        "end_line": meta["end_line"],
                        "symbol_name": (
                            meta.get("symbol_name")
                            or None
                        ),
                        "content": doc,
                        "chunk_id": meta.get(
                            "chunk_id",
                            ""
                        ),
                        "distance": dist,
                    }
                )

        return retrieved_chunks


_vector_store_instance = None


def get_vector_store() -> ChromaManager:
    """
    Returns the singleton ChromaManager instance.
    """

    global _vector_store_instance

    if _vector_store_instance is None:
        _vector_store_instance = ChromaManager()

    return _vector_store_instance
