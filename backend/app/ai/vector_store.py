import os
import uuid
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

    def __init__(
        self,
        persist_dir: Optional[str] = None
    ):
        self.persist_dir = (
            persist_dir
            or settings.CHROMA_PERSIST_DIRECTORY
        )

        os.makedirs(
            self.persist_dir,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        self.embedding_provider = (
            get_embedding_provider()
        )

    # ========================================================
    # COLLECTION
    # ========================================================

    def _get_collection_name(
        self,
        repository_id: str
    ) -> str:
        """
        Converts repository ID into a valid Chroma
        collection name.
        """

        # Chroma collection names must be 3-63 characters.
        # Replace hyphens with underscores.
        sanitized_id = repository_id.replace(
            "-",
            "_"
        )

        return f"repo_{sanitized_id}"

    def get_or_create_collection(
        self,
        repository_id: str
    ):
        """
        Gets an existing repository collection
        or creates it with dimension metadata.
        """

        col_name = self._get_collection_name(
            repository_id
        )

        target_dim = getattr(self.embedding_provider, "dimension", 384)
        metadata = {
            "dimension": target_dim,
            "provider": getattr(settings, "EMBEDDING_PROVIDER", "mock"),
            "model": getattr(self.embedding_provider, "model_name", "unknown"),
        }

        return self.client.get_or_create_collection(
            name=col_name,
            metadata=metadata
        )

    def get_collection_dimension(
        self,
        collection
    ) -> Optional[int]:
        """
        Inspects the expected vector dimension of a Chroma collection,
        checking collection metadata or probing existing stored embeddings.
        """
        try:
            meta = collection.metadata or {}
            if "dimension" in meta and meta["dimension"] is not None:
                return int(meta["dimension"])
        except Exception:
            pass

        try:
            if collection.count() > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                embs = sample.get("embeddings")
                if embs is not None and len(embs) > 0 and len(embs[0]) > 0:
                    return len(embs[0])
        except Exception as e:
            logger.debug("Could not inspect collection dimension via sample: %s", e)

        return None

    def safe_replace_collection(
        self,
        repository_id: str,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Safely replaces the Chroma collection for a repository with the provided chunks.
        Ensures that replacement chunks are non-empty, embeddings are successfully generated,
        and the staging collection is fully populated BEFORE touching the existing collection.
        If any step fails, the existing collection is preserved intact.
        """
        if not chunks:
            logger.warning(
                "Cannot replace collection for repo '%s' with empty chunks list; retaining existing collection.",
                repository_id,
            )
            return False

        # Validate that candidate chunks are non-empty and well-formed
        valid_chunks = [
            c for c in chunks
            if c and c.get("content") and str(c.get("content")).strip() and c.get("chunk_id")
        ]
        if not valid_chunks:
            logger.warning(
                "Cannot replace collection for repo '%s': all candidate chunks are empty or invalid; retaining existing collection.",
                repository_id,
            )
            return False

        # Deduplicate chunks by chunk_id while preserving order
        seen_ids = set()
        deduped_chunks = []
        for c in valid_chunks:
            cid = str(c["chunk_id"])
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped_chunks.append(c)
        valid_chunks = deduped_chunks

        documents = [c["content"] for c in valid_chunks]
        ids = [str(c["chunk_id"]) for c in valid_chunks]
        metadatas = [
            {
                "file_path": c.get("file_path", ""),
                "language": c.get("language", ""),
                "start_line": int(c.get("start_line", 1)),
                "end_line": int(c.get("end_line", 1)),
                "symbol_name": c.get("symbol_name") or "",
                "chunk_id": str(c.get("chunk_id")),
                "defined_symbols": c.get("defined_symbols", ""),
                "called_symbols": c.get("called_symbols", ""),
                "imported_symbols": c.get("imported_symbols", ""),
            }
            for c in valid_chunks
        ]

        target_dim = getattr(self.embedding_provider, "dimension", 384)

        # 1. Generate embeddings BEFORE modifying or deleting any existing collection
        try:
            embeddings = self.embedding_provider.embed_documents(documents)
        except Exception as embed_err:
            logger.error(
                "Failed to generate embeddings for repo '%s' replacement: %s; retaining existing collection.",
                repository_id,
                embed_err,
                exc_info=True,
            )
            return False

        if not embeddings or len(embeddings) != len(documents):
            logger.error(
                "Embedding provider generated %s embeddings for %s documents for repo '%s'; retaining existing collection.",
                len(embeddings) if embeddings else 0,
                len(documents),
                repository_id,
            )
            return False

        # Validate vector dimension matches expected target_dim
        if len(embeddings[0]) != target_dim:
            logger.error(
                "Generated embeddings dimension %s does not match expected target dimension %s for repo '%s'; retaining existing collection.",
                len(embeddings[0]),
                target_dim,
                repository_id,
            )
            return False

        target_col_name = self._get_collection_name(repository_id)
        staging_col_name = f"staging_{target_col_name}_{uuid.uuid4().hex[:8]}"
        staging_col = None

        # 2. Create and populate staging collection
        try:
            staging_metadata = {
                "dimension": target_dim,
                "provider": getattr(settings, "EMBEDDING_PROVIDER", "mock"),
                "model": getattr(self.embedding_provider, "model_name", "unknown"),
            }
            staging_col = self.client.create_collection(
                name=staging_col_name,
                metadata=staging_metadata,
            )

            batch_size = 100
            for i in range(0, len(valid_chunks), batch_size):
                staging_col.upsert(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )

            staged_count = staging_col.count()
            if staged_count != len(valid_chunks):
                logger.error(
                    "Staging collection count (%s) does not match expected chunk count (%s) for repo '%s'; retaining existing collection.",
                    staged_count,
                    len(valid_chunks),
                    repository_id,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

        except Exception as stage_err:
            logger.error(
                "Failed to populate staging collection for repo '%s': %s; retaining existing collection.",
                repository_id,
                stage_err,
                exc_info=True,
            )
            if staging_col is not None:
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
            return False

        # 3. Safe replacement of target collection using staging collection
        existing_col = None
        try:
            existing_col = self.client.get_collection(name=target_col_name)
        except Exception:
            existing_col = None

        if existing_col is not None:
            if not hasattr(existing_col, "modify") or not hasattr(staging_col, "modify"):
                logger.warning(
                    "Collection modification/rename is not supported by Chroma client for repo '%s'; retaining original collection.",
                    repository_id,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

            backup_col_name = f"backup_{target_col_name}_{uuid.uuid4().hex[:8]}"
            try:
                # Rename existing collection to temporary backup
                existing_col.modify(name=backup_col_name)
            except Exception as modify_err:
                logger.warning(
                    "Failed to stage existing collection for repo '%s': %s; retaining original collection.",
                    repository_id,
                    modify_err,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

            try:
                # Promote staging collection to target collection name
                staging_col.modify(name=target_col_name)
            except Exception as rename_err:
                logger.error(
                    "Failed to promote staging collection to '%s': %s. Rolling back to original collection.",
                    target_col_name,
                    rename_err,
                )
                try:
                    existing_col.modify(name=target_col_name)
                except Exception as rb_err:
                    logger.critical(
                        "CRITICAL: Failed to rollback backup collection '%s' to '%s': %s",
                        backup_col_name,
                        target_col_name,
                        rb_err,
                    )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

            # Delete the backup collection now that staging collection is safely active
            try:
                self.client.delete_collection(name=backup_col_name)
            except Exception as cleanup_err:
                logger.warning(
                    "Failed to delete temporary backup collection '%s': %s",
                    backup_col_name,
                    cleanup_err,
                )
        else:
            # Target collection does not exist yet; promote staging directly
            if not hasattr(staging_col, "modify"):
                logger.warning(
                    "Collection modification/rename is not supported by Chroma client for repo '%s'; retaining original state.",
                    repository_id,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

            try:
                staging_col.modify(name=target_col_name)
            except Exception as promote_err:
                logger.error(
                    "Failed to promote staging collection to '%s': %s",
                    target_col_name,
                    promote_err,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

        logger.info(
            "Safely replaced Chroma collection '%s' with %s chunks.",
            target_col_name,
            len(valid_chunks),
        )
        return True

    def rebuild_collection_from_db(
        self,
        repository_id: str
    ) -> bool:
        """
        Safely rebuilds the ChromaDB vector collection for a specific repository
        using stored database chunks or files with the active embedding provider.
        Preserves user accounts, repository metadata, conversations, and other records.
        Never deletes an existing Chroma collection unless replacement chunks have
        been successfully reconstructed, validated, and staged.
        """
        from app.database import SessionLocal
        from app.models.repository import Repository
        from app.models.file import RepositoryFile
        from app.models.chunk import CodeChunk

        active_model = getattr(self.embedding_provider, "model_name", "unknown")
        active_dim = getattr(self.embedding_provider, "dimension", 384)
        logger.info(
            "Rebuilding vector collection for repository '%s' (active provider: %s, dim: %s)...",
            repository_id,
            active_model,
            active_dim,
        )

        db = SessionLocal()
        try:
            repo = db.query(Repository).filter(Repository.id == repository_id).first()
            if not repo:
                logger.warning(
                    "Repository '%s' not found in database; cannot rebuild vector collection.",
                    repository_id,
                )
                return False

            file_count = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).count()
            chunk_count = db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).count()

            if file_count > 0:
                from app.github.parser import is_ignored_path, parse_file_into_chunks
                files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).all()
                valid_files = [f for f in files if not is_ignored_path(f.file_path)]
                all_chunks = []
                for f in valid_files:
                    file_chunks = parse_file_into_chunks(
                        file_path=f.file_path,
                        content=f.content,
                        language=f.language
                    )
                    all_chunks.extend(file_chunks)

                # Validate reconstructed chunks from stored files
                if not all_chunks:
                    logger.warning(
                        "Parsing stored files for repository '%s' produced zero chunks; retaining existing collection.",
                        repo.id,
                    )
                    return False

                success = self.safe_replace_collection(repo.id, all_chunks)
                if success:
                    logger.info(
                        "Successfully rebuilt collection for repo '%s' from %s stored files (%s chunks).",
                        repo.id,
                        len(valid_files),
                        len(all_chunks),
                    )
                    return True
                else:
                    logger.error(
                        "Failed to safely replace collection for repo '%s'; retaining existing collection.",
                        repo.id,
                    )
                    return False

            elif chunk_count > 0:
                stored_chunks = db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).all()
                chunks_payload = [
                    {
                        "chunk_id": c.chunk_id,
                        "file_path": c.file_path,
                        "language": c.language,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "symbol_name": c.symbol_name or "",
                        "content": c.content,
                        "defined_symbols": "",
                        "called_symbols": "",
                        "imported_symbols": "",
                    }
                    for c in stored_chunks
                    if c.content and c.content.strip() and c.chunk_id
                ]

                # Validate reconstructed chunks from CodeChunk records
                if not chunks_payload:
                    logger.warning(
                        "Stored CodeChunk records for repository '%s' produced zero usable chunks; retaining existing collection.",
                        repo.id,
                    )
                    return False

                success = self.safe_replace_collection(repo.id, chunks_payload)
                if success:
                    logger.info(
                        "Successfully rebuilt collection for repo '%s' from %s stored CodeChunk records.",
                        repo.id,
                        len(chunks_payload),
                    )
                    return True
                else:
                    logger.error(
                        "Failed to safely replace collection for repo '%s'; retaining existing collection.",
                        repo.id,
                    )
                    return False

            else:
                logger.warning(
                    "No files or chunks found in database for repository '%s'; retaining existing collection.",
                    repo.id,
                )
                return False

        except Exception as e:
            logger.error(
                "Error while rebuilding vector collection for repo '%s': %s",
                repository_id,
                e,
                exc_info=True,
            )
            return False
        finally:
            if 'db' in locals() and db.is_active:
                db.close()

    def delete_collection(
        self,
        repository_id: str
    ):
        """
        Deletes the Chroma collection for a repository.
        """

        col_name = self._get_collection_name(
            repository_id
        )

        try:

            self.client.delete_collection(
                name=col_name
            )

            logger.info(
                "Deleted Chroma collection: %s",
                col_name
            )

        except Exception as e:

            logger.warning(
                "Could not delete collection %s: %s",
                col_name,
                e
            )

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert_chunks(
        self,
        repository_id: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Generates embeddings and stores repository
        chunks in ChromaDB.
        """

        if not chunks:
            return

        collection = self.get_or_create_collection(
            repository_id
        )

        existing_dim = self.get_collection_dimension(collection)
        target_dim = getattr(self.embedding_provider, "dimension", 384)
        if existing_dim is not None and existing_dim != target_dim:
            logger.warning(
                "Collection '%s' dimension mismatch during upsert (expected %s, provider generates %s). Safely replacing collection...",
                self._get_collection_name(repository_id),
                existing_dim,
                target_dim,
            )
            success = self.safe_replace_collection(repository_id, chunks)
            if not success:
                logger.error(
                    "Failed to safely replace collection for repo '%s' during upsert dimension mismatch.",
                    repository_id,
                )
            return

        documents = [
            chunk["content"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "file_path": chunk["file_path"],
                "language": chunk["language"],
                "start_line": int(
                    chunk["start_line"]
                ),
                "end_line": int(
                    chunk["end_line"]
                ),
                "symbol_name": (
                    chunk.get("symbol_name")
                    or ""
                ),
                "chunk_id": chunk["chunk_id"],
                "defined_symbols": chunk.get("defined_symbols", ""),
                "called_symbols": chunk.get("called_symbols", ""),
                "imported_symbols": chunk.get("imported_symbols", ""),
            }
            for chunk in chunks
        ]

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        # -----------------------------------------------------
        # Generate embeddings
        # -----------------------------------------------------

        embeddings = (
            self.embedding_provider.embed_documents(
                documents
            )
        )

        # -----------------------------------------------------
        # Batch upsert
        # -----------------------------------------------------

        batch_size = 100

        for i in range(
            0,
            len(chunks),
            batch_size
        ):

            collection.upsert(
                ids=ids[
                    i:i + batch_size
                ],
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
            "Upserted %s chunks to collection '%s'",
            len(chunks),
            self._get_collection_name(
                repository_id
            )
        )

    # ========================================================
    # QUERY
    # ========================================================

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

        collection_count = collection.count()

        if collection_count == 0:

            logger.warning(
                "Chroma collection is empty | repo=%s",
                repository_id
            )

            return []

        # -----------------------------------------------------
        # Check Dimension Compatibility
        # -----------------------------------------------------

        expected_dim = self.get_collection_dimension(collection)
        actual_dim = getattr(self.embedding_provider, "dimension", 384)

        if expected_dim is not None and expected_dim != actual_dim:
            logger.warning(
                "Chroma collection dimension mismatch for repo %s: collection expects %s, but active provider %s generates %s. Rebuilding collection from stored chunks...",
                repository_id,
                expected_dim,
                getattr(self.embedding_provider, "model_name", "unknown"),
                actual_dim,
            )
            rebuilt = self.rebuild_collection_from_db(repository_id)
            if rebuilt:
                collection = self.get_or_create_collection(repository_id)
                collection_count = collection.count()
                if collection_count == 0:
                    return []
            else:
                logger.error(
                    "Automatic rebuild of collection for repo %s failed or had no data; querying existing collection will likely fail.",
                    repository_id,
                )

        # -----------------------------------------------------
        # Generate query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_provider.embed_query(
                query_text
            )
        )

        # -----------------------------------------------------
        # Query ChromaDB with defensive recovery
        # -----------------------------------------------------

        try:
            results = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=min(
                    top_k,
                    collection_count
                ),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        except Exception as e:
            err_msg = str(e)
            if "dimension" in err_msg.lower() or "expecting embedding with dimension" in err_msg.lower():
                logger.warning(
                    "Caught Chroma dimension error for repo %s: %s. Attempting collection rebuild recovery...",
                    repository_id,
                    err_msg,
                )
                rebuilt = self.rebuild_collection_from_db(repository_id)
                if rebuilt:
                    collection = self.get_or_create_collection(repository_id)
                    collection_count = collection.count()
                    if collection_count == 0:
                        return []
                    results = collection.query(
                        query_embeddings=[
                            query_embedding
                        ],
                        n_results=min(
                            top_k,
                            collection_count
                        ),
                        include=[
                            "documents",
                            "metadatas",
                            "distances",
                        ],
                    )
                else:
                    raise
            else:
                raise

        retrieved_chunks = []

        if not results:

            return retrieved_chunks

        documents = results.get(
            "documents"
        )

        metadatas = results.get(
            "metadatas"
        )

        distances = results.get(
            "distances"
        )

        if not documents or not metadatas:

            return retrieved_chunks

        docs = documents[0]
        metas = metadatas[0]

        if distances:
            distance_list = distances[0]
        else:
            distance_list = []

        # =====================================================
        # TEMPORARY VECTOR STORE DEBUG
        # =====================================================

        print(
            "\n========== CHROMA QUERY DEBUG =========="
        )

        print(
            f"Collection: "
            f"{self._get_collection_name(repository_id)}"
        )

        print(
            f"Collection count: {collection_count}"
        )

        print(
            f"Query: {query_text}"
        )

        print(
            f"Returned chunks: {len(docs)}"
        )

        print(
            "----------------------------------------"
        )

        for idx, meta in enumerate(
            metas,
            start=1
        ):

            distance = (
                distance_list[idx - 1]
                if idx - 1 < len(
                    distance_list
                )
                else None
            )

            print(
                f"{idx}. "
                f"FILE: {meta.get('file_path')} | "
                f"LINES: "
                f"{meta.get('start_line')}-"
                f"{meta.get('end_line')} | "
                f"DISTANCE: {distance} | "
                f"SYMBOL: "
                f"{meta.get('symbol_name')}"
            )

        print(
            "========================================\n"
        )

        # =====================================================
        # Build retrieved chunks
        # =====================================================

        for idx, (doc, meta) in enumerate(
            zip(
                docs,
                metas
            )
        ):

            dist = (
                distance_list[idx]
                if idx < len(
                    distance_list
                )
                else 1.0
            )

            retrieved_chunks.append(
                {
                    "file_path": meta.get(
                        "file_path",
                        ""
                    ),
                    "language": meta.get(
                        "language",
                        ""
                    ),
                    "start_line": meta.get(
                        "start_line",
                        1
                    ),
                    "end_line": meta.get(
                        "end_line",
                        1
                    ),
                    "symbol_name": (
                        meta.get(
                            "symbol_name"
                        )
                        or None
                    ),
                    "content": doc,
                    "chunk_id": meta.get(
                        "chunk_id",
                        ""
                    ),
                    "defined_symbols": meta.get("defined_symbols", ""),
                    "called_symbols": meta.get("called_symbols", ""),
                    "imported_symbols": meta.get("imported_symbols", ""),
                    "distance": dist,
                }
            )

        return retrieved_chunks


# ============================================================
# SINGLETON VECTOR STORE
# ============================================================

_vector_store_instance = None


def get_vector_store() -> ChromaManager:
    """
    Returns the singleton ChromaManager instance.
    """

    global _vector_store_instance

    if _vector_store_instance is None:

        _vector_store_instance = (
            ChromaManager()
        )

    return _vector_store_instance