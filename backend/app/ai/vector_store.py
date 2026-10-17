import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable

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
        self._pending_backups: Dict[str, Dict[str, Any]] = {}

    # ========================================================
    # EMBEDDING IDENTITY & COMPATIBILITY
    # ========================================================

    @property
    def embedding_model_name(self) -> str:
        """
        Public property returning the active embedding model name.
        Uses the active provider's model_name, with backward-compatible fallbacks
        to common model attributes or provider-specific configuration.
        """
        provider = getattr(self, "embedding_provider", None)
        if provider is not None:
            name = getattr(provider, "model_name", None)
            if callable(name):
                try:
                    name = name()
                except Exception:
                    name = None
            if name and str(name).strip() and str(name).strip().lower() != "unknown":
                return str(name).strip()

            for attr in ("model", "_model_name", "_model"):
                val = getattr(provider, attr, None)
                if val:
                    if callable(val):
                        try:
                            val = val()
                        except Exception:
                            continue
                    if isinstance(val, str) and val.strip() and val.strip().lower() != "unknown":
                        return val.strip()

        from app.ai.embeddings import get_active_embedding_model_name
        return get_active_embedding_model_name()

    @property
    def model_name(self) -> str:
        """Alias for embedding_model_name."""
        return self.embedding_model_name

    @property
    def embedding_dimension(self) -> int:
        """
        Public property returning the active embedding dimension.
        """
        provider = getattr(self, "embedding_provider", None)
        if provider is not None:
            dim = getattr(provider, "dimension", None)
            if callable(dim):
                try:
                    dim = dim()
                except Exception:
                    dim = None
            if dim is not None:
                return int(dim)
            dim = getattr(provider, "_dim", None)
            if dim is not None:
                return int(dim)
        return int(getattr(settings, "EMBEDDING_DIMENSION", 384))

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

        target_dim = self.embedding_dimension
        metadata = {
            "dimension": target_dim,
            "provider": getattr(settings, "EMBEDDING_PROVIDER", "mock"),
            "model": self.embedding_model_name,
        }

        return self.client.get_or_create_collection(
            name=col_name,
            metadata=metadata
        )

    def _copy_collection_data(
        self,
        src_col,
        dest_col
    ) -> bool:
        """
        Safely copies all IDs, embeddings, documents, and metadatas
        from src_col into dest_col in bounded batches using pagination.
        Strictly validates that non-empty source collections have complete,
        usable embeddings and consistent IDs, embeddings, documents, and metadata.
        Never reports success if a non-empty collection lacks vectors.
        """
        try:
            total_items = src_col.count()
            if total_items == 0:
                return dest_col.count() == 0

            src_dim = self.get_collection_dimension(src_col)
            if src_dim is None or src_dim <= 0:
                logger.error(
                    "Copy aborted: source collection '%s' has %s items but lacks valid vector dimension.",
                    getattr(src_col, "name", "unknown"),
                    total_items,
                )
                return False

            batch_size = 100
            offset = 0

            while offset < total_items:
                data = src_col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "documents", "metadatas"],
                )
                ids = data.get("ids", [])
                if not ids:
                    logger.error(
                        "Source collection '%s' returned empty IDs at offset %s while count is %s.",
                        getattr(src_col, "name", "unknown"),
                        offset,
                        total_items,
                    )
                    return False

                batch_len = len(ids)

                raw_embs = data.get("embeddings")
                if raw_embs is None:
                    logger.error(
                        "Source collection '%s' lacks embeddings at offset %s (got %s IDs).",
                        getattr(src_col, "name", "unknown"),
                        offset,
                        batch_len,
                    )
                    return False

                if hasattr(raw_embs, "tolist"):
                    embs = raw_embs.tolist()
                else:
                    embs = list(raw_embs) if raw_embs is not None else []

                if len(embs) != batch_len:
                    logger.error(
                        "Source collection '%s' embedding count mismatch at offset %s: expected %s, got %s.",
                        getattr(src_col, "name", "unknown"),
                        offset,
                        batch_len,
                        len(embs),
                    )
                    return False

                for idx, emb in enumerate(embs):
                    if emb is None or len(emb) != src_dim:
                        logger.error(
                            "Source collection '%s' invalid vector at offset %s (index %s): expected dim %s, got %s.",
                            getattr(src_col, "name", "unknown"),
                            offset,
                            idx,
                            src_dim,
                            len(emb) if emb is not None else None,
                        )
                        return False

                docs = data.get("documents")
                if docs is not None and len(docs) != batch_len:
                    logger.error(
                        "Source collection '%s' document count mismatch at offset %s: expected %s, got %s.",
                        getattr(src_col, "name", "unknown"),
                        offset,
                        batch_len,
                        len(docs),
                    )
                    return False

                metas = data.get("metadatas")
                if metas is not None and len(metas) != batch_len:
                    logger.error(
                        "Source collection '%s' metadata count mismatch at offset %s: expected %s, got %s.",
                        getattr(src_col, "name", "unknown"),
                        offset,
                        batch_len,
                        len(metas),
                    )
                    return False

                add_kwargs = {
                    "ids": ids,
                    "embeddings": embs,
                }
                if docs is not None:
                    add_kwargs["documents"] = docs
                if metas is not None:
                    add_kwargs["metadatas"] = metas

                dest_col.add(**add_kwargs)
                offset += batch_len

            # Verify destination count
            if dest_col.count() != total_items:
                logger.error(
                    "Copy failed: destination count (%s) does not match source count (%s).",
                    dest_col.count(),
                    total_items,
                )
                return False

            # Verify vector dimensions
            dest_dim = self.get_collection_dimension(dest_col)
            if dest_dim is None or dest_dim != src_dim:
                logger.error(
                    "Copy failed: destination dimension (%s) does not match source dimension (%s).",
                    dest_dim,
                    src_dim,
                )
                return False

            return True
        except Exception as copy_err:
            logger.error("Failed to copy collection data: %s", copy_err, exc_info=True)
            return False

    def get_collection_dimension(
        self,
        collection
    ) -> Optional[int]:
        """
        Inspects the true vector dimension of a Chroma collection.
        PRIORITIZES actual stored embeddings over metadata to avoid stale metadata bugs.
        If the collection has no vectors, checks metadata as a secondary hint.
        """
        # 1. First probe actual stored vectors (ground truth)
        try:
            if collection.count() > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                embs = sample.get("embeddings")
                if embs is not None and len(embs) > 0 and len(embs[0]) > 0:
                    actual_dim = len(embs[0])
                    # Synchronize metadata if it became stale
                    meta = collection.metadata or {}
                    if meta.get("dimension") != actual_dim and hasattr(collection, "modify"):
                        try:
                            new_meta = dict(meta)
                            new_meta["dimension"] = actual_dim
                            collection.modify(metadata=new_meta)
                        except Exception:
                            pass
                    return actual_dim
        except Exception as e:
            logger.debug("Could not inspect collection dimension via sample: %s", e)

        # 2. If collection has no vectors, metadata is checked as secondary fallback
        try:
            meta = collection.metadata or {}
            if "dimension" in meta and meta["dimension"] is not None:
                return int(meta["dimension"])
        except Exception:
            pass

        return None

    def is_collection_compatible(
        self,
        collection,
        repository_id: Optional[str] = None,
        db_repo: Optional[Any] = None,
    ) -> bool:
        """
        Determines whether an existing Chroma collection is compatible with the
        active embedding provider in both vector dimension and embedding model identity.
        Returns False if:
        - Collection has vectors and its dimension does not match active embedding_dimension.
        - Collection or database has model metadata that does not match active embedding_model_name.
        - Collection has vectors but neither collection metadata nor database record provides
          verifiable model metadata (legacy collection with missing/unknown metadata).
        Empty collections (count == 0) are considered compatible (no vectors to conflict).
        """
        try:
            col_count = collection.count()
        except Exception:
            col_count = 0

        if col_count == 0:
            return True

        # 1. Dimension compatibility
        existing_dim = self.get_collection_dimension(collection)
        active_dim = self.embedding_dimension
        if existing_dim is None or existing_dim != active_dim:
            return False

        # 2. Model identity compatibility
        active_model = self.embedding_model_name
        if not active_model or str(active_model).strip().lower() == "unknown":
            return False

        meta = collection.metadata or {}
        col_model = meta.get("model")

        if db_repo is None and repository_id is not None:
            try:
                from app.database import SessionLocal
                from app.models.repository import Repository
                db = SessionLocal()
                try:
                    db_repo = db.query(Repository).filter(Repository.id == repository_id).first()
                finally:
                    db.close()
            except Exception as db_err:
                logger.debug("Could not lookup repository %s in DB: %s", repository_id, db_err)

        db_model = getattr(db_repo, "embedding_model", None) if db_repo else None

        # Explicitly treat 'unknown' in either collection or DB record as unverified:
        # never assume an unknown model matches the active model.
        if col_model and str(col_model).strip().lower() == "unknown":
            return False
        if db_model and str(db_model).strip().lower() == "unknown":
            return False

        has_col_model = bool(col_model and str(col_model).strip() and str(col_model).strip().lower() != "unknown")
        has_db_model = bool(db_model and str(db_model).strip() and str(db_model).strip().lower() != "unknown")

        # If reliable model metadata exists, verify compatibility
        if has_col_model:
            if str(col_model).strip() != active_model:
                return False
            if has_db_model and str(db_model).strip() != active_model:
                return False
            return True

        if has_db_model:
            if str(db_model).strip() != active_model:
                return False
            return True

        # Neither collection metadata nor DB record has verifiable model metadata.
        # Treat missing legacy metadata carefully; do not claim an unknown model is compatible without evidence.
        return False

    def safe_replace_collection(
        self,
        repository_id: str,
        chunks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        deferred_cleanup: bool = False,
    ) -> bool:
        """
        Safely replaces the Chroma collection for a repository with the provided chunks.
        Ensures that:
        1. Replacement chunks are non-empty and embeddings are verified BEFORE touching anything.
        2. Staging collection is strictly verified for exact count AND confirmed target dimension.
        3. If renaming is supported by the ChromaDB client, uses atomic rename with verified rollback.
        4. If renaming is NOT supported, creates and verifies backup copy against original before replacing,
           and rolls back from backup if replacement fails. If backup verification fails, returns False
           WITHOUT touching the original collection.
        5. Failed rollback attempts never delete the backup, and log the exact backup name at CRITICAL level.
        """
        if not chunks:
            logger.warning(
                "Cannot replace collection for repo '%s' with empty chunks list; retaining existing collection.",
                repository_id,
            )
            return False

        # Validate and deduplicate candidate chunks in a single pass to avoid duplicate full-size lists
        seen_ids = set()
        valid_chunks = []
        for c in chunks:
            if c and c.get("content") and str(c.get("content")).strip() and c.get("chunk_id"):
                cid = str(c["chunk_id"])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    valid_chunks.append(c)
        del seen_ids

        if not valid_chunks:
            logger.warning(
                "Cannot replace collection for repo '%s': all candidate chunks are empty or invalid; retaining existing collection.",
                repository_id,
            )
            return False

        target_dim = self.embedding_dimension
        active_model = self.embedding_model_name

        target_col_name = self._get_collection_name(repository_id)
        staging_col_name = f"staging_{target_col_name}_{uuid.uuid4().hex[:8]}"
        staging_col = None

        # 1. Create and populate staging collection in bounded batches
        # This keeps peak memory bounded to batch_size chunks, avoiding storing all vectors in Python lists at once.
        staging_metadata = {
            "dimension": target_dim,
            "provider": getattr(settings, "EMBEDDING_PROVIDER", "local"),
            "model": active_model,
        }
        raw_upsert_batch = os.environ.get("CHROMA_UPSERT_BATCH_SIZE")
        upsert_batch_size = 16
        if raw_upsert_batch is not None:
            try:
                parsed_batch = int(str(raw_upsert_batch).strip())
                if parsed_batch > 0:
                    upsert_batch_size = parsed_batch
            except (ValueError, TypeError):
                pass
        try:
            staging_col = self.client.create_collection(
                name=staging_col_name,
                metadata=staging_metadata,
            )

            for i in range(0, len(valid_chunks), upsert_batch_size):
                batch_chunks = valid_chunks[i : i + upsert_batch_size]
                batch_docs = [c["content"] for c in batch_chunks]
                batch_ids = [str(c["chunk_id"]) for c in batch_chunks]
                batch_metas = [
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
                    for c in batch_chunks
                ]

                try:
                    batch_embs = self.embedding_provider.embed_documents(batch_docs)
                except Exception as embed_err:
                    logger.error(
                        "Failed to generate embeddings for repo '%s' batch: %s; retaining existing collection.",
                        repository_id,
                        embed_err,
                        exc_info=True,
                    )
                    raise

                if not batch_embs or len(batch_embs) != len(batch_docs):
                    raise RuntimeError(
                        f"Embedding provider generated {len(batch_embs) if batch_embs else 0} embeddings for {len(batch_docs)} documents for repo '{repository_id}'."
                    )

                if len(batch_embs[0]) != target_dim:
                    raise RuntimeError(
                        f"Generated embeddings dimension {len(batch_embs[0])} does not match expected target dimension {target_dim} for repo '{repository_id}'."
                    )

                staging_col.upsert(
                    ids=batch_ids,
                    embeddings=batch_embs,
                    documents=batch_docs,
                    metadatas=batch_metas,
                )
                del batch_docs, batch_ids, batch_metas, batch_embs

                if progress_callback is not None:
                    try:
                        progress_callback(min(i + upsert_batch_size, len(valid_chunks)), len(valid_chunks))
                    except Exception as cb_err:
                        logger.warning(
                            "Embedding progress callback failed for repo '%s': %s",
                            repository_id,
                            cb_err,
                        )

            # 2. Require strict staging verification (Requirement 1)
            # Must pass exact expected chunk count and confirmed vector dimension equal to target_dim
            staged_count = staging_col.count()
            staged_dim = self.get_collection_dimension(staging_col)
            if staged_count != len(valid_chunks) or staged_dim is None or staged_dim != target_dim:
                logger.error(
                    "Staging collection strict verification failed for repo '%s' "
                    "(expected count=%s, got %s; expected dim=%s, got %s); retaining existing collection.",
                    repository_id,
                    len(valid_chunks),
                    staged_count,
                    target_dim,
                    staged_dim,
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

        # 4. Check for existing target collection
        existing_col = None
        try:
            existing_col = self.client.get_collection(name=target_col_name)
        except Exception:
            existing_col = None

        # -------------------------------------------------------------
        # Branch A: Target collection does not exist yet
        # -------------------------------------------------------------
        if existing_col is None:
            if hasattr(staging_col, "modify"):
                try:
                    staging_col.modify(name=target_col_name)
                    # Verify promoted collection
                    promoted_col = self.client.get_collection(name=target_col_name)
                    promoted_dim = self.get_collection_dimension(promoted_col)
                    if promoted_col.count() == len(valid_chunks) and promoted_dim is not None and promoted_dim == target_dim:
                        if deferred_cleanup:
                            self._pending_backups[repository_id] = {
                                "backup_col_name": None,
                                "target_col_name": target_col_name,
                                "orig_count": 0,
                                "orig_dim": None,
                                "rename_supported": True,
                            }
                        logger.info("Created Chroma collection '%s' with %s chunks.", target_col_name, len(valid_chunks))
                        return True
                    else:
                        logger.error("Failed verification for newly promoted collection '%s'.", target_col_name)
                        try:
                            self.client.delete_collection(name=target_col_name)
                        except Exception:
                            pass
                        return False
                except Exception as promote_err:
                    logger.error("Failed to promote staging collection to '%s': %s", target_col_name, promote_err)
                    try:
                        self.client.delete_collection(name=staging_col_name)
                    except Exception:
                        pass
                    return False
            else:
                try:
                    target_col = self.client.create_collection(
                        name=target_col_name,
                        metadata=staging_metadata
                    )
                    copy_ok = self._copy_collection_data(staging_col, target_col)
                    try:
                        self.client.delete_collection(name=staging_col_name)
                    except Exception:
                        pass
                    target_dim_found = self.get_collection_dimension(target_col)
                    if copy_ok and target_col.count() == len(valid_chunks) and target_dim_found is not None and target_dim_found == target_dim:
                        if deferred_cleanup:
                            self._pending_backups[repository_id] = {
                                "backup_col_name": None,
                                "target_col_name": target_col_name,
                                "orig_count": 0,
                                "orig_dim": None,
                                "rename_supported": False,
                            }
                        logger.info("Created Chroma collection '%s' with %s chunks via verified copy.", target_col_name, len(valid_chunks))
                        return True
                    else:
                        logger.error("Failed verification for newly copied collection '%s'.", target_col_name)
                        try:
                            self.client.delete_collection(name=target_col_name)
                        except Exception:
                            pass
                        return False
                except Exception as create_err:
                    logger.error("Failed to create collection '%s': %s", target_col_name, create_err, exc_info=True)
                    try:
                        self.client.delete_collection(name=staging_col_name)
                    except Exception:
                        pass
                    return False

        # -------------------------------------------------------------
        # Branch B: Target collection exists — perform safe replacement
        # -------------------------------------------------------------
        orig_count = existing_col.count()
        orig_dim = self.get_collection_dimension(existing_col)
        rename_supported = callable(getattr(existing_col, "modify", None)) and callable(getattr(staging_col, "modify", None))

        # ---------------------------------------------------------
        # Path B1: Client supports collection renaming
        # ---------------------------------------------------------
        if rename_supported:
            backup_col_name = f"backup_{target_col_name}_{uuid.uuid4().hex[:8]}"
            rename_initiated = False
            try:
                # Rename existing collection to temporary backup
                existing_col.modify(name=backup_col_name)
                rename_initiated = True
            except (AttributeError, NotImplementedError) as unsupported_err:
                logger.info(
                    "Collection modification/rename not supported on Chroma client (%s); falling back to verified backup-and-restore.",
                    unsupported_err,
                )
            except Exception as modify_err:
                logger.error(
                    "Failed to stage existing collection for repo '%s' to backup '%s': %s; retaining original collection.",
                    repository_id,
                    backup_col_name,
                    modify_err,
                )
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                return False

            if rename_initiated:
                promoted_successfully = False
                try:
                    # Promote staging collection to target collection name
                    staging_col.modify(name=target_col_name)
                    # Verify replacement collection count and vector dimensions
                    promoted_col = self.client.get_collection(name=target_col_name)
                    promoted_count = promoted_col.count()
                    promoted_dim = self.get_collection_dimension(promoted_col)
                    if promoted_count == len(valid_chunks) and promoted_dim is not None and promoted_dim == target_dim:
                        promoted_successfully = True
                    else:
                        logger.error(
                            "Promoted collection '%s' failed verification (count: %s vs %s, dim: %s vs %s). Initiating rollback.",
                            target_col_name,
                            promoted_count,
                            len(valid_chunks),
                            promoted_dim,
                            target_dim,
                        )
                except Exception as rename_err:
                    logger.error(
                        "Failed to promote staging collection to '%s': %s. Initiating rollback to original collection.",
                        target_col_name,
                        rename_err,
                    )

                if promoted_successfully:
                    if deferred_cleanup:
                        self._pending_backups[repository_id] = {
                            "backup_col_name": backup_col_name,
                            "target_col_name": target_col_name,
                            "orig_count": orig_count,
                            "orig_dim": orig_dim,
                            "rename_supported": True,
                        }
                        logger.info(
                            "Safely replaced Chroma collection '%s' with %s chunks (%s-d); backup '%s' retained pending database commit.",
                            target_col_name,
                            len(valid_chunks),
                            target_dim,
                            backup_col_name,
                        )
                        return True

                    # Delete backup collection only after promotion & verification succeed
                    try:
                        self.client.delete_collection(name=backup_col_name)
                    except Exception as cleanup_err:
                        logger.warning("Non-fatal: Failed to delete temporary backup collection '%s': %s", backup_col_name, cleanup_err)
                    logger.info("Safely replaced Chroma collection '%s' with %s chunks (%s-d).", target_col_name, len(valid_chunks), target_dim)
                    return True
                else:
                    # ROLLBACK to original collection (Requirement 3 & 4)
                    # First delete target_col_name if staging took it over but failed verification
                    try:
                        col_at_target = self.client.get_collection(name=target_col_name)
                        if hasattr(staging_col, "id") and hasattr(col_at_target, "id") and col_at_target.id == staging_col.id:
                            self.client.delete_collection(name=target_col_name)
                    except Exception:
                        pass
                    try:
                        self.client.delete_collection(name=staging_col_name)
                    except Exception:
                        pass

                    # Restore backup collection to target_col_name
                    try:
                        backup_col = self.client.get_collection(name=backup_col_name)
                        backup_col.modify(name=target_col_name)
                        # Verify restored collection
                        restored_target = self.client.get_collection(name=target_col_name)
                        restored_count = restored_target.count()
                        restored_dim = self.get_collection_dimension(restored_target)
                        count_ok = (restored_count == orig_count)
                        if orig_count == 0:
                            dim_ok = True
                        else:
                            dim_ok = (orig_dim is not None and restored_dim is not None and restored_dim == orig_dim)
                        if count_ok and dim_ok:
                            logger.info("Successfully rolled back and verified original collection '%s' from backup '%s'. Original vectors preserved.", target_col_name, backup_col_name)
                        else:
                            # Restoring to target_col_name succeeded, but post-rename verification failed.
                            # Attempt to preserve the original collection under the identifiable backup name again.
                            backup_preserved = False
                            if hasattr(restored_target, "modify"):
                                try:
                                    restored_target.modify(name=backup_col_name)
                                    backup_preserved = True
                                except Exception as re_backup_err:
                                    logger.warning(
                                        "Could not rename collection at '%s' back to backup name '%s': %s",
                                        target_col_name,
                                        backup_col_name,
                                        re_backup_err,
                                    )

                            if backup_preserved:
                                logger.critical(
                                    "CRITICAL: Rollback collection '%s' failed verification (count: %s vs %s, dim: %s vs %s). "
                                    "The original vectors have been preserved under backup collection '%s'. DO NOT DELETE '%s'.",
                                    target_col_name,
                                    restored_count,
                                    orig_count,
                                    restored_dim,
                                    orig_dim,
                                    backup_col_name,
                                    backup_col_name,
                                )
                            else:
                                logger.critical(
                                    "CRITICAL: Rollback collection '%s' failed verification (count: %s vs %s, dim: %s vs %s) "
                                    "and could not be renamed back to '%s'. Original vectors remain under collection '%s'. DO NOT DELETE '%s'.",
                                    target_col_name,
                                    restored_count,
                                    orig_count,
                                    restored_dim,
                                    orig_dim,
                                    backup_col_name,
                                    target_col_name,
                                    target_col_name,
                                )
                    except Exception as rb_err:
                        actual_col_name = None
                        try:
                            self.client.get_collection(name=backup_col_name)
                            actual_col_name = backup_col_name
                        except Exception:
                            try:
                                self.client.get_collection(name=target_col_name)
                                actual_col_name = target_col_name
                            except Exception:
                                pass

                        if actual_col_name:
                            logger.critical(
                                "CRITICAL: Failed to rollback backup collection '%s' to '%s': %s. "
                                "Original vectors remain preserved in collection '%s'. DO NOT DELETE '%s'.",
                                backup_col_name,
                                target_col_name,
                                rb_err,
                                actual_col_name,
                                actual_col_name,
                            )
                        else:
                            logger.critical(
                                "CRITICAL: Failed to rollback backup collection '%s' to '%s': %s. "
                                "Could not determine surviving collection name between '%s' and '%s'.",
                                backup_col_name,
                                target_col_name,
                                rb_err,
                                backup_col_name,
                                target_col_name,
                            )
                    return False

        # ---------------------------------------------------------
        # Path B2: Client does NOT support renaming
        # Use verified backup-and-restore copy strategy (Requirement 2, 3, 4)
        # ---------------------------------------------------------
        logger.info(
            "Collection modification/rename not supported on Chroma client for repo '%s'; executing verified backup-and-restore swap...",
            repository_id,
        )
        backup_col_name = f"backup_{target_col_name}_{uuid.uuid4().hex[:8]}"
        backup_col = None
        backup_verified = False

        try:
            backup_col = self.client.create_collection(
                name=backup_col_name,
                metadata=existing_col.metadata
            )
            copy_ok = self._copy_collection_data(existing_col, backup_col)
            if copy_ok:
                backup_count = backup_col.count()
                backup_dim = self.get_collection_dimension(backup_col)
                # 1. Exact count equality
                count_ok = (backup_count == orig_count)
                # 2. Confirmed vector dimension matching original collection's actual stored vectors
                if orig_count == 0:
                    dim_ok = True
                else:
                    dim_ok = (orig_dim is not None and backup_dim is not None and backup_dim == orig_dim)
                # 3. Verify copied IDs and embeddings are present and usable
                usable_ok = True
                if orig_count > 0:
                    sample = backup_col.get(limit=1, include=["embeddings", "documents", "metadatas"])
                    sample_ids = sample.get("ids", [])
                    sample_embs = sample.get("embeddings")
                    has_embs = (sample_embs is not None and len(sample_embs) > 0 and len(sample_embs[0]) > 0)
                    usable_ok = bool(sample_ids and has_embs)

                if count_ok and dim_ok and usable_ok:
                    backup_verified = True
                else:
                    logger.error(
                        "Backup collection '%s' verification failed against original '%s' "
                        "(count: %s vs %s, dim: %s vs %s, usable_ok: %s); aborting replacement to preserve original.",
                        backup_col_name,
                        target_col_name,
                        backup_count,
                        orig_count,
                        backup_dim,
                        orig_dim,
                        usable_ok,
                    )
        except Exception as backup_err:
            logger.error(
                "Failed to create verified backup of existing collection '%s': %s; aborting replacement to preserve original collection.",
                target_col_name,
                backup_err,
            )

        # If backup verification fails, abort and preserve original collection untouched (Requirement 2)
        if not backup_verified:
            if backup_col is not None:
                try:
                    self.client.delete_collection(name=backup_col_name)
                except Exception:
                    pass
            try:
                self.client.delete_collection(name=staging_col_name)
            except Exception:
                pass
            logger.warning(
                "Safe backup verification failed for repo '%s'; original collection '%s' was preserved untouched.",
                repository_id,
                target_col_name,
            )
            return False

        # Now that backup is verified, delete existing collection to make room for replacement
        try:
            self.client.delete_collection(name=target_col_name)
        except Exception as del_err:
            logger.error("Failed to delete existing collection '%s': %s. Original collection preserved in backup '%s'.", target_col_name, del_err, backup_col_name)
            try:
                self.client.delete_collection(name=staging_col_name)
            except Exception:
                pass
            return False

        # Create target collection and copy from staging
        replacement_ok = False
        try:
            target_col = self.client.create_collection(
                name=target_col_name,
                metadata=staging_metadata
            )
            copy_stg_ok = self._copy_collection_data(staging_col, target_col)
            target_dim_found = self.get_collection_dimension(target_col)
            # Verify replacement collection count and vector dimensions
            if copy_stg_ok and target_col.count() == len(valid_chunks) and target_dim_found is not None and target_dim_found == target_dim:
                replacement_ok = True
            else:
                logger.error(
                    "Replacement collection '%s' failed verification (count: %s vs %s, dim: %s vs %s). Initiating rollback.",
                    target_col_name,
                    target_col.count() if target_col else 0,
                    len(valid_chunks),
                    target_dim_found,
                    target_dim,
                )
        except Exception as repl_err:
            logger.error("Failed to create/populate replacement collection '%s': %s. Initiating rollback.", target_col_name, repl_err)

        if replacement_ok:
            if deferred_cleanup:
                self._pending_backups[repository_id] = {
                    "backup_col_name": backup_col_name,
                    "target_col_name": target_col_name,
                    "orig_count": orig_count,
                    "orig_dim": orig_dim,
                    "rename_supported": False,
                }
                try:
                    self.client.delete_collection(name=staging_col_name)
                except Exception:
                    pass
                logger.info(
                    "Safely replaced Chroma collection '%s' with %s chunks via verified backup-and-restore; backup '%s' retained pending database commit.",
                    target_col_name,
                    len(valid_chunks),
                    backup_col_name,
                )
                return True

            try:
                self.client.delete_collection(name=backup_col_name)
            except Exception:
                pass
            try:
                self.client.delete_collection(name=staging_col_name)
            except Exception:
                pass
            logger.info("Safely replaced Chroma collection '%s' with %s chunks via verified backup-and-restore.", target_col_name, len(valid_chunks))
            return True
        else:
            # Strengthened rollback verification (Requirement 3)
            try:
                self.client.delete_collection(name=target_col_name)
            except Exception:
                pass

            restored_ok = False
            try:
                restored_col = self.client.create_collection(
                    name=target_col_name,
                    metadata=backup_col.metadata
                )
                restore_copy_ok = self._copy_collection_data(backup_col, restored_col)
                if restore_copy_ok:
                    # Verify:
                    # 1. Retrieved under target name
                    # 2. Count equals backup count
                    # 3. Vector dimension equals backup dimension
                    retrieved_target = self.client.get_collection(name=target_col_name)
                    restored_count = retrieved_target.count()
                    restored_dim = self.get_collection_dimension(retrieved_target)

                    backup_count = backup_col.count()
                    backup_dim = self.get_collection_dimension(backup_col)

                    count_ok = (restored_count == backup_count)
                    if backup_count == 0:
                        dim_ok = True
                    else:
                        dim_ok = (backup_dim is not None and restored_dim is not None and restored_dim == backup_dim)

                    if count_ok and dim_ok:
                        restored_ok = True
                        logger.info("Successfully restored and verified original collection '%s' from backup.", target_col_name)
                        try:
                            self.client.delete_collection(name=backup_col_name)
                        except Exception:
                            pass
                    else:
                        logger.critical(
                            "CRITICAL: Restored collection '%s' failed verification (count: %s vs %s, dim: %s vs %s). "
                            "Retaining verified backup collection '%s'. DO NOT DELETE '%s'.",
                            target_col_name,
                            restored_count,
                            backup_count,
                            restored_dim,
                            backup_dim,
                            backup_col_name,
                            backup_col_name,
                        )
                else:
                    logger.critical(
                        "CRITICAL: Failed to copy data during rollback from backup '%s' to '%s'. "
                        "Retaining verified backup collection '%s'. DO NOT DELETE '%s'.",
                        backup_col_name,
                        target_col_name,
                        backup_col_name,
                        backup_col_name,
                    )
            except Exception as rb_copy_err:
                logger.critical(
                    "CRITICAL: Exception during rollback restoration of collection '%s' from backup '%s': %s. "
                    "Original vectors remain safely preserved in backup collection '%s'. DO NOT DELETE '%s'.",
                    target_col_name,
                    backup_col_name,
                    rb_copy_err,
                    backup_col_name,
                    backup_col_name,
                )

            try:
                self.client.delete_collection(name=staging_col_name)
            except Exception:
                pass
            return False

    def has_pending_replacement(self, repository_id: str) -> bool:
        """
        Checks whether there is an active deferred collection replacement
        waiting for external database transaction finalization.
        """
        return repository_id in self._pending_backups

    def finalize_replacement(self, repository_id: str) -> bool:
        """
        Finalizes a safe collection replacement after the external database commit succeeds.
        Deletes the preserved backup collection and clears pending replacement tracking.
        """
        info = self._pending_backups.pop(repository_id, None)
        if not info:
            return True

        backup_name = info.get("backup_col_name")
        if backup_name:
            try:
                self.client.delete_collection(name=backup_name)
                logger.info(
                    "Finalized collection replacement for repo '%s': temporary backup '%s' deleted.",
                    repository_id,
                    backup_name,
                )
            except Exception as cleanup_err:
                logger.warning(
                    "Non-fatal: Failed to delete temporary backup collection '%s' during finalization for repo '%s': %s",
                    backup_name,
                    repository_id,
                    cleanup_err,
                )
        return True

    def rollback_replacement(self, repository_id: str) -> bool:
        """
        Rolls back a replaced Chroma collection to the preserved backup collection
        when an external database transaction fails.
        Verifies that the restored collection matches the original count and dimension.
        If backup restoration fails, falls back to database-backed rebuild.
        Returns True if consistency was verified and restored; False otherwise.
        """
        info = self._pending_backups.pop(repository_id, None)
        if not info:
            logger.warning(
                "No pending replacement found to rollback for repository '%s'. Attempting database-backed recovery...",
                repository_id,
            )
            return self.rebuild_collection_from_db(repository_id)

        target_col_name = info["target_col_name"]
        backup_name = info.get("backup_col_name")
        orig_count = info.get("orig_count", 0)
        orig_dim = info.get("orig_dim")
        rename_supported = info.get("rename_supported", False)

        # Case 1: Initial ingestion (no prior collection existed).
        # Discard the replacement target collection so no orphaned vectors remain.
        if not backup_name:
            try:
                self.client.delete_collection(name=target_col_name)
                logger.info(
                    "Successfully rolled back newly created collection '%s' for repository '%s'.",
                    target_col_name,
                    repository_id,
                )
                return True
            except Exception as del_err:
                logger.error(
                    "Failed to delete newly created collection '%s' during rollback for repository '%s': %s",
                    target_col_name,
                    repository_id,
                    del_err,
                )
                return False

        # Case 2: Prior collection was preserved in backup_name.
        try:
            backup_col = self.client.get_collection(name=backup_name)
        except Exception as bk_err:
            logger.critical(
                "CRITICAL: Preserved backup collection '%s' for repo '%s' could not be retrieved: %s. "
                "Attempting database-backed rebuild...",
                backup_name,
                repository_id,
                bk_err,
            )
            return self.rebuild_collection_from_db(repository_id)

        # Step 2a: Try rename restoration if supported
        if rename_supported and callable(getattr(backup_col, "modify", None)):
            try:
                try:
                    self.client.delete_collection(name=target_col_name)
                except Exception:
                    pass

                backup_col.modify(name=target_col_name)
                restored_col = self.client.get_collection(name=target_col_name)
                restored_count = restored_col.count()
                restored_dim = self.get_collection_dimension(restored_col)

                count_ok = (restored_count == orig_count)
                dim_ok = (orig_count == 0 or (orig_dim is not None and restored_dim is not None and restored_dim == orig_dim))

                if count_ok and dim_ok:
                    logger.info(
                        "Successfully rolled back and verified original collection '%s' from backup '%s' for repo '%s'.",
                        target_col_name,
                        backup_name,
                        repository_id,
                    )
                    return True
                else:
                    logger.critical(
                        "CRITICAL: Restored collection '%s' failed verification during rename rollback for repo '%s' "
                        "(count: %s vs %s, dim: %s vs %s). Attempting database-backed rebuild...",
                        target_col_name,
                        repository_id,
                        restored_count,
                        orig_count,
                        restored_dim,
                        orig_dim,
                    )
                    return self.rebuild_collection_from_db(repository_id)
            except Exception as rename_err:
                logger.warning(
                    "Rename rollback from '%s' to '%s' failed for repo '%s': %s. Falling back to copy restoration...",
                    backup_name,
                    target_col_name,
                    repository_id,
                    rename_err,
                )

        # Step 2b: Fallback to verified copy restoration
        try:
            try:
                self.client.delete_collection(name=target_col_name)
            except Exception:
                pass

            restored_col = self.client.create_collection(
                name=target_col_name,
                metadata=backup_col.metadata,
            )
            copy_ok = self._copy_collection_data(backup_col, restored_col)
            restored_count = restored_col.count()
            restored_dim = self.get_collection_dimension(restored_col)

            count_ok = (copy_ok and restored_count == orig_count)
            dim_ok = (orig_count == 0 or (orig_dim is not None and restored_dim is not None and restored_dim == orig_dim))

            if count_ok and dim_ok:
                try:
                    self.client.delete_collection(name=backup_name)
                except Exception:
                    pass
                logger.info(
                    "Successfully restored and verified original collection '%s' from backup '%s' via copy for repo '%s'.",
                    target_col_name,
                    backup_name,
                    repository_id,
                )
                return True
            else:
                logger.critical(
                    "CRITICAL: Copy restoration from backup '%s' failed verification for repo '%s' "
                    "(count: %s vs %s, dim: %s vs %s). Attempting database-backed rebuild...",
                    backup_name,
                    repository_id,
                    restored_count,
                    orig_count,
                    restored_dim,
                    orig_dim,
                )
                return self.rebuild_collection_from_db(repository_id)
        except Exception as copy_err:
            logger.critical(
                "CRITICAL: Copy restoration from backup '%s' encountered error for repo '%s': %s. "
                "Attempting database-backed rebuild...",
                backup_name,
                repository_id,
                copy_err,
            )
            return self.rebuild_collection_from_db(repository_id)

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

        active_model = self.embedding_model_name
        active_dim = self.embedding_dimension
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

            # Query existing Chroma collection count if present, to protect against partial replacement
            existing_col_count = 0
            try:
                col_name = self._get_collection_name(repo.id)
                existing_col = self.client.get_collection(name=col_name)
                existing_col_count = existing_col.count()
            except Exception:
                existing_col_count = 0

            expected_file_count = getattr(repo, "file_count", 0) or 0
            expected_chunk_count = getattr(repo, "chunk_count", 0) or 0

            db_files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).all()
            db_chunks = db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).all()

            # -------------------------------------------------------------
            # Evaluate Candidate 1: Stored RepositoryFile records
            # -------------------------------------------------------------
            file_chunks = []
            files_valid = False
            files_reason = ""

            if not db_files:
                files_reason = "No RepositoryFile records found in database."
            elif expected_file_count > 0 and len(db_files) < expected_file_count:
                files_reason = (
                    f"Stored file count ({len(db_files)}) is less than repository's "
                    f"recorded file_count ({expected_file_count})."
                )
            else:
                from app.github.parser import is_ignored_path, parse_file_into_chunks
                valid_files = [f for f in db_files if not is_ignored_path(f.file_path)]

                missing_content_files = [f.file_path for f in valid_files if f.content is None]
                if missing_content_files:
                    files_reason = (
                        f"{len(missing_content_files)} stored file(s) have NULL content: "
                        f"{missing_content_files[:5]}"
                    )
                else:
                    for f in valid_files:
                        if f.content:
                            parsed = parse_file_into_chunks(
                                file_path=f.file_path,
                                content=f.content,
                                language=f.language,
                            )
                            file_chunks.extend(parsed)

                    if not file_chunks:
                        files_reason = "Parsing stored files produced zero chunks."
                    elif expected_chunk_count > 0 and len(file_chunks) < expected_chunk_count:
                        files_reason = (
                            f"Parsed chunks ({len(file_chunks)}) fewer than repository's "
                            f"recorded chunk_count ({expected_chunk_count})."
                        )
                    elif len(db_chunks) > 0 and len(file_chunks) < len(db_chunks):
                        files_reason = (
                            f"Parsed chunks ({len(file_chunks)}) fewer than stored "
                            f"CodeChunk records ({len(db_chunks)})."
                        )
                    elif existing_col_count > 0 and len(file_chunks) < existing_col_count:
                        files_reason = (
                            f"Parsed chunks ({len(file_chunks)}) fewer than existing "
                            f"collection vectors ({existing_col_count})."
                        )
                    else:
                        files_valid = True

            # -------------------------------------------------------------
            # Evaluate Candidate 2: Stored CodeChunk records
            # -------------------------------------------------------------
            chunks_payload = []
            chunks_valid = False
            chunks_reason = ""

            if not db_chunks:
                chunks_reason = "No CodeChunk records found in database."
            else:
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
                    for c in db_chunks
                    if c.content and c.content.strip() and c.chunk_id
                ]

                if not chunks_payload:
                    chunks_reason = "Stored CodeChunk records produced zero usable chunks with non-empty content and ID."
                elif len(chunks_payload) < len(db_chunks):
                    chunks_reason = (
                        f"Usable CodeChunk records ({len(chunks_payload)}) fewer than total "
                        f"CodeChunk records ({len(db_chunks)})."
                    )
                elif expected_chunk_count > 0 and len(chunks_payload) < expected_chunk_count:
                    chunks_reason = (
                        f"Usable CodeChunk records ({len(chunks_payload)}) fewer than repository's "
                        f"recorded chunk_count ({expected_chunk_count})."
                    )
                elif existing_col_count > 0 and len(chunks_payload) < existing_col_count:
                    chunks_reason = (
                        f"Usable CodeChunk records ({len(chunks_payload)}) fewer than existing "
                        f"collection vectors ({existing_col_count})."
                    )
                else:
                    chunks_valid = True

            # -------------------------------------------------------------
            # Select rebuild source and enforce completeness
            # -------------------------------------------------------------
            chunks_to_replace = []
            source_description = ""

            if files_valid:
                chunks_to_replace = file_chunks
                source_description = f"{len(valid_files)} stored files ({len(file_chunks)} chunks)"
            elif chunks_valid:
                if db_files:
                    logger.info(
                        "Stored files for repo '%s' were not fully usable (%s); "
                        "safely falling back to %s validated CodeChunk records.",
                        repo.id,
                        files_reason,
                        len(chunks_payload),
                    )
                chunks_to_replace = chunks_payload
                source_description = f"{len(chunks_payload)} stored CodeChunk records"
            else:
                logger.error(
                    "Cannot establish safe completeness of rebuild source for repo '%s'. "
                    "File-based rebuild rejected: %s. "
                    "Chunk-based rebuild rejected: %s. "
                    "Aborting rebuild and preserving existing collection (%s vectors) intact.",
                    repo.id,
                    files_reason,
                    chunks_reason,
                    existing_col_count,
                )
                return False

            success = self.safe_replace_collection(repo.id, chunks_to_replace)
            if success:
                repo.embedding_model = active_model
                repo.embedding_dimension = active_dim
                db.commit()
                logger.info(
                    "Successfully rebuilt collection for repo '%s' from %s.",
                    repo.id,
                    source_description,
                )
                return True
            else:
                logger.error(
                    "Failed to safely replace collection for repo '%s'; retaining existing collection.",
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
        Generates embeddings and stores repository chunks in ChromaDB.
        Safely detects and handles dimension mismatches without relying on stale metadata.
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(
            repository_id
        )

        target_dim = self.embedding_dimension
        active_model = self.embedding_model_name
        col_count = collection.count()

        if col_count == 0:
            # Empty collection: no stored vectors exist, so there is no dimension conflict.
            # Synchronize collection metadata with active target dimension and model if stale.
            meta = collection.metadata or {}
            if (meta.get("dimension") != target_dim or meta.get("model") != active_model) and hasattr(collection, "modify"):
                try:
                    new_meta = dict(meta)
                    new_meta["dimension"] = target_dim
                    new_meta["model"] = active_model
                    new_meta["provider"] = getattr(settings, "EMBEDDING_PROVIDER", "local")
                    collection.modify(metadata=new_meta)
                except Exception as meta_err:
                    logger.debug("Could not refresh metadata on empty collection: %s", meta_err)
        else:
            # Collection has stored vectors: check dimension and model compatibility
            if not self.is_collection_compatible(collection, repository_id=repository_id):
                logger.warning(
                    "Collection '%s' compatibility mismatch during upsert (active provider dim: %s, model: %s). "
                    "Safely replacing collection with replacement chunks...",
                    self._get_collection_name(repository_id),
                    self.embedding_dimension,
                    self.embedding_model_name,
                )
                success = self.safe_replace_collection(repository_id, chunks)
                if not success:
                    logger.error(
                        "Failed to safely replace collection for repo '%s' during upsert compatibility mismatch. "
                        "Original collection was preserved intact.",
                        repository_id,
                    )
                return

        # -----------------------------------------------------
        # Batch embedding generation and upsert
        # -----------------------------------------------------
        raw_upsert_batch = os.environ.get("CHROMA_UPSERT_BATCH_SIZE")
        upsert_batch_size = 16
        if raw_upsert_batch is not None:
            try:
                parsed_batch = int(str(raw_upsert_batch).strip())
                if parsed_batch > 0:
                    upsert_batch_size = parsed_batch
            except (ValueError, TypeError):
                pass

        for i in range(0, len(chunks), upsert_batch_size):
            batch_slice = chunks[i : i + upsert_batch_size]
            batch_docs = [chunk["content"] for chunk in batch_slice]
            batch_ids = [str(chunk["chunk_id"]) for chunk in batch_slice]
            batch_metas = [
                {
                    "file_path": chunk["file_path"],
                    "language": chunk["language"],
                    "start_line": int(chunk["start_line"]),
                    "end_line": int(chunk["end_line"]),
                    "symbol_name": chunk.get("symbol_name") or "",
                    "chunk_id": str(chunk["chunk_id"]),
                    "defined_symbols": chunk.get("defined_symbols", ""),
                    "called_symbols": chunk.get("called_symbols", ""),
                    "imported_symbols": chunk.get("imported_symbols", ""),
                }
                for chunk in batch_slice
            ]

            try:
                batch_embeddings = self.embedding_provider.embed_documents(batch_docs)
            except Exception as embed_err:
                logger.error("Failed to generate embeddings during upsert for repo '%s': %s", repository_id, embed_err, exc_info=True)
                return

            if not batch_embeddings or len(batch_embeddings) != len(batch_docs):
                logger.error(
                    "Embedding provider generated %s embeddings for %s documents for repo '%s'. Aborting upsert.",
                    len(batch_embeddings) if batch_embeddings else 0,
                    len(batch_docs),
                    repository_id,
                )
                return

            if len(batch_embeddings[0]) != target_dim:
                logger.error(
                    "Generated embeddings dimension %s does not match expected target dimension %s for repo '%s'. Aborting upsert.",
                    len(batch_embeddings[0]),
                    target_dim,
                    repository_id,
                )
                return

            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            del batch_docs, batch_ids, batch_metas, batch_embeddings

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
        # Check Dimension & Model Compatibility
        # -----------------------------------------------------

        if collection_count > 0 and not self.is_collection_compatible(collection, repository_id=repository_id):
            col_dim = self.get_collection_dimension(collection)
            col_meta = collection.metadata or {}
            col_model = col_meta.get("model", "unknown")
            logger.warning(
                "Chroma collection compatibility mismatch for repo %s: collection (dim=%s, model=%s) "
                "is not compatible with active provider (dim=%s, model=%s). Rebuilding collection from stored chunks...",
                repository_id,
                col_dim,
                col_model,
                self.embedding_dimension,
                self.embedding_model_name,
            )
            rebuilt = self.rebuild_collection_from_db(repository_id)
            if not rebuilt:
                logger.error(
                    "Automatic rebuild of collection for repo %s failed or could not establish completeness safely; "
                    "refusing to query incompatible collection.",
                    repository_id,
                )
                return []

            collection = self.get_or_create_collection(repository_id)
            collection_count = collection.count()
            if collection_count == 0:
                logger.error(
                    "Rebuilt collection for repo %s is unexpectedly empty; refusing to query.",
                    repository_id,
                )
                return []

            if not self.is_collection_compatible(collection, repository_id=repository_id):
                logger.error(
                    "Rebuilt collection for repo %s remains incompatible with active provider (dim=%s, model=%s); "
                    "refusing to query incompatible collection.",
                    repository_id,
                    self.embedding_dimension,
                    self.embedding_model_name,
                )
                return []

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
                if not rebuilt:
                    logger.error(
                        "Rebuild recovery failed after Chroma dimension error for repo %s; returning empty results.",
                        repository_id,
                    )
                    return []
                collection = self.get_or_create_collection(repository_id)
                collection_count = collection.count()
                if collection_count == 0:
                    logger.error(
                        "Rebuilt collection for repo %s is unexpectedly empty after dimension error recovery.",
                        repository_id,
                    )
                    return []
                if not self.is_collection_compatible(collection, repository_id=repository_id):
                    logger.error(
                        "Rebuilt collection for repo %s remains incompatible with active provider after dimension error recovery.",
                        repository_id,
                    )
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