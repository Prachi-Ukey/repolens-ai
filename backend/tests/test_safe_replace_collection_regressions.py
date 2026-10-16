"""
Regression tests for ChromaManager.safe_replace_collection and upsert_chunks.
Verifies failure handling during replacement creation, vector copying, promotion,
verification, and rollback, proving that the original collection remains intact whenever
a safe replacement has not completed.
"""

import os
import sys
import uuid
import tempfile
import shutil
import pytest
import chromadb
from unittest.mock import patch, MagicMock

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.ai.vector_store import ChromaManager
from app.ai.embeddings import MockEmbedding


class TestSafeReplaceCollectionRegressions:
    """Test suite proving original collections are preserved across all failure modes."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="chroma_regress_")
        self.manager = ChromaManager(persist_dir=self.temp_dir)
        self.manager.embedding_provider = MockEmbedding(dim=384)

    def teardown_method(self):
        if hasattr(self.manager, "client"):
            try:
                self.manager.client.close()
            except Exception:
                pass
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _seed_original_collection(self, repo_id: str, count: int = 1, dim: int = 768, doc_prefix: str = "original_doc"):
        target_name = self.manager._get_collection_name(repo_id)
        col = self.manager.client.create_collection(
            name=target_name,
            metadata={"dimension": dim, "model": f"model_{dim}"}
        )
        ids = [f"orig_{i}" for i in range(count)]
        embs = [[0.5] * dim for _ in range(count)]
        docs = [f"{doc_prefix}_{i}" for i in range(count)]
        metas = [{"chunk_id": f"orig_{i}", "file_path": f"src/file_{i}.py", "language": "python", "start_line": 1, "end_line": 10} for i in range(count)]
        col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        return col

    def _make_candidate_chunks(self, count: int = 2):
        return [
            {
                "chunk_id": f"cand_{i}",
                "content": f"replacement code content {i}",
                "file_path": f"src/new_{i}.py",
                "language": "python",
                "start_line": 1,
                "end_line": 5,
            }
            for i in range(count)
        ]

    def test_failure_during_replacement_creation_preserves_original(self):
        """If creating the staging collection fails, original collection is untouched."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)

        candidate_chunks = self._make_candidate_chunks(count=3)
        orig_create = self.manager.client.create_collection

        def mock_create_failure(name, *args, **kwargs):
            if "staging_" in name:
                raise RuntimeError("Simulated out of disk space during staging creation")
            return orig_create(name, *args, **kwargs)

        with patch.object(self.manager.client, "create_collection", side_effect=mock_create_failure):
            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
            assert success is False

        # Verify original collection remains intact and available
        col = self.manager.client.get_collection(self.manager._get_collection_name(repo_id))
        assert col.count() == 2
        docs = col.get()["documents"]
        assert "original_doc_0" in docs
        assert "original_doc_1" in docs
        assert self.manager.get_collection_dimension(col) == 768

    def test_failure_during_staging_upsert_preserves_original(self):
        """If populating staging collection fails, staging is cleaned up and original is untouched."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=1, dim=384)

        candidate_chunks = self._make_candidate_chunks(count=2)

        with patch("chromadb.api.models.Collection.Collection.upsert", side_effect=RuntimeError("Simulated upsert failure")):
            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
            assert success is False

        # Verify original collection is preserved
        col = self.manager.client.get_collection(self.manager._get_collection_name(repo_id))
        assert col.count() == 1
        assert col.get()["documents"] == ["original_doc_0"]

    def test_failure_during_vector_copying_preserves_original(self):
        """In the non-rename path, if copying to backup fails, original collection is never touched."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        orig_col = self._seed_original_collection(repo_id, count=2, dim=768)
        candidate_chunks = self._make_candidate_chunks(count=1)

        # Simulate a client where modify (rename) is not supported
        with patch.object(self.manager, "_copy_collection_data", return_value=False):
            with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
                success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                assert success is False

        # Verify original collection was never deleted or modified
        col = self.manager.client.get_collection(self.manager._get_collection_name(repo_id))
        assert col.count() == 2
        assert self.manager.get_collection_dimension(col) == 768
        assert "original_doc_0" in col.get()["documents"]

    def test_failure_during_promotion_triggers_rollback_and_preserves_original(self):
        """If promoting the staging collection fails, rollback restores the original collection."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=1, dim=768)
        candidate_chunks = self._make_candidate_chunks(count=2)

        target_name = self.manager._get_collection_name(repo_id)
        orig_modify = chromadb.api.models.Collection.Collection.modify

        def mock_modify(col_self, **kwargs):
            # Allow renaming existing collection to backup, but fail when promoting staging to target_name
            if kwargs.get("name") == target_name and col_self.name.startswith("staging_"):
                raise RuntimeError("Simulated rename lock failure during promotion")
            return orig_modify(col_self, **kwargs)

        with patch.object(chromadb.api.models.Collection.Collection, "modify", autospec=True, side_effect=mock_modify):
            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
            assert success is False

        # Verify rollback restored the original collection under target_name
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 1
        assert col.get()["documents"] == ["original_doc_0"]
        assert self.manager.get_collection_dimension(col) == 768

    def test_failure_during_replacement_verification_triggers_rollback(self):
        """If promoted collection fails verification (e.g. count or dimension check), rollback restores original."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=1, dim=768)
        candidate_chunks = self._make_candidate_chunks(count=2)

        target_name = self.manager._get_collection_name(repo_id)

        # Mock get_collection_dimension on target_name to return mismatched dimension during promotion verification
        orig_get_dim = self.manager.get_collection_dimension

        def mock_dim_verifier(col):
            if col.name == target_name and col.count() == 2:
                # Return wrong dimension during post-promotion verification
                return 999
            return orig_get_dim(col)

        with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim_verifier):
            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
            assert success is False

        # Verify rollback restored the original collection
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 1
        assert col.get()["documents"] == ["original_doc_0"]

    def test_non_rename_path_failure_during_target_population_restores_from_backup(self):
        """When rename is not supported, if populating new target fails, rollback restores from backup copy."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=1, dim=768)
        candidate_chunks = self._make_candidate_chunks(count=2)
        target_name = self.manager._get_collection_name(repo_id)

        orig_create = self.manager.client.create_collection

        # Simulate modify not supported
        with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
            # Simulate failure when creating the replacement target collection
            def mock_create(name, *args, **kwargs):
                if name == target_name:
                    # Let target creation fail once to trigger rollback
                    if not getattr(mock_create, "failed_once", False):
                        mock_create.failed_once = True
                        raise RuntimeError("Simulated target creation failure")
                return orig_create(name, *args, **kwargs)

            mock_create.failed_once = False
            with patch.object(self.manager.client, "create_collection", side_effect=mock_create):
                success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                assert success is False

        # Verify rollback restored original collection
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 1
        assert col.get()["documents"] == ["original_doc_0"]

    def test_upsert_chunks_does_not_rely_on_stale_metadata_when_empty(self):
        """Empty collection with stale 768-d metadata should not trigger dimension mismatch error on 384-d upsert."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        target_name = self.manager._get_collection_name(repo_id)

        # Create empty collection with stale 768-d metadata
        empty_col = self.manager.client.create_collection(
            name=target_name,
            metadata={"dimension": 768, "model": "legacy-gemini"}
        )
        assert empty_col.count() == 0

        # Upsert 384-d chunks
        candidate_chunks = self._make_candidate_chunks(count=2)
        self.manager.upsert_chunks(repo_id, candidate_chunks)

        # Confirm collection was updated with 2 chunks and metadata refreshed
        col_after = self.manager.client.get_collection(target_name)
        assert col_after.count() == 2
        assert self.manager.get_collection_dimension(col_after) == 384
        assert col_after.metadata.get("dimension") == 384

    def test_upsert_chunks_probes_real_vectors_to_detect_mismatch_despite_stale_metadata(self):
        """Collection with 768-d vectors and stale 384 metadata should still be detected as 768-d and safely replaced."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        target_name = self.manager._get_collection_name(repo_id)

        # Seed collection with 768-d vectors, but lie in metadata saying 384
        col = self.manager.client.create_collection(
            name=target_name,
            metadata={"dimension": 384, "model": "fake-metadata"}
        )
        col.add(ids=["old_1"], embeddings=[[0.05] * 768], documents=["old 768 doc"], metadatas=[{"chunk_id": "old_1", "file_path": "old.py", "language": "python", "start_line": 1, "end_line": 2}])
        assert col.count() == 1

        # Real vector probe must detect 768, not the stale 384 in metadata
        detected_dim = self.manager.get_collection_dimension(col)
        assert detected_dim == 768

        # Upserting 384-d chunks should trigger safe replacement
        candidate_chunks = self._make_candidate_chunks(count=2)
        self.manager.upsert_chunks(repo_id, candidate_chunks)

        # Confirm collection was replaced and now holds 384-d vectors
        col_after = self.manager.client.get_collection(target_name)
        assert col_after.count() == 2
        assert self.manager.get_collection_dimension(col_after) == 384
        docs = col_after.get()["documents"]
        assert "replacement code content 0" in docs
        assert "replacement code content 1" in docs

    def test_unknown_staging_dimension_aborts_and_preserves_original(self):
        """If staging collection dimension is None/unknown, replacement is aborted and original collection is untouched."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)

        candidate_chunks = self._make_candidate_chunks(count=3)
        orig_get_dim = self.manager.get_collection_dimension

        def mock_dim(col):
            # Return None when inspecting staging collection dimension
            if col.name.startswith("staging_"):
                return None
            return orig_get_dim(col)

        with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim):
            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
            assert success is False

        # Verify original collection remains intact and untouched
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 2
        assert self.manager.get_collection_dimension(col) == 768
        docs = col.get()["documents"]
        assert "original_doc_0" in docs
        assert "original_doc_1" in docs

    def test_backup_dimension_mismatch_aborts_and_preserves_original(self):
        """In non-rename path, if backup collection dimension does not match original collection, abort and preserve original."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)

        candidate_chunks = self._make_candidate_chunks(count=2)
        orig_get_dim = self.manager.get_collection_dimension

        def mock_dim(col):
            # Return mismatched dimension when inspecting the backup collection
            if col.name.startswith("backup_"):
                return 512
            return orig_get_dim(col)

        # Simulate non-rename path by disallowing modify
        with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
            with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim):
                success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                assert success is False

        # Verify original collection is preserved untouched
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 2
        assert orig_get_dim(col) == 768
        docs = col.get()["documents"]
        assert "original_doc_0" in docs
        assert "original_doc_1" in docs

    def test_backup_ids_missing_despite_nonzero_source_count_aborts(self):
        """If source collection count > 0 but Chroma get returns empty IDs during copy to backup, abort and preserve original."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        orig_col = self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)
        candidate_chunks = self._make_candidate_chunks(count=2)

        orig_col_get = chromadb.api.models.Collection.Collection.get

        def mock_get(col_self, *args, **kwargs):
            if col_self.name == target_name:
                return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
            return orig_col_get(col_self, *args, **kwargs)

        # Direct test on _copy_collection_data: returns False when count > 0 but IDs are empty
        dest_col = self.manager.client.create_collection(name=f"test_dest_{uuid.uuid4().hex[:8]}")
        with patch("chromadb.api.models.Collection.Collection.get", side_effect=mock_get):
            assert self.manager._copy_collection_data(orig_col, dest_col) is False

        # End-to-end test in safe_replace_collection under non-rename fallback
        with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
            with patch("chromadb.api.models.Collection.Collection.get", side_effect=mock_get):
                success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                assert success is False

        # Verify original collection remains available and undamaged
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 2
        assert "original_doc_0" in col.get()["documents"]

    def test_restored_dimension_mismatch_retains_backup(self, caplog):
        """In non-rename rollback, if restored collection dimension mismatches backup, backup is retained and CRITICAL is logged."""
        import logging
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)
        candidate_chunks = self._make_candidate_chunks(count=2)

        backup_names_seen = []
        orig_create = self.manager.client.create_collection

        # Simulate modify not supported
        with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
            def mock_create(name, *args, **kwargs):
                if name.startswith("backup_"):
                    backup_names_seen.append(name)
                return orig_create(name, *args, **kwargs)

            # Trigger replacement failure by making replacement copy fail
            orig_copy = self.manager._copy_collection_data
            rollback_started = False

            def mock_copy(src, dest):
                nonlocal rollback_started
                # Allow backup copy from existing_col, but fail replacement copy from staging_col
                if src.name.startswith("staging_") and dest.name == target_name:
                    rollback_started = True
                    return False
                return orig_copy(src, dest)

            # During rollback, make restored target dimension return a mismatch (e.g., 999 instead of 768)
            orig_get_dim = self.manager.get_collection_dimension
            def mock_dim(col):
                if rollback_started and col.name == target_name:
                    return 999  # Mismatch against backup_dim (768)
                return orig_get_dim(col)

            with caplog.at_level(logging.CRITICAL):
                with patch.object(self.manager.client, "create_collection", side_effect=mock_create):
                    with patch.object(self.manager, "_copy_collection_data", side_effect=mock_copy):
                        with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim):
                            success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                            assert success is False

        # Verify that:
        # 1. CRITICAL log was emitted with the exact backup collection name
        assert len(backup_names_seen) == 1
        backup_name = backup_names_seen[0]
        assert backup_name in caplog.text
        assert "CRITICAL" in caplog.text

        # 2. Backup collection was NOT deleted and is still available in ChromaDB
        backup_col = self.manager.client.get_collection(backup_name)
        assert backup_col is not None
        assert backup_col.count() == 2
        assert "original_doc_0" in backup_col.get()["documents"]
        assert orig_get_dim(backup_col) == 768

    def test_failed_rollback_verification_preserves_backup(self, caplog):
        """If rollback fails (e.g. data copy fails or exception during restore), backup is retained and CRITICAL logged."""
        import logging
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)
        candidate_chunks = self._make_candidate_chunks(count=2)

        backup_names_seen = []
        orig_create = self.manager.client.create_collection

        # Simulate modify not supported
        with patch("chromadb.api.models.Collection.Collection.modify", side_effect=AttributeError("modify not supported")):
            def mock_create(name, *args, **kwargs):
                if name.startswith("backup_"):
                    backup_names_seen.append(name)
                return orig_create(name, *args, **kwargs)

            orig_copy = self.manager._copy_collection_data
            def mock_copy(src, dest):
                # 1. Allow backup copy (existing -> backup)
                if src.name == target_name and dest.name.startswith("backup_"):
                    return orig_copy(src, dest)
                # 2. Fail replacement copy (staging -> target)
                if src.name.startswith("staging_") and dest.name == target_name:
                    return False
                # 3. Fail rollback restore copy (backup -> target)
                if src.name.startswith("backup_") and dest.name == target_name:
                    return False
                return orig_copy(src, dest)

            with caplog.at_level(logging.CRITICAL):
                with patch.object(self.manager.client, "create_collection", side_effect=mock_create):
                    with patch.object(self.manager, "_copy_collection_data", side_effect=mock_copy):
                        success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                        assert success is False

        # Verify backup was preserved and CRITICAL logged with backup name
        assert len(backup_names_seen) == 1
        backup_name = backup_names_seen[0]
        assert backup_name in caplog.text
        assert "CRITICAL" in caplog.text

        # Verify the backup is safely preserved in ChromaDB
        backup_col = self.manager.client.get_collection(backup_name)
        assert backup_col.count() == 2
        assert "original_doc_0" in backup_col.get()["documents"]
        assert self.manager.get_collection_dimension(backup_col) == 768

    def test_batched_copying_larger_than_one_batch(self):
        """_copy_collection_data correctly paginates and copies collections with > 100 items (multi-batch)."""
        src_name = f"test_src_{uuid.uuid4().hex[:8]}"
        dest_name = f"test_dest_{uuid.uuid4().hex[:8]}"

        dim = 384
        count = 150  # Bounded batch size in _copy_collection_data is 100
        src_col = self.manager.client.create_collection(
            name=src_name,
            metadata={"dimension": dim, "model": "mock-384"}
        )

        ids = [f"item_{i}" for i in range(count)]
        embs = [[0.01 * (i % 10)] * dim for i in range(count)]
        docs = [f"document content {i}" for i in range(count)]
        metas = [{"chunk_id": f"item_{i}", "file_path": f"src/mod_{i}.py", "language": "python", "start_line": 1, "end_line": 5} for i in range(count)]

        # Populate source in two batches
        src_col.add(ids=ids[:100], embeddings=embs[:100], documents=docs[:100], metadatas=metas[:100])
        src_col.add(ids=ids[100:], embeddings=embs[100:], documents=docs[100:], metadatas=metas[100:])
        assert src_col.count() == 150

        dest_col = self.manager.client.create_collection(
            name=dest_name,
            metadata={"dimension": dim, "model": "mock-384"}
        )

        # Copy data using bounded batch pagination
        success = self.manager._copy_collection_data(src_col, dest_col)
        assert success is True

        # Verify destination count and vector dimension
        assert dest_col.count() == 150
        assert self.manager.get_collection_dimension(dest_col) == dim

        # Verify items across batch boundaries exist in destination
        dest_sample = dest_col.get(ids=["item_0", "item_99", "item_149"], include=["embeddings", "documents", "metadatas"])
        assert set(dest_sample["ids"]) == {"item_0", "item_99", "item_149"}
        assert len(dest_sample["embeddings"][0]) == dim
        assert dest_sample["documents"] == ["document content 0", "document content 99", "document content 149"]

    def test_rename_rollback_restoration_verification_failure_preserves_backup_name(self, caplog):
        """In rename path, if restoring backup to target succeeds but verification fails, original collection is preserved under backup name."""
        import logging
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)
        candidate_chunks = self._make_candidate_chunks(count=2)

        backup_names = []
        backup_restored_to_target = False
        orig_modify = chromadb.api.models.Collection.Collection.modify

        def tracking_modify(col_self, **kwargs):
            nonlocal backup_restored_to_target
            name = kwargs.get("name")
            if name and name.startswith("backup_"):
                if name not in backup_names:
                    backup_names.append(name)
            # Fail staging promotion to initiate rollback to original collection
            if col_self.name.startswith("staging_") and name == target_name:
                raise RuntimeError("Simulated rename lock failure during staging promotion")
            # 1. Track rename from backup_names[0] to target_name: call orig_modify first,
            # and set backup_restored_to_target = True only after orig_modify() returns successfully.
            if backup_names and col_self.name == backup_names[0] and name == target_name:
                res = orig_modify(col_self, **kwargs)
                backup_restored_to_target = True
                return res
            return orig_modify(col_self, **kwargs)

        orig_get_dim = self.manager.get_collection_dimension

        # 2. Return real dimension during staging and replacement verification.
        # Return 999 ONLY when verifying the restored original collection after the backup has been renamed back to target_name.
        def mock_dim(col):
            if backup_restored_to_target and col.name == target_name:
                return 999
            return orig_get_dim(col)

        with caplog.at_level(logging.CRITICAL):
            with patch.object(chromadb.api.models.Collection.Collection, "modify", autospec=True, side_effect=tracking_modify):
                with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim):
                    success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                    assert success is False

        # 5. Assertion proving the backup-restoration step was reached
        assert backup_restored_to_target is True

        # 3. Verify rollback verification failed, original collection was renamed back to identifiable backup name,
        # and both original documents and original dimension remain accessible there.
        assert len(backup_names) == 1
        backup_name = backup_names[0]
        assert backup_name in caplog.text
        assert "CRITICAL" in caplog.text
        assert "The original vectors have been preserved under backup collection" in caplog.text

        # Verify original vectors remain accessible under the known backup collection name
        backup_col = self.manager.client.get_collection(backup_name)
        assert backup_col is not None
        assert backup_col.count() == 2
        docs = backup_col.get()["documents"]
        assert "original_doc_0" in docs
        assert "original_doc_1" in docs
        assert orig_get_dim(backup_col) == 768

        # Target name does not contain the unverified collection
        with pytest.raises(Exception):
            self.manager.client.get_collection(target_name)

    def test_rename_rollback_verification_failure_when_re_rename_fails_logs_actual_collection_name(self, caplog):
        """In rename path, if verification fails and renaming back to backup also fails, logs actual target collection name without claiming backup exists."""
        import logging
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_original_collection(repo_id, count=2, dim=768)
        target_name = self.manager._get_collection_name(repo_id)
        candidate_chunks = self._make_candidate_chunks(count=2)

        backup_names = []
        backup_restored_to_target = False
        re_backup_attempted = False
        orig_modify = chromadb.api.models.Collection.Collection.modify

        def tracking_modify(col_self, **kwargs):
            nonlocal backup_restored_to_target, re_backup_attempted
            name = kwargs.get("name")
            if name and name.startswith("backup_"):
                if name not in backup_names:
                    backup_names.append(name)
            # Fail staging promotion to initiate rollback to original collection
            if col_self.name.startswith("staging_") and name == target_name:
                raise RuntimeError("Simulated rename lock failure during staging promotion")
            # 4. Simulate failure only when attempting to rename restored original collection back to backup_col_name,
            # triggered only after the restoration rename has successfully completed.
            if backup_restored_to_target and col_self.name == target_name and backup_names and name == backup_names[0]:
                re_backup_attempted = True
                raise RuntimeError("Simulated rename failure when preserving backup")
            # 1. Track rename from backup_names[0] to target_name: call orig_modify first,
            # and set backup_restored_to_target = True only after orig_modify() returns successfully.
            if backup_names and col_self.name == backup_names[0] and name == target_name:
                res = orig_modify(col_self, **kwargs)
                backup_restored_to_target = True
                return res
            return orig_modify(col_self, **kwargs)

        orig_get_dim = self.manager.get_collection_dimension

        # 2. Return real dimension during staging and replacement verification.
        # Return 999 ONLY when verifying the restored original collection after the backup has been renamed back to target_name.
        def mock_dim(col):
            if backup_restored_to_target and col.name == target_name:
                return 999
            return orig_get_dim(col)

        with caplog.at_level(logging.CRITICAL):
            with patch.object(chromadb.api.models.Collection.Collection, "modify", autospec=True, side_effect=tracking_modify):
                with patch.object(self.manager, "get_collection_dimension", side_effect=mock_dim):
                    success = self.manager.safe_replace_collection(repo_id, candidate_chunks)
                    assert success is False

        # 5. Assertion proving the backup-restoration step was reached
        assert backup_restored_to_target is True
        assert re_backup_attempted is True
        assert len(backup_names) == 1
        backup_name = backup_names[0]

        # Verify logs explicitly state vectors remain under target_name and could not be renamed back to backup_name
        assert "CRITICAL" in caplog.text
        assert f"Original vectors remain under collection '{target_name}'" in caplog.text
        assert f"could not be renamed back to '{backup_name}'" in caplog.text

        # Verify original vectors remain intact and accessible under target_name
        col = self.manager.client.get_collection(target_name)
        assert col.count() == 2
        docs = col.get()["documents"]
        assert "original_doc_0" in docs
        assert "original_doc_1" in docs
        assert orig_get_dim(col) == 768



