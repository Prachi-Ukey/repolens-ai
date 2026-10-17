import os
import sys
import tempfile
import shutil
import uuid
import logging
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database import Base
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.ai.vector_store import ChromaManager
from app.ai.embeddings import MockEmbedding
from app.services.ingestion_service import reindex_existing_repository


class TestReindexConsistency:
    """
    Regression test suite for repository re-indexing database and vector consistency.
    Verifies that a failure during the database update/commit after successful Chroma
    replacement safely rolls back the Chroma collection and preserves the previous
    usable state, preventing split-brain inconsistencies and data loss.
    """

    def setup_method(self):
        # 1. Setup isolated in-memory SQLite database
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # 2. Setup isolated temporary directory for ChromaDB
        self.chroma_dir = tempfile.mkdtemp(prefix="chroma_reindex_test_")
        self.manager = ChromaManager(persist_dir=self.chroma_dir)
        self.manager.embedding_provider = MockEmbedding(dim=384)

    def teardown_method(self):
        if hasattr(self.manager, "client"):
            try:
                self.manager.client.close()
            except Exception:
                pass
        try:
            shutil.rmtree(self.chroma_dir, ignore_errors=True)
        except Exception:
            pass
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def _seed_repository(self, repo_id: str, old_chunks_count: int = 2):
        """Creates a repository in the test DB with files, chunks, and matching Chroma vectors."""
        db = self.Session()
        try:
            user = User(
                id=f"user_{uuid.uuid4().hex[:8]}",
                email=f"dev_{uuid.uuid4().hex[:8]}@example.com",
                username=f"user_{uuid.uuid4().hex[:8]}",
                hashed_password="hashed_pwd_test",
            )
            db.add(user)
            db.flush()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                owner="example",
                name="test-repo",
                url="https://github.com/example/test-repo",
                primary_language="python",
                file_count=1,
                chunk_count=old_chunks_count,
                status="completed",
                embedding_model="mock-embedding-384d",
                embedding_dimension=384,
            )
            db.add(repo)

            rf = RepositoryFile(
                repository_id=repo_id,
                file_path="src/main.py",
                file_name="main.py",
                language="python",
                size_bytes=200,
                line_count=20,
                sha_hash="old_sha_12345",
                content="def old_alpha():\n    return 1\n\ndef old_beta():\n    return 2\n",
            )
            db.add(rf)
            db.flush()

            old_chunk_objs = []
            old_chroma_chunks = []
            for i in range(old_chunks_count):
                chunk_id = f"old_chunk_{i}"
                content = f"def old_func_{i}():\n    return {i}"
                chunk_obj = CodeChunk(
                    repository_id=repo_id,
                    file_id=rf.id,
                    file_path="src/main.py",
                    language="python",
                    start_line=i * 5 + 1,
                    end_line=i * 5 + 4,
                    symbol_name=f"old_func_{i}",
                    chunk_id=chunk_id,
                    content=content,
                    sha_hash=f"sha_old_{i}",
                )
                old_chunk_objs.append(chunk_obj)
                old_chroma_chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "file_path": "src/main.py",
                    "language": "python",
                    "start_line": i * 5 + 1,
                    "end_line": i * 5 + 4,
                    "symbol_name": f"old_func_{i}",
                })

            db.bulk_save_objects(old_chunk_objs)
            db.commit()

            # Seed Chroma collection with old vectors
            self.manager.safe_replace_collection(repo_id, old_chroma_chunks)
            col = self.manager.client.get_collection(self.manager._get_collection_name(repo_id))
            assert col.count() == old_chunks_count
        finally:
            db.close()

    def test_reindex_sql_commit_failure_restores_previous_vectors_and_records(self):
        """
        Scenario 1:
        Chroma replacement succeeds, but the subsequent SQL commit fails.
        Verifies:
        - The DB transaction rolls back, preserving previous CodeChunk and RepositoryFile records.
        - The Chroma collection is rolled back and restored to previous usable vectors.
        - System avoids split-brain state.
        """
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_repository(repo_id, old_chunks_count=2)

        # Confirm initial state
        col_name = self.manager._get_collection_name(repo_id)
        col = self.manager.client.get_collection(col_name)
        assert col.count() == 2
        orig_docs = set(col.get()["documents"])
        assert "def old_func_0():\n    return 0" in orig_docs

        # Mock parse_file_into_chunks to generate new replacement chunks
        new_chunks = [
            {
                "chunk_id": f"new_chunk_{i}",
                "content": f"def new_feature_{i}():\n    return 'new_{i}'",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": i * 10 + 1,
                "end_line": i * 10 + 8,
                "symbol_name": f"new_feature_{i}",
            }
            for i in range(3)
        ]

        def failing_session_factory():
            s = self.Session()
            s.commit = MagicMock(side_effect=RuntimeError("Simulated database disk I/O failure during commit"))
            return s

        with patch("app.services.ingestion_service.SessionLocal", side_effect=failing_session_factory), \
             patch("app.services.ingestion_service.get_vector_store", return_value=self.manager), \
             patch("app.github.parser.parse_file_into_chunks", return_value=new_chunks):

            with pytest.raises(RuntimeError, match="Simulated database disk I/O failure"):
                reindex_existing_repository(repo_id)

        # VERIFICATION 1: Database records are preserved intact (not deleted, not updated)
        verify_db = self.Session()
        try:
            db_chunks = verify_db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).all()
            assert len(db_chunks) == 2
            db_chunk_ids = {c.chunk_id for c in db_chunks}
            assert "old_chunk_0" in db_chunk_ids
            assert "old_chunk_1" in db_chunk_ids
            assert "new_chunk_0" not in db_chunk_ids
        finally:
            verify_db.close()

        # VERIFICATION 2: Chroma vector collection was restored to previous usable data
        restored_col = self.manager.client.get_collection(col_name)
        assert restored_col.count() == 2
        restored_docs = set(restored_col.get()["documents"])
        assert "def old_func_0():\n    return 0" in restored_docs
        assert "def old_func_1():\n    return 1" in restored_docs
        assert "def new_feature_0():\n    return 'new_0'" not in restored_docs

        # VERIFICATION 3: No orphaned pending backups remain
        assert not self.manager.has_pending_replacement(repo_id)

    def test_reindex_recovery_failure_raises_explicit_critical_error(self, caplog):
        """
        Scenario 2:
        Chroma replacement succeeds, SQL commit fails, and recovery itself fails.
        Verifies:
        - System does not report success.
        - Raises explicit RuntimeError stating re-index failed AND vector store recovery failed.
        - Emits a CRITICAL log message.
        """
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_repository(repo_id, old_chunks_count=2)

        new_chunks = [
            {
                "chunk_id": "new_chunk_0",
                "content": "def new_broken(): pass",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "symbol_name": "new_broken",
            }
        ]

        def failing_session_factory():
            s = self.Session()
            s.commit = MagicMock(side_effect=RuntimeError("Simulated SQL commit failure"))
            return s

        with patch("app.services.ingestion_service.SessionLocal", side_effect=failing_session_factory), \
             patch("app.services.ingestion_service.get_vector_store", return_value=self.manager), \
             patch("app.github.parser.parse_file_into_chunks", return_value=new_chunks), \
             patch.object(self.manager, "rollback_replacement", return_value=False):

            with caplog.at_level(logging.CRITICAL):
                with pytest.raises(RuntimeError) as exc_info:
                    reindex_existing_repository(repo_id)

                assert "vector store recovery failed" in str(exc_info.value).lower()

            assert any("CRITICAL: Vector collection recovery failed" in record.message for record in caplog.records)

    def test_successful_reindex_replaces_old_records_without_duplicates(self):
        """
        Scenario 3:
        Successful re-index replaces previous records and vectors cleanly.
        Verifies:
        - Database chunk count reflects new chunks only (old chunks purged).
        - Chroma collection reflects new chunks only (no duplicates).
        - Vector store finalizes replacement and deletes backup.
        """
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self._seed_repository(repo_id, old_chunks_count=2)

        new_chunks = [
            {
                "chunk_id": f"replacement_chunk_{i}",
                "content": f"def replacement_func_{i}(): return {i * 10}",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": i * 4 + 1,
                "end_line": i * 4 + 3,
                "symbol_name": f"replacement_func_{i}",
            }
            for i in range(3)
        ]

        with patch("app.services.ingestion_service.SessionLocal", side_effect=self.Session), \
             patch("app.services.ingestion_service.get_vector_store", return_value=self.manager), \
             patch("app.github.parser.parse_file_into_chunks", return_value=new_chunks):

            result = reindex_existing_repository(repo_id)

        assert result["chunks_indexed"] == 3

        # Check DB
        verify_db = self.Session()
        try:
            chunks = verify_db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).all()
            assert len(chunks) == 3
            ids = {c.chunk_id for c in chunks}
            assert "replacement_chunk_0" in ids
            assert "replacement_chunk_1" in ids
            assert "replacement_chunk_2" in ids
            assert "old_chunk_0" not in ids
        finally:
            verify_db.close()

        # Check Chroma
        col = self.manager.client.get_collection(self.manager._get_collection_name(repo_id))
        assert col.count() == 3
        docs = set(col.get()["documents"])
        assert "def replacement_func_0(): return 0" in docs
        assert "def old_func_0():\n    return 0" not in docs

        # No pending backups
        assert not self.manager.has_pending_replacement(repo_id)

    def test_deferred_cleanup_and_rollback_direct_on_manager(self):
        """
        Unit test on ChromaManager deferred cleanup:
        - safe_replace_collection with deferred_cleanup=True retains backup.
        - rollback_replacement restores original vectors.
        """
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        target_name = self.manager._get_collection_name(repo_id)

        # Seed original
        orig_col = self.manager.client.create_collection(
            name=target_name,
            metadata={"dimension": 384, "model": "mock-embedding-384d"}
        )
        orig_col.add(
            ids=["orig_1", "orig_2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["orig doc 1", "orig doc 2"],
            metadatas=[{"chunk_id": "orig_1"}, {"chunk_id": "orig_2"}]
        )

        candidate_chunks = [
            {
                "chunk_id": "cand_1",
                "content": "candidate doc 1",
                "file_path": "a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
            }
        ]

        # Replace with deferred_cleanup=True
        success = self.manager.safe_replace_collection(repo_id, candidate_chunks, deferred_cleanup=True)
        assert success is True
        assert self.manager.has_pending_replacement(repo_id) is True

        # Target currently has candidate chunk
        active_col = self.manager.client.get_collection(target_name)
        assert active_col.count() == 1
        assert "candidate doc 1" in active_col.get()["documents"]

        # Now trigger rollback_replacement
        rolled_back = self.manager.rollback_replacement(repo_id)
        assert rolled_back is True
        assert self.manager.has_pending_replacement(repo_id) is False

        # Target has original 2 chunks again
        restored = self.manager.client.get_collection(target_name)
        assert restored.count() == 2
        restored_docs = set(restored.get()["documents"])
        assert "orig doc 1" in restored_docs
        assert "orig doc 2" in restored_docs
