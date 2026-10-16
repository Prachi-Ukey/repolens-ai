import shutil
import tempfile
import uuid
import pytest
from unittest.mock import patch, MagicMock

from app.database import Base, SessionLocal, engine
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.ai.embeddings import MockEmbedding
from app.ai.vector_store import ChromaManager


class TestRebuildSafetyAndQueryVerification:
    """
    Regression tests covering:
    1. Query after failed rebuild: verifying no query is issued against an incompatible collection.
    2. Incomplete or unusable stored-file data: verifying existing collection is preserved intact.
    3. Safe fallback to complete CodeChunk records when files are incomplete.
    4. Valid rebuild: verifying collection is replaced successfully with active model and dimension.
    """

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ChromaManager(persist_dir=self.temp_dir)
        self.manager.embedding_provider = MockEmbedding(dim=384, model_name="active-model-384d")
        Base.metadata.create_all(bind=engine)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_query_after_failed_rebuild_aborts_without_querying_incompatible_collection(self):
        """
        When collection is incompatible and rebuild fails, query_similar_chunks must log an error,
        return empty results, and NEVER issue a query against the incompatible collection.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"query_fail_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-incompatible",
                owner="org",
                name="test-incompatible",
                status="completed",
                embedding_model="old-model-128d",
                embedding_dimension=128,
                file_count=1,
                chunk_count=1,
            )
            db.add(repo)
            db.commit()

            # Seed Chroma collection with 128-dimensional vectors under old-model
            legacy_provider = MockEmbedding(dim=128, model_name="old-model-128d")
            legacy_manager = ChromaManager(persist_dir=self.temp_dir)
            legacy_manager.embedding_provider = legacy_provider
            col = legacy_manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_legacy_1"],
                embeddings=[[0.05] * 128],
                documents=["def legacy_code(): pass"],
                metadatas=[{"file_path": "legacy.py", "chunk_id": "chunk_legacy_1"}],
            )
            assert col.count() == 1
            assert legacy_manager.get_collection_dimension(col) == 128

            # Active manager (384d) checks compatibility -> incompatible
            assert self.manager.is_collection_compatible(col, repository_id=repo_id) is False

            # Simulate rebuild failure
            with patch.object(self.manager, "rebuild_collection_from_db", return_value=False) as mock_rebuild:
                with patch.object(col, "query", wraps=col.query) as mock_query:
                    results = self.manager.query_similar_chunks(repo_id, "search query")
                    assert results == []
                    # Verify rebuild was attempted
                    assert mock_rebuild.called
                    # Crucial: verify that no query was issued against the incompatible collection!
                    assert not mock_query.called

            # Also verify scenario where rebuild returns True but collection remains incompatible
            with patch.object(self.manager, "rebuild_collection_from_db", return_value=True):
                with patch.object(self.manager, "is_collection_compatible", return_value=False):
                    with patch.object(col, "query", wraps=col.query) as mock_query2:
                        results = self.manager.query_similar_chunks(repo_id, "search query")
                        assert results == []
                        assert not mock_query2.called

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()

    def test_incomplete_stored_files_count_preserves_existing_collection(self):
        """
        When stored files count is less than repository's recorded file_count and chunks
        are missing, rebuild must abort and preserve the existing Chroma collection.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"incomplete_f_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            # Recorded file_count = 3, chunk_count = 3
            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-partial-files",
                owner="org",
                name="test-partial-files",
                status="completed",
                file_count=3,
                chunk_count=3,
            )
            db.add(repo)
            db.commit()

            # Stored files only has 1 file (missing 2 files)
            file1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/file1.py",
                file_name="file1.py",
                language="python",
                size_bytes=30,
                line_count=2,
                sha_hash="h1",
                content="def func1(): return 1\n",
            )
            db.add(file1)
            db.commit()

            # Existing Chroma collection has 3 vectors
            col = self.manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_1", "chunk_2", "chunk_3"],
                embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
                documents=["doc 1", "doc 2", "doc 3"],
                metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
            )
            assert col.count() == 3

            # Attempt rebuild -> should fail because file set is incomplete (1 < 3) and no CodeChunk records exist
            rebuilt = self.manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is False

            # Existing collection must be preserved intact with all 3 vectors
            col_after = self.manager.get_or_create_collection(repo_id)
            assert col_after.count() == 3
            data = col_after.get()["documents"]
            assert set(data) == {"doc 1", "doc 2", "doc 3"}

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()

    def test_unusable_stored_files_null_content_preserves_existing_collection(self):
        """
        When stored files have NULL content and no CodeChunk records exist,
        rebuild must abort and preserve the existing Chroma collection.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"null_content_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-null-content",
                owner="org",
                name="test-null-content",
                status="completed",
                file_count=2,
                chunk_count=2,
            )
            db.add(repo)
            db.commit()

            # Stored files have NULL content
            f1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/a.py",
                file_name="a.py",
                language="python",
                size_bytes=100,
                line_count=10,
                content=None,  # Unusable NULL content
            )
            f2 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/b.py",
                file_name="b.py",
                language="python",
                size_bytes=100,
                line_count=10,
                content=None,
            )
            db.add_all([f1, f2])
            db.commit()

            col = self.manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_orig_1", "chunk_orig_2"],
                embeddings=[[0.05] * 384, [0.06] * 384],
                documents=["def original_a(): pass", "def original_b(): pass"],
                metadatas=[{"k": "a"}, {"k": "b"}],
            )
            assert col.count() == 2

            rebuilt = self.manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is False

            col_after = self.manager.get_or_create_collection(repo_id)
            assert col_after.count() == 2
            assert "def original_a(): pass" in col_after.get()["documents"]

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()

    def test_parsed_chunks_fewer_than_existing_collection_preserves_collection(self):
        """
        When parsing stored files produces fewer chunks than the populated Chroma collection,
        rebuild must abort rather than silently truncating the collection.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"trunc_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-trunc",
                owner="org",
                name="test-trunc",
                status="completed",
                file_count=1,
                chunk_count=3,
            )
            db.add(repo)
            db.commit()

            # Stored file only parses into 1 chunk
            file1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/one.py",
                file_name="one.py",
                language="python",
                size_bytes=30,
                line_count=2,
                content="def one(): return 1\n",
            )
            db.add(file1)
            db.commit()

            # Existing collection has 3 vectors
            col = self.manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_1", "chunk_2", "chunk_3"],
                embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
                documents=["vec 1", "vec 2", "vec 3"],
                metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
            )
            assert col.count() == 3

            # Rebuild attempt: file produces 1 chunk (< 3 existing vectors) -> must abort
            rebuilt = self.manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is False

            # Existing collection remains unchanged with 3 vectors
            col_after = self.manager.get_or_create_collection(repo_id)
            assert col_after.count() == 3
            assert set(col_after.get()["documents"]) == {"vec 1", "vec 2", "vec 3"}

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()

    def test_safe_fallback_to_complete_code_chunks_when_files_incomplete(self):
        """
        When RepositoryFile records are incomplete or unusable, but CodeChunk records are complete
        and validated, rebuild safely falls back to CodeChunk records.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"fallback_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-fallback",
                owner="org",
                name="test-fallback",
                status="completed",
                embedding_model="old-model",
                embedding_dimension=128,
                file_count=2,
                chunk_count=2,
            )
            db.add(repo)
            db.commit()

            # Stored files have NULL content (unusable for file-based rebuild)
            f1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/m.py",
                file_name="m.py",
                language="python",
                content=None,
            )
            db.add(f1)
            db.commit()

            # But CodeChunk records ARE complete and valid for all 2 chunks
            c1 = CodeChunk(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_id=f1.id,
                file_path="src/m.py",
                language="python",
                start_line=1,
                end_line=5,
                symbol_name="func_m1",
                chunk_id="chunk_m1",
                content="def func_m1(): return 'm1'\n",
            )
            c2 = CodeChunk(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_id=f1.id,
                file_path="src/m.py",
                language="python",
                start_line=6,
                end_line=10,
                symbol_name="func_m2",
                chunk_id="chunk_m2",
                content="def func_m2(): return 'm2'\n",
            )
            db.add_all([c1, c2])
            db.commit()

            # Existing collection has 2 vectors under old dimension (128d)
            legacy_manager = ChromaManager(persist_dir=self.temp_dir)
            legacy_manager.embedding_provider = MockEmbedding(dim=128, model_name="old-model")
            col = legacy_manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_m1", "chunk_m2"],
                embeddings=[[0.01] * 128, [0.02] * 128],
                documents=["def func_m1(): return 'm1'\n", "def func_m2(): return 'm2'\n"],
                metadatas=[{"chunk_id": "chunk_m1"}, {"chunk_id": "chunk_m2"}],
            )
            assert col.count() == 2
            assert legacy_manager.get_collection_dimension(col) == 128

            # Active manager (384d) triggers rebuild -> falls back to CodeChunk and succeeds
            rebuilt = self.manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is True

            col_rebuilt = self.manager.get_or_create_collection(repo_id)
            assert col_rebuilt.count() == 2
            assert self.manager.get_collection_dimension(col_rebuilt) == 384
            assert col_rebuilt.metadata.get("model") == "active-model-384d"

            db.refresh(repo)
            assert repo.embedding_dimension == 384
            assert repo.embedding_model == "active-model-384d"

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()

    def test_valid_file_based_rebuild_replaces_collection_successfully(self):
        """
        When stored files are complete and valid, rebuild succeeds, updates collection vectors
        and dimension, and query_similar_chunks retrieves chunks accurately.
        """
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"valid_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/test-valid-rebuild",
                owner="org",
                name="test-valid-rebuild",
                status="completed",
                embedding_model="old-model",
                embedding_dimension=128,
                file_count=1,
                chunk_count=2,
            )
            db.add(repo)
            db.commit()

            code = (
                "def calculate_total(a, b):\n"
                "    return a + b\n\n"
                "def calculate_discount(price, pct):\n"
                "    return price * (1 - pct)\n"
            )
            file1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/billing.py",
                file_name="billing.py",
                language="python",
                size_bytes=len(code),
                line_count=5,
                content=code,
            )
            db.add(file1)
            db.commit()

            # Seed old 128d Chroma collection
            legacy_manager = ChromaManager(persist_dir=self.temp_dir)
            legacy_manager.embedding_provider = MockEmbedding(dim=128, model_name="old-model")
            col = legacy_manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_old_1", "chunk_old_2"],
                embeddings=[[0.05] * 128, [0.06] * 128],
                documents=["def calculate_total(): pass", "def calculate_discount(): pass"],
                metadatas=[{"chunk_id": "chunk_old_1"}, {"chunk_id": "chunk_old_2"}],
            )
            assert col.count() == 2
            assert legacy_manager.get_collection_dimension(col) == 128

            # Rebuild via active manager (384d)
            rebuilt = self.manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is True

            col_rebuilt = self.manager.get_or_create_collection(repo_id)
            assert col_rebuilt.count() == 2
            assert self.manager.get_collection_dimension(col_rebuilt) == 384
            assert col_rebuilt.metadata.get("model") == "active-model-384d"

            db.refresh(repo)
            assert repo.embedding_dimension == 384
            assert repo.embedding_model == "active-model-384d"

            # Query should now work seamlessly against the new 384-d collection
            results = self.manager.query_similar_chunks(repo_id, "calculate discount")
            assert len(results) > 0
            found_discount = any("calculate_discount" in r["content"] for r in results)
            assert found_discount is True

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            db.close()
