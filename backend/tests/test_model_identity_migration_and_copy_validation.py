import os
import shutil
import tempfile
import uuid
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, text

from app.config import settings
from app.database import Base, SessionLocal, _run_schema_migrations
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.ai.embeddings import (
    MockEmbedding,
    LocalEmbedding,
    OpenAIEmbedding,
    GeminiEmbedding,
    get_active_embedding_model_name,
    get_embedding_provider,
)
from app.ai.vector_store import ChromaManager


class TestEmbeddingModelIdentity:
    """Verifies that active embedding model names are consistently and publicly accessible."""

    def test_mock_embedding_model_name_default_and_custom(self):
        mock_default = MockEmbedding(dim=384)
        assert mock_default.model_name == "mock-embedding-384d"
        assert mock_default.dimension == 384

        mock_custom = MockEmbedding(dim=384, model_name="custom-test-model")
        assert mock_custom.model_name == "custom-test-model"
        assert mock_custom.dimension == 384

    def test_active_embedding_model_name_helper(self, monkeypatch):
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
        monkeypatch.setattr(settings, "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2", raising=False)
        assert get_active_embedding_model_name() == "all-MiniLM-L6-v2"

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001", raising=False)
        assert get_active_embedding_model_name() == "gemini-embedding-001"

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
        monkeypatch.setattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small", raising=False)
        assert get_active_embedding_model_name() == "text-embedding-3-small"

    def test_chroma_manager_embedding_properties(self):
        temp_dir = tempfile.mkdtemp()
        try:
            manager = ChromaManager(persist_dir=temp_dir)
            manager.embedding_provider = MockEmbedding(dim=384, model_name="custom-provider-model")
            assert manager.embedding_model_name == "custom-provider-model"
            assert manager.model_name == "custom-provider-model"
            assert manager.embedding_dimension == 384

            # Verify collection creation records the actual model name in metadata
            repo_id = f"repo_{uuid.uuid4().hex[:8]}"
            col = manager.get_or_create_collection(repo_id)
            assert col.metadata.get("model") == "custom-provider-model"
            assert col.metadata.get("dimension") == 384
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_health_endpoint_includes_embedding_model(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding_model" in data
        assert data["embedding_model"] is not None
        assert data["embedding_model"] != "unknown"


class TestModelAwareReindexing:
    """Verifies dimension and model identity compatibility checks and auto-rebuild."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ChromaManager(persist_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_collection_compatible_same_dim_different_model(self):
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        self.manager.embedding_provider = MockEmbedding(dim=384, model_name="model-alpha-384")
        col = self.manager.get_or_create_collection(repo_id)
        col.upsert(
            ids=["doc_1"],
            embeddings=[[0.1] * 384],
            documents=["Sample document"],
            metadatas=[{"chunk_id": "doc_1"}],
        )

        # Same model and dimension -> Compatible
        assert self.manager.is_collection_compatible(col, repository_id=repo_id) is True

        # Switch active model to different model with identical 384 dimensions -> Incompatible
        self.manager.embedding_provider = MockEmbedding(dim=384, model_name="model-beta-384")
        assert self.manager.is_collection_compatible(col, repository_id=repo_id) is False

    def test_is_collection_compatible_legacy_missing_metadata(self):
        """Legacy collection with stored vectors but missing model metadata must not be claimed compatible."""
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        col_name = self.manager._get_collection_name(repo_id)
        # Create legacy collection with dimension metadata but missing model metadata
        col = self.manager.client.create_collection(name=col_name, metadata={"dimension": 384})
        col.upsert(
            ids=["legacy_1"],
            embeddings=[[0.05] * 384],
            documents=["legacy code content"],
            metadatas=[{"chunk_id": "legacy_1"}],
        )
        self.manager.embedding_provider = MockEmbedding(dim=384, model_name="all-MiniLM-L6-v2")

        # Without evidence of model compatibility, it must return False
        assert self.manager.is_collection_compatible(col, repository_id=repo_id) is False

    def test_is_collection_compatible_empty_collection(self):
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"
        col = self.manager.get_or_create_collection(repo_id)
        assert col.count() == 0
        assert self.manager.is_collection_compatible(col, repository_id=repo_id) is True

    def test_query_similar_auto_rebuilds_when_384d_model_changes(self):
        """When 384-d model changes to another 384-d model, query triggers safe auto-rebuild from DB."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())
        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"test_model_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/org/model-test",
                owner="org",
                name="model-test",
                status="completed",
                embedding_model="model-v1",
                embedding_dimension=384,
                file_count=1,
                chunk_count=1,
            )
            db.add(repo)
            db.commit()

            repo_file = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/index.py",
                file_name="index.py",
                language="python",
                size_bytes=50,
                line_count=3,
                sha_hash="hash1",
                content="def compute(): return 42\n",
            )
            db.add(repo_file)
            db.commit()

            chunk = CodeChunk(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_id=repo_file.id,
                file_path="src/index.py",
                language="python",
                start_line=1,
                end_line=3,
                symbol_name="compute",
                chunk_id="chunk_idx_1",
                content="def compute(): return 42\n",
                sha_hash="chunkhash1",
            )
            db.add(chunk)
            db.commit()

            # Seed Chroma collection under model-v1
            self.manager.embedding_provider = MockEmbedding(dim=384, model_name="model-v1")
            col = self.manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_idx_1"],
                embeddings=[[0.01] * 384],
                documents=["def compute(): return 42\n"],
                metadatas=[{"chunk_id": "chunk_idx_1", "file_path": "src/index.py"}],
            )
            assert col.metadata.get("model") == "model-v1"

            # Switch active provider to model-v2 (same dimension: 384)
            self.manager.embedding_provider = MockEmbedding(dim=384, model_name="model-v2")
            assert self.manager.is_collection_compatible(col, repository_id=repo_id) is False

            # Query should detect model incompatibility and auto-rebuild to model-v2
            results = self.manager.query_similar_chunks(repo_id, "compute")
            assert len(results) > 0
            assert "compute" in results[0]["content"]

            # Confirm collection was rebuilt with model-v2
            col_after = self.manager.get_or_create_collection(repo_id)
            assert col_after.metadata.get("model") == "model-v2"

            # Confirm DB record was updated to model-v2
            repo_after = db.query(Repository).filter(Repository.id == repo_id).first()
            assert repo_after.embedding_model == "model-v2"
        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()


class TestDatabaseMigrationFailures:
    """Verifies that schema migrations fail fast when leaving schema incompatible with ORM."""

    def test_schema_migration_fails_fast_on_incompatible_schema(self):
        # Create an in-memory SQLite database with an incompatible repositories table
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text("CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR)"))
            conn.commit()

        # Simulate a migration error (e.g. database execution failure during ALTER TABLE)
        with patch("sqlalchemy.engine.base.Connection.execute", side_effect=RuntimeError("Simulated ALTER TABLE lock")):
            with pytest.raises(RuntimeError) as exc_info:
                _run_schema_migrations(test_engine)
            assert "Database schema migration failed" in str(exc_info.value)
            assert "embedding_model" in str(exc_info.value)

    def test_schema_migration_succeeds_idempotently(self):
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text("CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR)"))
            conn.commit()

        # Run migration first time: adds embedding_model and embedding_dimension
        _run_schema_migrations(test_engine)

        # Run migration second time: idempotent, no error
        _run_schema_migrations(test_engine)

    def test_legacy_repositories_assigned_unknown_and_null_dimension(self):
        """Legacy repository records must receive 'unknown' and NULL dimension, not assuming MiniLM/384."""
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text("CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR)"))
            conn.execute(text("INSERT INTO repositories (id, name) VALUES ('repo_leg_1', 'legacy_project')"))
            conn.commit()

        _run_schema_migrations(test_engine)

        with test_engine.connect() as conn:
            row = conn.execute(text("SELECT embedding_model, embedding_dimension FROM repositories WHERE id = 'repo_leg_1'")).fetchone()
            assert row is not None
            assert row[0] == "unknown"
            assert row[1] is None

    def test_migration_preserves_existing_embedding_metadata_without_overwrite(self):
        """Records with existing embedding metadata are never overwritten by migration."""
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text(
                "CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR, "
                "embedding_model VARCHAR, embedding_dimension INTEGER)"
            ))
            conn.execute(text(
                "INSERT INTO repositories (id, name, embedding_model, embedding_dimension) "
                "VALUES ('repo_gemini', 'gemini_project', 'gemini-embedding-001', 768)"
            ))
            conn.execute(text(
                "INSERT INTO repositories (id, name, embedding_model, embedding_dimension) "
                "VALUES ('repo_null', 'null_project', NULL, NULL)"
            ))
            conn.commit()

        _run_schema_migrations(test_engine)

        with test_engine.connect() as conn:
            gemini_row = conn.execute(text("SELECT embedding_model, embedding_dimension FROM repositories WHERE id = 'repo_gemini'")).fetchone()
            assert gemini_row[0] == "gemini-embedding-001"
            assert gemini_row[1] == 768

            null_row = conn.execute(text("SELECT embedding_model, embedding_dimension FROM repositories WHERE id = 'repo_null'")).fetchone()
            assert null_row[0] == "unknown"
            assert null_row[1] is None

    def test_partially_completed_migration_completes_safely(self):
        """A database with embedding_model already added completes cleanly when adding embedding_dimension."""
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text("CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR, embedding_model VARCHAR)"))
            conn.execute(text("INSERT INTO repositories (id, name, embedding_model) VALUES ('repo_p', 'partial_project', 'custom-model')"))
            conn.commit()

        _run_schema_migrations(test_engine)

        from sqlalchemy import inspect
        cols = {c["name"] for c in inspect(test_engine).get_columns("repositories")}
        assert "embedding_model" in cols
        assert "embedding_dimension" in cols

        with test_engine.connect() as conn:
            row = conn.execute(text("SELECT embedding_model, embedding_dimension FROM repositories WHERE id = 'repo_p'")).fetchone()
            assert row[0] == "custom-model"
            assert row[1] is None

    def test_migration_retry_after_failure(self):
        """If a migration fails on first attempt, a retry succeeds cleanly once failure cause is resolved."""
        test_engine = create_engine("sqlite:///:memory:")
        with test_engine.connect() as conn:
            conn.execute(text("CREATE TABLE repositories (id VARCHAR PRIMARY KEY, name VARCHAR)"))
            conn.commit()

        # First run fails due to simulated error
        with patch("sqlalchemy.engine.base.Connection.execute", side_effect=RuntimeError("Transient lock")):
            with pytest.raises(RuntimeError):
                _run_schema_migrations(test_engine)

        # Retry runs normally and succeeds
        _run_schema_migrations(test_engine)

        from sqlalchemy import inspect
        cols = {c["name"] for c in inspect(test_engine).get_columns("repositories")}
        assert "embedding_model" in cols
        assert "embedding_dimension" in cols

    def test_new_repository_defaults_from_application_config(self, monkeypatch):
        """Newly instantiated Repository records default to current application configuration."""
        from app.models.repository import Repository

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
        monkeypatch.setattr(settings, "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 384)

        repo_local = Repository(user_id="u1", url="http://example.com", owner="o", name="n1")
        assert repo_local.embedding_model == "all-MiniLM-L6-v2"
        assert repo_local.embedding_dimension == 384

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 768)

        repo_gemini = Repository(user_id="u2", url="http://example.com", owner="o", name="n2")
        assert repo_gemini.embedding_model == "gemini-embedding-001"
        assert repo_gemini.embedding_dimension == 768



class TestCollectionCopyValidation:
    """Verifies strict validation in _copy_collection_data."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ChromaManager(persist_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_copy_fails_safely_when_source_lacks_usable_embeddings(self):
        src_name = f"src_{uuid.uuid4().hex[:8]}"
        dest_name = f"dest_{uuid.uuid4().hex[:8]}"
        src_col = self.manager.client.create_collection(name=src_name)
        dest_col = self.manager.client.create_collection(name=dest_name)

        # Mock src_col.get to return documents without embeddings
        with patch.object(src_col, "count", return_value=1):
            with patch.object(self.manager, "get_collection_dimension", return_value=384):
                with patch.object(src_col, "get", return_value={
                    "ids": ["item_1"],
                    "embeddings": None,
                    "documents": ["doc content"],
                    "metadatas": [{"k": "v"}],
                }):
                    success = self.manager._copy_collection_data(src_col, dest_col)
                    assert success is False
                    assert dest_col.count() == 0

    def test_copy_fails_when_vector_dimensions_inconsistent(self):
        src_name = f"src_{uuid.uuid4().hex[:8]}"
        dest_name = f"dest_{uuid.uuid4().hex[:8]}"
        src_col = self.manager.client.create_collection(name=src_name)
        dest_col = self.manager.client.create_collection(name=dest_name)

        # Vector in batch has wrong dimension (128 instead of 384)
        with patch.object(src_col, "count", return_value=1):
            with patch.object(self.manager, "get_collection_dimension", return_value=384):
                with patch.object(src_col, "get", return_value={
                    "ids": ["item_1"],
                    "embeddings": [[0.1] * 128],
                    "documents": ["doc content"],
                    "metadatas": [{"k": "v"}],
                }):
                    success = self.manager._copy_collection_data(src_col, dest_col)
                    assert success is False
                    assert dest_col.count() == 0

    def test_copy_fails_when_metadata_or_docs_length_inconsistent(self):
        src_name = f"src_{uuid.uuid4().hex[:8]}"
        dest_name = f"dest_{uuid.uuid4().hex[:8]}"
        src_col = self.manager.client.create_collection(name=src_name)
        dest_col = self.manager.client.create_collection(name=dest_name)

        # 2 IDs but only 1 metadata dict
        with patch.object(src_col, "count", return_value=2):
            with patch.object(self.manager, "get_collection_dimension", return_value=384):
                with patch.object(src_col, "get", return_value={
                    "ids": ["item_1", "item_2"],
                    "embeddings": [[0.1] * 384, [0.2] * 384],
                    "documents": ["doc 1", "doc 2"],
                    "metadatas": [{"k": "v"}],  # Mismatched length
                }):
                    success = self.manager._copy_collection_data(src_col, dest_col)
                    assert success is False
                    assert dest_col.count() == 0

    def test_copy_succeeds_when_all_data_consistent_and_verified(self):
        src_name = f"src_{uuid.uuid4().hex[:8]}"
        dest_name = f"dest_{uuid.uuid4().hex[:8]}"
        src_col = self.manager.client.create_collection(name=src_name)
        dest_col = self.manager.client.create_collection(name=dest_name)

        src_col.add(
            ids=["item_1", "item_2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["doc 1", "doc 2"],
            metadatas=[{"k": "1"}, {"k": "2"}],
        )

        success = self.manager._copy_collection_data(src_col, dest_col)
        assert success is True
        assert dest_col.count() == 2
        assert self.manager.get_collection_dimension(dest_col) == 384
        dest_data = dest_col.get(include=["documents", "metadatas"])
        assert dest_data["documents"] == ["doc 1", "doc 2"]
