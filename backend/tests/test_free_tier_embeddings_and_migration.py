"""
Comprehensive test suite for RepoLens AI Free-Tier Strategy, Local Embeddings,
Vector Store Dimension Migration, and Quota Handling.
"""

import os
import sys
import uuid
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.config import Settings
from app.database import SessionLocal, Base, engine, init_db
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.ai.embeddings import (
    LocalEmbedding,
    GeminiEmbedding,
    MockEmbedding,
    get_embedding_provider,
    _cached_local_model,
)
from app.ai.llm import GeminiLLM, get_llm_provider
from app.ai.vector_store import ChromaManager


class TestFreeTierLocalEmbedding:
    """Test LocalEmbedding lazy singleton loading, batching, determinism, and error handling."""

    def test_local_embedding_properties(self):
        local_emb = LocalEmbedding()
        assert local_emb.dimension == 384
        assert local_emb.model_name == "all-MiniLM-L6-v2"

    def test_local_embedding_lazy_singleton_reuse(self):
        """Verify that multiple LocalEmbedding instances share the same cached model."""
        inst1 = LocalEmbedding()
        inst2 = LocalEmbedding()

        # Both instances have the same configuration
        assert inst1.dimension == inst2.dimension == 384
        assert inst1.model_name == inst2.model_name == "all-MiniLM-L6-v2"

        # Model instance is lazily loaded and identical
        model1 = inst1.model
        model2 = inst2.model
        assert model1 is model2

    def test_local_embedding_document_and_query_deterministic(self):
        """Verify deterministic vector generation with 384 dimensions."""
        local_emb = LocalEmbedding(batch_size=16)
        docs = ["def test_func(): return True", "import os, sys"]

        embeddings = local_emb.embed_documents(docs)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384

        # Determinism check: embedding the same text twice yields identical vectors
        embeddings_again = local_emb.embed_documents(docs)
        for i in range(len(embeddings[0])):
            assert abs(embeddings[0][i] - embeddings_again[0][i]) < 1e-6

        query_vec = local_emb.embed_query("test function")
        assert len(query_vec) == 384
        query_again = local_emb.embed_query("test function")
        for i in range(len(query_vec)):
            assert abs(query_vec[i] - query_again[i]) < 1e-6

    def test_local_embedding_handles_empty_inputs(self):
        local_emb = LocalEmbedding()
        assert local_emb.embed_documents([]) == []
        query_vec = local_emb.embed_query("")
        assert len(query_vec) == 384
        assert all(v == 0.0 for v in query_vec)

    def test_local_embedding_handles_download_failure_actionably(self, monkeypatch):
        """Verify that model loading failures raise actionable errors without silent mock substitution."""
        import app.ai.embeddings as emb_mod
        
        # Reset cache for test
        orig_cache = emb_mod._cached_local_model
        emb_mod._cached_local_model = None
        try:
            def mock_fail_st(name):
                raise ConnectionError("Hugging Face hub unreachable")

            monkeypatch.setattr(emb_mod, "SentenceTransformer", mock_fail_st)
            failing_emb = LocalEmbedding()

            with pytest.raises(RuntimeError) as exc_info:
                failing_emb.embed_documents(["sample code"])

            err_text = str(exc_info.value)
            assert "Failed to load or download local embedding model" in err_text
            assert "Hugging Face" in err_text
        finally:
            emb_mod._cached_local_model = orig_cache


class TestGeminiQuotaAndSecurityHandling:
    """Test Gemini embedding and LLM security, header authentication, and 429 quota handling."""

    def test_gemini_embedding_rate_limit_429_actionable_error(self):
        """HTTP 429 RESOURCE_EXHAUSTED must raise a descriptive error without leaking keys."""
        api_key = "test-secret-key-gemini-999"
        gemini = GeminiEmbedding(api_key=api_key)

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_resp.text = f'{{"error": {{"code": 429, "message": "Resource exhausted for key {api_key}", "status": "RESOURCE_EXHAUSTED"}}}}'

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc_info:
                gemini.embed_documents(["code chunk to embed"])

            err_msg = str(exc_info.value)
            assert "RESOURCE_EXHAUSTED" in err_msg
            assert "EMBEDDING_PROVIDER=local" in err_msg
            assert api_key not in err_msg
            assert "[REDACTED]" in err_msg or "test-secret-key" not in err_msg

    def test_gemini_llm_header_authentication_and_rate_limit(self):
        """GeminiLLM must pass API key in x-goog-api-key header and handle 429 gracefully."""
        api_key = "test-secret-key-llm-888"
        gemini_llm = GeminiLLM(api_key=api_key, model="gemini-2.5-flash")

        captured_requests = []

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_requests.append({"url": url, "headers": headers or {}})
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.headers = {}
            mock_resp.text = "RESOURCE_EXHAUSTED"
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            res = gemini_llm.generate_response("Sys", "User")
            assert "RESOURCE_EXHAUSTED" in res or "rate limit reached" in res
            assert len(captured_requests) > 0
            req = captured_requests[0]
            # Key must NOT be in URL
            assert "key=" not in req["url"]
            # Key MUST be in headers
            assert req["headers"].get("x-goog-api-key") == api_key


class TestVectorStoreDimensionMigration:
    """Test vector store dimension mismatch detection, safe migration, and DB metadata recording."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self):
        self.temp_chroma_dir = tempfile.mkdtemp(prefix="repolens_free_tier_test_")
        yield
        if os.path.exists(self.temp_chroma_dir):
            shutil.rmtree(self.temp_chroma_dir, ignore_errors=True)

    def test_rebuild_incompatible_collection_from_768_to_384(self):
        """Simulate migrating an existing 768-d Gemini collection to 384-d local embeddings."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())

        try:
            # Setup DB repo and chunk records
            user = User(
                id=str(uuid.uuid4()),
                email=f"migrate_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="hashed_pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/testorg/migratetest",
                owner="testorg",
                name="migratetest",
                status="completed",
                embedding_model="gemini-embedding-001",
                embedding_dimension=768,
                file_count=1,
                chunk_count=1,
            )
            db.add(repo)
            db.commit()

            file_id = str(uuid.uuid4())
            repo_file = RepositoryFile(
                id=file_id,
                repository_id=repo.id,
                file_path="app/main.py",
                file_name="main.py",
                language="python",
                size_bytes=80,
                line_count=5,
                sha_hash="mainhash",
                content="def start_app():\n    return 'running'\n",
            )
            db.add(repo_file)
            db.commit()

            chunk = CodeChunk(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_id=repo_file.id,
                file_path="app/main.py",
                language="python",
                start_line=1,
                end_line=5,
                symbol_name="start_app",
                chunk_id="chunk_migrate_1",
                content="def start_app():\n    return 'running'\n",
                sha_hash="chunkhash",
            )
            db.add(chunk)
            db.commit()

            # Seed collection with 768-d vectors (simulating legacy Gemini deployment)
            legacy_manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            legacy_manager.embedding_provider = MockEmbedding(dim=768)
            col_768 = legacy_manager.get_or_create_collection(repo_id)
            col_768.upsert(
                ids=["chunk_migrate_1"],
                embeddings=[[0.02] * 768],
                documents=["def start_app():\n    return 'running'\n"],
                metadatas=[{
                    "file_path": "app/main.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 5,
                    "symbol_name": "start_app",
                    "chunk_id": "chunk_migrate_1",
                }],
            )
            assert legacy_manager.get_collection_dimension(col_768) == 768

            # Switch active provider to 384-d LocalEmbedding (or 384-d mock for test environment)
            active_manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            active_manager.embedding_provider = MockEmbedding(dim=384)

            # Query should detect dimension mismatch and safely auto-rebuild to 384
            results = active_manager.query_similar_chunks(repo_id, "start application")
            assert len(results) > 0
            assert "start_app" in results[0]["content"]

            # Confirm collection dimension is now 384
            col_migrated = active_manager.get_or_create_collection(repo_id)
            assert active_manager.get_collection_dimension(col_migrated) == 384

            # Confirm SQLite repository record was safely updated
            repo_after = db.query(Repository).filter(Repository.id == repo_id).first()
            assert repo_after.embedding_dimension == 384

        finally:
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo_id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).delete()
            db.query(Repository).filter(Repository.id == repo_id).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()


class TestProductionFreeConfigValidation:
    """Test that production settings cleanly support free local embeddings without API keys."""

    def test_production_succeeds_with_local_embedding_provider_and_no_gemini_key(self):
        """In production, EMBEDDING_PROVIDER='local' must not require GEMINI_API_KEY."""
        s = Settings(
            APP_ENV="production",
            SECRET_KEY="super-secret-random-prod-key-12345",
            EMBEDDING_PROVIDER="local",
            EMBEDDING_DIMENSION=384,
            LLM_PROVIDER="mock",
            GEMINI_API_KEY=None,
        )
        assert s.APP_ENV == "production"
        assert s.EMBEDDING_PROVIDER == "local"
        assert s.EMBEDDING_DIMENSION == 384
        assert s.GEMINI_API_KEY is None

    def test_production_fails_fast_when_gemini_embedding_has_no_key(self):
        """In production, EMBEDDING_PROVIDER='gemini' must fail fast if GEMINI_API_KEY is missing."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be configured when EMBEDDING_PROVIDER='gemini'"):
            Settings(
                APP_ENV="production",
                SECRET_KEY="super-secret-random-prod-key-12345",
                EMBEDDING_PROVIDER="gemini",
                GEMINI_API_KEY=None,
            )
