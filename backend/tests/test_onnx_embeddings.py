import os
import sys
import uuid
import tempfile
import shutil
import pytest
import numpy as np

from app.config import Settings, settings
from app.ai.embeddings import (
    ONNXLocalEmbedding,
    ONNXEmbedding,
    get_embedding_provider,
    get_active_embedding_model_name,
)
from app.ai.vector_store import ChromaManager


class TestONNXLocalEmbeddingUnit:
    """Targeted unit tests for ONNX runtime quantized embeddings."""

    @pytest.fixture(autouse=True)
    def setup_provider(self):
        self.provider = ONNXLocalEmbedding(batch_size=4)

    def test_dimensions_dtype_and_model_name(self):
        assert self.provider.dimension == 384
        assert self.provider.model_name == "all-MiniLM-L6-v2-qint8"

        docs = ["def calculate_profit(revenue, costs): return revenue - costs"]
        embeddings = self.provider.embed_documents(docs)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert isinstance(embeddings[0][0], float)

    def test_l2_normalization(self):
        docs = [
            "Repository analysis and code intelligence assistant.",
            "FastAPI endpoint returning JSON response.",
        ]
        embeddings = self.provider.embed_documents(docs)
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert norm == pytest.approx(1.0, rel=1e-4)

    def test_deterministic_output(self):
        text = "class IngestionService: pass"
        vec1 = self.provider.embed_query(text)
        vec2 = self.provider.embed_query(text)
        np.testing.assert_allclose(vec1, vec2, rtol=1e-5, atol=1e-5)

    def test_empty_and_whitespace_inputs(self):
        # Empty document list
        assert self.provider.embed_documents([]) == []

        # Empty and whitespace documents
        docs = ["", "   ", "\n\t"]
        embeddings = self.provider.embed_documents(docs)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
            assert np.linalg.norm(emb) == pytest.approx(1.0, rel=1e-4)

        # Empty query
        query_vec = self.provider.embed_query("")
        assert query_vec == [0.0] * 384
        query_vec_spaces = self.provider.embed_query("   ")
        assert query_vec_spaces == [0.0] * 384

    def test_oversized_input_truncation(self):
        # Generate 1,000 words, far exceeding the 256 token limit
        long_text = "def function_with_many_tokens():\n" + "    value = 42\n" * 500
        embeddings = self.provider.embed_documents([long_text])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0, rel=1e-4)

    def test_batch_processing_and_custom_batch_size(self):
        # 9 items with batch_size=4 should produce 3 batches (4, 4, 1)
        emb_small_batch = ONNXLocalEmbedding(batch_size=4)
        docs = [f"Code snippet number {i} for batching verification." for i in range(9)]
        results = emb_small_batch.embed_documents(docs)
        assert len(results) == 9
        for vec in results:
            assert len(vec) == 384

    def test_semantic_ranking_fidelity(self):
        target = "authenticate user with JWT bearer token"
        candidate_matching = "def verify_jwt_token(token: str): return decode_jwt(token)"
        candidate_unrelated = "def calculate_matrix_determinant(matrix): return det(matrix)"

        target_vec = self.provider.embed_query(target)
        match_vec = self.provider.embed_query(candidate_matching)
        unrelated_vec = self.provider.embed_query(candidate_unrelated)

        sim_match = np.dot(target_vec, match_vec)
        sim_unrelated = np.dot(target_vec, unrelated_vec)

        assert sim_match > sim_unrelated
        assert sim_match > 0.40


class TestONNXProviderConfigurationAndIsolation:
    """Verify provider selection, settings, and runtime isolation."""

    def test_onnx_local_provider_selection(self, monkeypatch):
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "onnx_local")
        monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 384)

        provider = get_embedding_provider()
        assert isinstance(provider, ONNXLocalEmbedding)
        assert provider.dimension == 384
        assert get_active_embedding_model_name() == "all-MiniLM-L6-v2-qint8"

    def test_onnx_alias_provider_selection(self, monkeypatch):
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "onnx")
        monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 384)

        provider = get_embedding_provider()
        assert isinstance(provider, ONNXLocalEmbedding)
        assert provider.dimension == 384

    def test_settings_preserves_384_dimension_for_onnx(self):
        s = Settings(
            APP_ENV="development",
            EMBEDDING_PROVIDER="onnx_local",
        )
        assert s.EMBEDDING_PROVIDER == "onnx_local"
        assert s.EMBEDDING_DIMENSION == 384

    def test_production_onnx_requires_runtime_packages(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ValueError, match="onnxruntime and tokenizers are not installed"):
            Settings(
                APP_ENV="production",
                SECRET_KEY="strong-valid-secret-key-123456",
                EMBEDDING_PROVIDER="onnx_local",
                LLM_PROVIDER="gemini",
                GEMINI_API_KEY="test-valid-key",
            )

    def test_onnx_runtime_isolation_no_pytorch_import(self):
        """Run in a clean subprocess to guarantee torch and sentence_transformers are NOT imported."""
        import subprocess

        cmd = [
            sys.executable,
            "-c",
            "import sys; "
            "from app.ai.embeddings import ONNXLocalEmbedding; "
            "emb = ONNXLocalEmbedding(); "
            "vec = emb.embed_query('clean isolation probe'); "
            "assert len(vec) == 384; "
            "assert 'torch' not in sys.modules, 'torch leaked into sys.modules'; "
            "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers leaked into sys.modules'; "
            "print('ISOLATION_OK')",
        ]
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = subprocess.run(cmd, cwd=backend_dir, capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess failed:\nstdout: {res.stdout}\nstderr: {res.stderr}"
        assert "ISOLATION_OK" in res.stdout


class TestONNXChromaDBIntegration:
    """Verify ONNX provider with live ChromaDB storage, querying, and safe replacement."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self):
        self.temp_chroma = tempfile.mkdtemp(prefix="repolens_onnx_chroma_")
        self.manager = ChromaManager(persist_dir=self.temp_chroma)
        self.manager.embedding_provider = ONNXLocalEmbedding(batch_size=4)
        yield
        if os.path.exists(self.temp_chroma):
            shutil.rmtree(self.temp_chroma, ignore_errors=True)

    def test_chroma_collection_creation_and_retrieval(self):
        repo_id = str(uuid.uuid4())
        col = self.manager.get_or_create_collection(repo_id)

        chunks = [
            {
                "chunk_id": "c1",
                "file_path": "auth.py",
                "language": "python",
                "start_line": 1,
                "end_line": 10,
                "symbol_name": "login",
                "content": "def login(username, password): return authenticate(username, password)",
            },
            {
                "chunk_id": "c2",
                "file_path": "math_utils.py",
                "language": "python",
                "start_line": 1,
                "end_line": 15,
                "symbol_name": "compute_average",
                "content": "def compute_average(numbers): return sum(numbers) / len(numbers)",
            },
        ]

        self.manager.upsert_chunks(repo_id, chunks)

        assert self.manager.get_collection_dimension(col) == 384
        assert col.count() == 2

        # Query semantic similarity
        results = self.manager.query_similar_chunks(repo_id, "authenticate user credentials")
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c1"
        assert "login" in results[0]["content"]

    def test_safe_replace_collection_with_onnx(self):
        repo_id = str(uuid.uuid4())
        initial_chunks = [
            {
                "chunk_id": "init_1",
                "content": "initial version of code chunk",
                "file_path": "init.py",
                "language": "python",
                "start_line": 1,
                "end_line": 5,
            }
        ]
        self.manager.upsert_chunks(repo_id, initial_chunks)
        col = self.manager.get_or_create_collection(repo_id)
        assert col.count() == 1

        replacement_chunks = [
            {
                "chunk_id": "repl_1",
                "content": "replacement code chunk A",
                "file_path": "service.py",
                "language": "python",
                "start_line": 1,
                "end_line": 8,
            },
            {
                "chunk_id": "repl_2",
                "content": "replacement code chunk B",
                "file_path": "service.py",
                "language": "python",
                "start_line": 9,
                "end_line": 16,
            },
        ]

        success = self.manager.safe_replace_collection(repo_id, replacement_chunks)
        assert success is True

        updated_col = self.manager.get_or_create_collection(repo_id)
        assert updated_col.count() == 2
        assert self.manager.get_collection_dimension(updated_col) == 384
