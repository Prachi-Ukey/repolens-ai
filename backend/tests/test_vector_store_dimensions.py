import os
import shutil
import tempfile
import uuid

import pytest
import chromadb
from chromadb.errors import InvalidArgumentError
from unittest.mock import patch, MagicMock

from app.database import SessionLocal, Base, engine
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.ai.embeddings import (
    MockEmbedding,
    LocalEmbedding,
    GeminiEmbedding,
    EmbeddingProvider,
    get_embedding_provider,
)
from app.ai.vector_store import ChromaManager


class TestEmbeddingConsistency:
    def test_mock_embedding_document_and_query_consistency(self):
        provider = MockEmbedding()
        assert provider.dimension == 384
        assert "384" in provider.model_name

        docs = [
            "function authenticate() { return true; }",
            "import os, sys",
        ]
        doc_embeddings = provider.embed_documents(docs)
        query_embedding = provider.embed_query("authenticate")

        assert len(doc_embeddings) == 2
        assert len(doc_embeddings[0]) == 384
        assert len(doc_embeddings[1]) == 384
        assert len(query_embedding) == 384
        assert len(doc_embeddings[0]) == len(query_embedding)

    def test_custom_dimension_mock_embedding(self):
        provider_128 = MockEmbedding(dim=128)
        assert provider_128.dimension == 128

        doc_vecs = provider_128.embed_documents(["test document"])
        query_vec = provider_128.embed_query("test query")

        assert len(doc_vecs[0]) == 128
        assert len(query_vec) == 128

        provider_768 = MockEmbedding(dim=768)
        assert provider_768.dimension == 768
        assert len(provider_768.embed_query("test")) == 768

    def test_local_embedding_dimension(self):
        try:
            local = LocalEmbedding()
            assert local.dimension == 384
            assert local.model_name == "all-MiniLM-L6-v2"

            doc_emb = local.embed_documents(["hello test"])
            q_emb = local.embed_query("hello")

            assert len(doc_emb[0]) == 384
            assert len(q_emb) == 384
        except RuntimeError:
            pytest.skip("sentence-transformers not installed in environment")

    def test_gemini_embedding_model_name_and_dimension(self):
        gemini = GeminiEmbedding(api_key="test-api-key-12345")
        assert gemini.model_name == "gemini-embedding-001"
        assert gemini.dimension == 768

    def test_gemini_embedding_explicit_requested_dimension_and_payload(self):
        gemini = GeminiEmbedding(
            api_key="test-api-key-12345",
            dimension=768,
        )
        captured_requests = []

        def mock_post(url, json=None, **kwargs):
            captured_requests.append({
                "url": url,
                "json": json,
                "headers": kwargs.get("headers", {}),
            })

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": [
                    {"values": [0.01] * 768},
                    {"values": [0.02] * 768},
                ]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            docs = [
                "function calculateMetrics() { return 42; }",
                "class RepositoryAnalysis: pass",
            ]
            embeddings = gemini.embed_documents(docs)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768
        assert len(embeddings[1]) == 768

        assert len(captured_requests) == 1
        req = captured_requests[0]

        # API key must be in the header, not in the URL.
        assert (
            "models/gemini-embedding-001:batchEmbedContents"
            in req["url"]
        )
        assert "key=" not in req["url"]
        assert (
            req["headers"]["x-goog-api-key"]
            == "test-api-key-12345"
        )

        req_items = req["json"]["requests"]
        assert len(req_items) == 2

        for item in req_items:
            assert item["model"] == "models/gemini-embedding-001"
            assert item["outputDimensionality"] == 768
            assert item["taskType"] == "RETRIEVAL_DOCUMENT"

    def test_gemini_embedding_query_consistency(self):
        gemini = GeminiEmbedding(
            api_key="test-api-key-12345",
            dimension=768,
        )
        captured_requests = []

        def mock_post(url, json=None, **kwargs):
            captured_requests.append({
                "url": url,
                "json": json,
                "headers": kwargs.get("headers", {}),
            })

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": [{"values": [0.05] * 768}]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            query_vec = gemini.embed_query(
                "calculateMetrics execution flow"
            )

        assert len(query_vec) == 768
        assert len(captured_requests) == 1
        assert (
            captured_requests[0]["json"]["requests"][0]
            ["outputDimensionality"]
            == 768
        )
        assert (
            captured_requests[0]["headers"]["x-goog-api-key"]
            == "test-api-key-12345"
        )
        assert "key=" not in captured_requests[0]["url"]

    def test_gemini_embedding_dimension_verification_failure(self):
        gemini = GeminiEmbedding(
            api_key="test-api-key-12345",
            dimension=768,
        )

        def mock_post_mismatch(url, json=None, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": [{"values": [0.01] * 384}]
            }
            return mock_resp

        with patch(
            "httpx.Client.post",
            side_effect=mock_post_mismatch,
        ):
            with pytest.raises(
                ValueError,
                match="Gemini embedding dimension mismatch",
            ):
                gemini.embed_documents(
                    ["test text with wrong returned dimensions"]
                )


class TestVectorStoreDimensionHandling:
    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self):
        self.temp_chroma_dir = tempfile.mkdtemp(
            prefix="repolens_chroma_test_"
        )
        yield
        if os.path.exists(self.temp_chroma_dir):
            shutil.rmtree(self.temp_chroma_dir, ignore_errors=True)

    def test_collection_dimension_detection(self):
        manager = ChromaManager(persist_dir=self.temp_chroma_dir)
        manager.embedding_provider = MockEmbedding(dim=384)
        col_384 = manager.get_or_create_collection("repo_test_384")
        assert manager.get_collection_dimension(col_384) == 384

        client = chromadb.PersistentClient(path=self.temp_chroma_dir)
        raw_col = client.get_or_create_collection(
            name="repo_legacy_128"
        )
        raw_col.upsert(
            ids=["chunk_1"],
            embeddings=[[0.1] * 128],
            documents=["legacy code chunk"],
            metadatas=[{
                "file_path": "legacy.py",
                "language": "python",
                "start_line": 1,
                "end_line": 10,
                "chunk_id": "chunk_1",
            }],
        )
        assert manager.get_collection_dimension(raw_col) == 128

        empty_col = client.get_or_create_collection(name="repo_empty")
        assert manager.get_collection_dimension(empty_col) is None

    def test_upsert_recreates_incompatible_collection(self):
        manager = ChromaManager(persist_dir=self.temp_chroma_dir)
        repo_id = str(uuid.uuid4())

        manager.embedding_provider = MockEmbedding(dim=128)
        initial_chunks = [{
            "chunk_id": "c1",
            "file_path": "app.py",
            "language": "python",
            "start_line": 1,
            "end_line": 10,
            "symbol_name": "main",
            "content": "def main(): pass",
        }]
        manager.upsert_chunks(repo_id, initial_chunks)

        col = manager.get_or_create_collection(repo_id)
        assert manager.get_collection_dimension(col) == 128

        manager.embedding_provider = MockEmbedding(dim=384)
        fresh_chunks = [{
            "chunk_id": "c2",
            "file_path": "server.py",
            "language": "python",
            "start_line": 1,
            "end_line": 15,
            "symbol_name": "start_server",
            "content": "def start_server(): return True",
        }]
        manager.upsert_chunks(repo_id, fresh_chunks)

        col_updated = manager.get_or_create_collection(repo_id)
        assert manager.get_collection_dimension(col_updated) == 384

        results = manager.query_similar_chunks(repo_id, "start server")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    def test_safe_auto_rebuild_from_db_on_dimension_mismatch(self):
        """Verify rebuilding incompatible vectors without losing DB records."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())

        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"dim_test_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="hashed_pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/testorg/dimtest",
                owner="testorg",
                name="dimtest",
                status="completed",
                file_count=1,
                chunk_count=1,
            )
            db.add(repo)
            db.commit()

            file_id = str(uuid.uuid4())
            repo_file = RepositoryFile(
                id=file_id,
                repository_id=repo.id,
                file_path="src/router.py",
                file_name="router.py",
                language="python",
                size_bytes=120,
                line_count=10,
                sha_hash="filehash",
                content="def handle_route():\n    return 'OK'\n",
            )
            db.add(repo_file)
            db.commit()

            chunk = CodeChunk(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_id=repo_file.id,
                file_path="src/router.py",
                language="python",
                start_line=1,
                end_line=10,
                symbol_name="handle_route",
                chunk_id="chunk_router_1",
                content="def handle_route():\n    return 'OK'\n",
                sha_hash="chunkhash",
            )
            db.add(chunk)
            db.commit()

            manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            manager.embedding_provider = MockEmbedding(dim=128)
            col = manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_router_1"],
                embeddings=[[0.05] * 128],
                documents=["def handle_route():\n    return 'OK'\n"],
                metadatas=[{
                    "file_path": "src/router.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 10,
                    "symbol_name": "handle_route",
                    "chunk_id": "chunk_router_1",
                }],
            )
            assert manager.get_collection_dimension(col) == 128

            active_manager = ChromaManager(
                persist_dir=self.temp_chroma_dir
            )
            active_manager.embedding_provider = MockEmbedding(dim=384)

            results = active_manager.query_similar_chunks(
                repo_id, "handle route"
            )
            assert len(results) > 0
            assert "handle_route" in results[0]["content"]

            rebuilt_col = active_manager.get_or_create_collection(repo_id)
            assert (
                active_manager.get_collection_dimension(rebuilt_col)
                == 384
            )

            assert db.query(User).filter(User.id == user.id).first() is not None
            repo_check = (
                db.query(Repository)
                .filter(Repository.id == repo.id)
                .first()
            )
            assert repo_check is not None
            assert repo_check.status == "completed"

        finally:
            db.query(CodeChunk).filter(
                CodeChunk.repository_id == repo_id
            ).delete()
            db.query(RepositoryFile).filter(
                RepositoryFile.repository_id == repo_id
            ).delete()
            db.query(Repository).filter(
                Repository.id == repo_id
            ).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()

    def test_rebuild_with_no_stored_files_or_chunks_retains_existing_collection(
        self,
    ):
        """An empty DB must not erase an existing Chroma collection."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())

        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"nofiles_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="hashed_pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/testorg/emptytest",
                owner="testorg",
                name="emptytest",
                status="completed",
                file_count=0,
                chunk_count=0,
            )
            db.add(repo)
            db.commit()

            manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            manager.embedding_provider = MockEmbedding(dim=384)
            col = manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_pre_1", "chunk_pre_2"],
                embeddings=[[0.05] * 384, [0.06] * 384],
                documents=[
                    "def seed_one(): pass",
                    "def seed_two(): pass",
                ],
                metadatas=[
                    {
                        "file_path": "seed.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "chunk_id": "chunk_pre_1",
                    },
                    {
                        "file_path": "seed.py",
                        "language": "python",
                        "start_line": 3,
                        "end_line": 4,
                        "chunk_id": "chunk_pre_2",
                    },
                ],
            )
            assert col.count() == 2

            rebuilt = manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is False

            col_after = manager.get_or_create_collection(repo_id)
            assert col_after.count() == 2
            stored_docs = col_after.get()["documents"]
            assert "def seed_one(): pass" in stored_docs
            assert "def seed_two(): pass" in stored_docs
            assert (
                db.query(Repository)
                .filter(Repository.id == repo_id)
                .first()
                is not None
            )

        finally:
            db.query(CodeChunk).filter(
                CodeChunk.repository_id == repo_id
            ).delete()
            db.query(RepositoryFile).filter(
                RepositoryFile.repository_id == repo_id
            ).delete()
            db.query(Repository).filter(
                Repository.id == repo_id
            ).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()

    def test_rebuild_with_file_parsing_producing_zero_chunks_retains_existing_collection(
        self,
    ):
        """Files producing no chunks must not erase existing vectors."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())

        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"zerochunk_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="hashed_pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/testorg/zeroparse",
                owner="testorg",
                name="zeroparse",
                status="completed",
                file_count=2,
                chunk_count=0,
            )
            db.add(repo)
            db.commit()

            file1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/empty.py",
                file_name="empty.py",
                language="python",
                size_bytes=0,
                line_count=0,
                sha_hash="emptyhash",
                content="",
            )
            file2 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/spaces.py",
                file_name="spaces.py",
                language="python",
                size_bytes=10,
                line_count=2,
                sha_hash="spacehash",
                content="   \n\n\t  ",
            )
            db.add_all([file1, file2])
            db.commit()

            manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            manager.embedding_provider = MockEmbedding(dim=384)
            col = manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_keep_1", "chunk_keep_2"],
                embeddings=[[0.08] * 384, [0.09] * 384],
                documents=[
                    "def keep_one(): pass",
                    "def keep_two(): pass",
                ],
                metadatas=[
                    {
                        "file_path": "keep.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "chunk_id": "chunk_keep_1",
                    },
                    {
                        "file_path": "keep.py",
                        "language": "python",
                        "start_line": 3,
                        "end_line": 4,
                        "chunk_id": "chunk_keep_2",
                    },
                ],
            )
            assert col.count() == 2

            rebuilt = manager.rebuild_collection_from_db(repo_id)
            assert rebuilt is False

            col_after = manager.get_or_create_collection(repo_id)
            assert col_after.count() == 2
            stored_docs = col_after.get()["documents"]
            assert "def keep_one(): pass" in stored_docs
            assert "def keep_two(): pass" in stored_docs

        finally:
            db.query(CodeChunk).filter(
                CodeChunk.repository_id == repo_id
            ).delete()
            db.query(RepositoryFile).filter(
                RepositoryFile.repository_id == repo_id
            ).delete()
            db.query(Repository).filter(
                Repository.id == repo_id
            ).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()

    def test_safe_replacement_preserves_collection_when_embedding_fails(self):
        """Embedding failure must not erase the existing collection."""
        db = SessionLocal()
        repo_id = str(uuid.uuid4())

        try:
            user = User(
                id=str(uuid.uuid4()),
                email=f"failtest_{uuid.uuid4().hex[:6]}@example.com",
                username=f"user_{uuid.uuid4().hex[:6]}",
                hashed_password="hashed_pw",
            )
            db.add(user)
            db.commit()

            repo = Repository(
                id=repo_id,
                user_id=user.id,
                url="https://github.com/testorg/failtest",
                owner="testorg",
                name="failtest",
                status="completed",
                file_count=1,
                chunk_count=1,
            )
            db.add(repo)
            db.commit()

            file1 = RepositoryFile(
                id=str(uuid.uuid4()),
                repository_id=repo.id,
                file_path="src/service.py",
                file_name="service.py",
                language="python",
                size_bytes=50,
                line_count=5,
                sha_hash="servicehash",
                content="def execute():\n    return 42\n",
            )
            db.add(file1)
            db.commit()

            manager = ChromaManager(persist_dir=self.temp_chroma_dir)
            manager.embedding_provider = MockEmbedding(dim=384)
            col = manager.get_or_create_collection(repo_id)
            col.upsert(
                ids=["chunk_orig_1"],
                embeddings=[[0.01] * 384],
                documents=["def original(): pass"],
                metadatas=[{
                    "file_path": "orig.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 2,
                    "chunk_id": "chunk_orig_1",
                }],
            )
            assert col.count() == 1

            with patch.object(
                manager.embedding_provider,
                "embed_documents",
                side_effect=RuntimeError("API failure"),
            ):
                rebuilt = manager.rebuild_collection_from_db(repo_id)
                assert rebuilt is False

            col_after = manager.get_or_create_collection(repo_id)
            assert col_after.count() == 1
            assert col_after.get()["documents"] == [
                "def original(): pass"
            ]

        finally:
            db.query(CodeChunk).filter(
                CodeChunk.repository_id == repo_id
            ).delete()
            db.query(RepositoryFile).filter(
                RepositoryFile.repository_id == repo_id
            ).delete()
            db.query(Repository).filter(
                Repository.id == repo_id
            ).delete()
            if "user" in locals():
                db.query(User).filter(User.id == user.id).delete()
            db.commit()
            db.close()
