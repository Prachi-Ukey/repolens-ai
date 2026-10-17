import os
import sys
import tempfile
import shutil
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure backend is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.database import SessionLocal
from app.models.user import User
from app.models.repository import Repository
from app.models.job import AnalysisJob
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.utils.security import get_password_hash, create_access_token
from app.github.parser import parse_file_into_chunks, process_repository_files, detect_language
from app.github.client import validate_github_url, parse_github_url, normalize_github_url
from app.ai.embeddings import MockEmbedding, LocalEmbedding, get_embedding_provider
from app.ai.vector_store import get_vector_store
from app.services.ingestion_service import run_ingestion_pipeline, reindex_existing_repository
from app.services.repo_service import create_repository, get_repository_files, build_file_tree, generate_architecture_graph
from app.ai.rag import RAGEngine
from app.ai.llm import MockLLM


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def test_users(db_session):
    # Create or fetch two distinct users to verify ownership isolation
    u1 = db_session.query(User).filter(User.username == "test_owner_1").first()
    if not u1:
        u1 = User(
            id="test-owner-uuid-1",
            username="test_owner_1",
            email="owner1@repolens.test",
            hashed_password=get_password_hash("Password123!")
        )
        db_session.add(u1)
        db_session.commit()
        db_session.refresh(u1)

    u2 = db_session.query(User).filter(User.username == "test_owner_2").first()
    if not u2:
        u2 = User(
            id="test-owner-uuid-2",
            username="test_owner_2",
            email="owner2@repolens.test",
            hashed_password=get_password_hash("Password456!")
        )
        db_session.add(u2)
        db_session.commit()
        db_session.refresh(u2)

    token1 = create_access_token({"sub": u1.id, "username": u1.username})
    token2 = create_access_token({"sub": u2.id, "username": u2.username})

    return {
        "user1": u1,
        "token1": token1,
        "headers1": {"Authorization": f"Bearer {token1}"},
        "user2": u2,
        "token2": token2,
        "headers2": {"Authorization": f"Bearer {token2}"},
    }


# =====================================================================
# 1. Successful authentication and invalid credentials (JSON & Swagger Form)
# =====================================================================
def test_01_authentication_flows(client, test_users):
    # 1. JSON login success
    resp = client.post("/auth/login", json={
        "username_or_email": "test_owner_1",
        "password": "Password123!"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"].lower() == "bearer"

    # Email login success
    resp_email = client.post("/auth/login", json={
        "username_or_email": "owner1@repolens.test",
        "password": "Password123!"
    })
    assert resp_email.status_code == 200

    # 2. Swagger OAuth2 form /auth/token endpoint success
    resp_form = client.post("/auth/token", data={
        "username": "test_owner_1",
        "password": "Password123!"
    })
    assert resp_form.status_code == 200
    form_data = resp_form.json()
    assert "access_token" in form_data
    assert form_data["token_type"].lower() == "bearer"

    # 3. Invalid credentials
    resp_bad = client.post("/auth/login", json={
        "username_or_email": "test_owner_1",
        "password": "WrongPassword!"
    })
    assert resp_bad.status_code == 401

    resp_nonexistent = client.post("/auth/login", json={
        "username_or_email": "nonexistent_user",
        "password": "AnyPassword"
    })
    assert resp_nonexistent.status_code == 401


# =====================================================================
# 2. Protected endpoints reject missing or invalid tokens
# =====================================================================
def test_02_protected_endpoints_reject_unauthorized(client):
    # Missing token
    assert client.get("/repositories").status_code == 401
    assert client.post("/repositories", json={"url": "https://github.com/psf/requests"}).status_code == 401
    assert client.get("/auth/me").status_code == 401

    # Invalid / malformed token
    bad_headers = {"Authorization": "Bearer invalid.token.payload"}
    assert client.get("/repositories", headers=bad_headers).status_code == 401
    assert client.get("/repositories/fake-id/analysis-status", headers=bad_headers).status_code == 401


# =====================================================================
# 3. Repository creation and ownership enforcement
# =====================================================================
def test_03_repository_ownership_enforcement(client, test_users, db_session):
    headers1 = test_users["headers1"]
    headers2 = test_users["headers2"]
    u1 = test_users["user1"]

    # Create repository owned by User 1 directly in db for test isolation
    repo1 = Repository(
        id="test-isolated-repo-u1",
        user_id=u1.id,
        url="https://github.com/test-owner/isolated-repo",
        owner="test-owner",
        name="isolated-repo",
        primary_language="Python",
        file_count=1,
        chunk_count=2,
        status="completed"
    )
    job1 = AnalysisJob(
        id="test-job-isolated-1",
        repository_id=repo1.id,
        status="completed",
        current_stage="completed",
        progress_percent=100,
        message="Analysis complete!"
    )
    db_session.merge(repo1)
    db_session.merge(job1)
    db_session.commit()

    # User 1 can access
    resp1 = client.get(f"/repositories/{repo1.id}", headers=headers1)
    assert resp1.status_code == 200

    status_resp1 = client.get(f"/repositories/{repo1.id}/analysis-status", headers=headers1)
    assert status_resp1.status_code == 200

    # User 2 CANNOT access User 1's repository or status (must return 404)
    resp2 = client.get(f"/repositories/{repo1.id}", headers=headers2)
    assert resp2.status_code == 404

    status_resp2 = client.get(f"/repositories/{repo1.id}/analysis-status", headers=headers2)
    assert status_resp2.status_code == 404

    files_resp2 = client.get(f"/repositories/{repo1.id}/files", headers=headers2)
    assert files_resp2.status_code == 404

    arch_resp2 = client.get(f"/repositories/{repo1.id}/architecture", headers=headers2)
    assert arch_resp2.status_code == 404


# =====================================================================
# 4. Successful analysis of a small local test fixture
# =====================================================================
def test_04_successful_analysis_local_fixture(db_session, test_users):
    u1 = test_users["user1"]
    temp_dir = tempfile.mkdtemp(prefix="repolens_fixture_")
    try:
        # Create small test Python repository fixture
        os.makedirs(os.path.join(temp_dir, "src"), exist_ok=True)
        with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write("# Sample Mini Repo\nThis repository provides utility functions for data processing.\n")
        with open(os.path.join(temp_dir, "src", "calculator.py"), "w", encoding="utf-8") as f:
            f.write("def add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n    return a + b\n\ndef multiply(a: int, b: int) -> int:\n    return a * b\n")

        repo = Repository(
            id="fixture-mini-repo-id",
            user_id=u1.id,
            url="https://github.com/mock-org/mini-repo",
            owner="mock-org",
            name="mini-repo",
            primary_language="Python",
            status="analyzing"
        )
        job = AnalysisJob(
            id="fixture-mini-job-id",
            repository_id=repo.id,
            status="pending",
            current_stage="validating",
            progress_percent=0,
            message="Queued for analysis..."
        )
        db_session.merge(repo)
        db_session.merge(job)
        db_session.commit()

        with patch("app.services.ingestion_service.download_or_clone_repository", return_value=temp_dir):
            with patch("app.services.ingestion_service.cleanup_repository_dir"):
                run_ingestion_pipeline(job.id, repo.id)

        # Verify database records
        db_session.expire_all()
        saved_repo = db_session.query(Repository).filter(Repository.id == repo.id).first()
        saved_job = db_session.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()

        assert saved_job.status == "completed"
        assert saved_job.progress_percent == 100
        assert saved_repo.status == "completed"
        assert saved_repo.file_count >= 2
        assert saved_repo.chunk_count >= 2
        assert saved_repo.embedding_dimension == 384

        # Verify persisted files & chunks
        files = db_session.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).all()
        chunks = db_session.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).all()
        assert len(files) == saved_repo.file_count
        assert len(chunks) == saved_repo.chunk_count
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 5. GitHub download or clone failure handling
# =====================================================================
def test_05_clone_failure_handling(db_session, test_users):
    u1 = test_users["user1"]
    repo = Repository(
        id="clone-fail-repo-id",
        user_id=u1.id,
        url="https://github.com/nonexistent/failing-repo",
        owner="nonexistent",
        name="failing-repo",
        status="analyzing"
    )
    job = AnalysisJob(
        id="clone-fail-job-id",
        repository_id=repo.id,
        status="pending",
        current_stage="validating",
        progress_percent=0
    )
    db_session.merge(repo)
    db_session.merge(job)
    db_session.commit()

    with patch("app.services.ingestion_service.download_or_clone_repository", side_effect=RuntimeError("Could not clone repository: Remote branch not found")):
        run_ingestion_pipeline(job.id, repo.id)

    db_session.expire_all()
    saved_repo = db_session.query(Repository).filter(Repository.id == repo.id).first()
    saved_job = db_session.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()

    assert saved_job.status == "failed"
    assert "Could not clone repository" in (saved_job.error_details or saved_job.message)
    assert saved_repo.status == "failed"


# =====================================================================
# 6. Empty repository or no supported source files
# =====================================================================
def test_06_empty_repository_handling(db_session, test_users):
    u1 = test_users["user1"]
    temp_dir = tempfile.mkdtemp(prefix="repolens_empty_")
    try:
        # Create repo with only ignored binaries
        with open(os.path.join(temp_dir, "archive.bin"), "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        repo = Repository(
            id="empty-repo-fixture-id",
            user_id=u1.id,
            url="https://github.com/mock-org/empty-repo",
            owner="mock-org",
            name="empty-repo",
            status="analyzing"
        )
        job = AnalysisJob(
            id="empty-job-fixture-id",
            repository_id=repo.id,
            status="pending",
            current_stage="validating",
            progress_percent=0
        )
        db_session.merge(repo)
        db_session.merge(job)
        db_session.commit()

        with patch("app.services.ingestion_service.download_or_clone_repository", return_value=temp_dir):
            with patch("app.services.ingestion_service.cleanup_repository_dir"):
                run_ingestion_pipeline(job.id, repo.id)

        db_session.expire_all()
        saved_repo = db_session.query(Repository).filter(Repository.id == repo.id).first()
        saved_job = db_session.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()

        assert saved_job.status == "completed"
        assert saved_repo.status == "completed"
        assert saved_repo.chunk_count == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 7. Malformed source files gracefully handled
# =====================================================================
def test_07_malformed_source_file_fallback():
    malformed_code = "def broken_syntax(:\n    return incomplete\n    def inside(:\n"
    chunks = parse_file_into_chunks("src/broken.py", malformed_code, "python")
    # Must not raise an exception and should produce fallback sliding-window chunks
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert chunks[0]["file_path"] == "src/broken.py"
    assert chunks[0]["start_line"] >= 1


# =====================================================================
# 8. AST parsing and code chunk generation
# =====================================================================
def test_08_ast_parsing_and_metadata():
    valid_code = """import os
from math import sqrt

class MatrixHelper:
    \"\"\"Helper class for matrix operations.\"\"\"
    def __init__(self, size: int):
        self.size = size

    def compute_norm(self, values: list) -> float:
        return sqrt(sum(x * x for x in values))

def standalone_helper(x: int) -> int:
    return x * 2
"""
    chunks = parse_file_into_chunks("src/matrix.py", valid_code, "python")
    assert len(chunks) >= 2

    symbol_names = [c.get("symbol_name") for c in chunks if c.get("symbol_name")]
    assert any("MatrixHelper" in s for s in symbol_names)
    assert any("standalone_helper" in s for s in symbol_names)

    for c in chunks:
        assert c["file_path"] == "src/matrix.py"
        assert c["start_line"] <= c["end_line"]
        assert len(c["content"].strip()) > 0


# =====================================================================
# 9. Embedding generation and dimension validation
# =====================================================================
def test_09_embedding_generation_and_dimensions():
    provider = get_embedding_provider()
    assert provider.dimension == 384

    texts = [
        "def process_data(items): return [x.strip() for x in items]",
        "class RequestSession: def get(self, url): pass"
    ]
    embs = provider.embed_documents(texts)
    assert len(embs) == 2
    assert len(embs[0]) == 384
    assert len(embs[1]) == 384

    query_emb = provider.embed_query("How to make HTTP GET request?")
    assert len(query_emb) == 384


# =====================================================================
# 10. ChromaDB indexing and retrieval
# =====================================================================
def test_10_chromadb_indexing_and_retrieval():
    vector_store = get_vector_store()
    repo_id = "test-retrieval-isolation-repo"

    test_chunks = [
        {
            "chunk_id": "c1",
            "file_path": "auth/token.py",
            "content": "def create_jwt_token(payload: dict) -> str:\n    return jwt.encode(payload, secret_key)\n",
            "start_line": 1,
            "end_line": 2,
            "symbol_name": "create_jwt_token",
            "language": "python"
        },
        {
            "chunk_id": "c2",
            "file_path": "db/session.py",
            "content": "def get_database_connection():\n    return create_engine(DATABASE_URL)\n",
            "start_line": 1,
            "end_line": 2,
            "symbol_name": "get_database_connection",
            "language": "python"
        }
    ]

    success = vector_store.safe_replace_collection(repo_id, test_chunks)
    assert success is True

    col = vector_store.get_or_create_collection(repo_id)
    assert col.count() == 2

    results = vector_store.query_similar_chunks(repo_id, "JWT token authentication creation", top_k=2)
    assert len(results) > 0
    top_chunk = results[0]
    assert "token" in top_chunk["content"].lower() or top_chunk["file_path"] == "auth/token.py"

    # Clean up
    vector_store.delete_collection(repo_id)


# =====================================================================
# 11. Accurate file and chunk counts
# =====================================================================
def test_11_accurate_persisted_file_and_chunk_counts(db_session, test_users):
    u1 = test_users["user1"]
    repo = db_session.query(Repository).filter(Repository.id == "fixture-mini-repo-id").first()
    if repo:
        real_file_count = db_session.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).count()
        real_chunk_count = db_session.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).count()
        assert repo.file_count == real_file_count
        assert repo.chunk_count == real_chunk_count


# =====================================================================
# 12. Analysis status transitions and progress polling endpoint
# =====================================================================
def test_12_analysis_status_transitions(client, test_users, db_session):
    u1 = test_users["user1"]
    headers1 = test_users["headers1"]

    repo = Repository(
        id="status-poll-test-repo",
        user_id=u1.id,
        url="https://github.com/mock-org/status-poll-repo",
        owner="mock-org",
        name="status-poll-repo",
        status="analyzing"
    )
    job = AnalysisJob(
        id="status-poll-job-id",
        repository_id=repo.id,
        status="processing",
        current_stage="chunking",
        progress_percent=80,
        message="Created 50 code chunks..."
    )
    db_session.merge(repo)
    db_session.merge(job)
    db_session.commit()

    resp = client.get(f"/repositories/{repo.id}/analysis-status", headers=headers1)
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job.id
    assert data["status"] == "processing"
    assert data["current_stage"] == "chunking"
    assert data["progress_percent"] == 80
    assert "code chunks" in data["message"]


# =====================================================================
# 13. Database or vector-index failures handled safely
# =====================================================================
def test_13_vector_index_failure_captured_safely(db_session, test_users):
    u1 = test_users["user1"]
    temp_dir = tempfile.mkdtemp(prefix="repolens_vecfail_")
    try:
        with open(os.path.join(temp_dir, "test.py"), "w", encoding="utf-8") as f:
            f.write("def foo(): return 1\n")

        repo = Repository(
            id="vecfail-repo-id",
            user_id=u1.id,
            url="https://github.com/mock-org/vecfail-repo",
            owner="mock-org",
            name="vecfail-repo",
            status="analyzing"
        )
        job = AnalysisJob(
            id="vecfail-job-id",
            repository_id=repo.id,
            status="pending",
            current_stage="validating",
            progress_percent=0
        )
        db_session.merge(repo)
        db_session.merge(job)
        db_session.commit()

        # Mock safe_replace_collection to return False (simulating vector failure)
        mock_vs = MagicMock()
        mock_vs.safe_replace_collection.return_value = False

        with patch("app.services.ingestion_service.download_or_clone_repository", return_value=temp_dir):
            with patch("app.services.ingestion_service.cleanup_repository_dir"):
                with patch("app.services.ingestion_service.get_vector_store", return_value=mock_vs):
                    run_ingestion_pipeline(job.id, repo.id)

        db_session.expire_all()
        saved_repo = db_session.query(Repository).filter(Repository.id == repo.id).first()
        saved_job = db_session.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()

        assert saved_job.status == "failed"
        assert "Failed to generate embeddings" in (saved_job.error_details or "")
        assert saved_repo.status == "failed"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 14. Successful repository overview and architecture retrieval
# =====================================================================
def test_14_repository_overview_and_architecture(client, test_users, db_session):
    u1 = test_users["user1"]
    headers1 = test_users["headers1"]

    repo = db_session.query(Repository).filter(Repository.id == "fixture-mini-repo-id").first()
    if not repo:
        repo = Repository(
            id="fixture-mini-repo-id",
            user_id=u1.id,
            url="https://github.com/mock-org/mini-repo",
            owner="mock-org",
            name="mini-repo",
            primary_language="Python",
            status="completed",
            summary_json=json.dumps({"overview": "Sample Mini Repo", "readme_preview": "# Sample Mini Repo\nThis repository provides utility functions."}),
            architecture_json=json.dumps({"nodes": [{"id": "n1", "label": "Calculator"}], "edges": []})
        )
        db_session.merge(repo)
        db_session.commit()

    overview_resp = client.get(f"/repositories/{repo.id}/overview", headers=headers1)
    assert overview_resp.status_code == 200
    ov_data = overview_resp.json()
    assert ov_data["id"] == repo.id
    assert ov_data["name"] == "mini-repo"

    arch_resp = client.get(f"/repositories/{repo.id}/architecture", headers=headers1)
    assert arch_resp.status_code == 200
    arch_data = arch_resp.json()
    assert "nodes" in arch_data
    assert "edges" in arch_data


# =====================================================================
# 15. RAG answers containing source references
# =====================================================================
def test_15_rag_answers_with_source_citations():
    vector_store = get_vector_store()
    mock_llm = MockLLM()
    engine = RAGEngine(vector_store=vector_store, llm_provider=mock_llm)

    test_chunks = [
        {
            "chunk_id": "rag_c1",
            "file_path": "requests/sessions.py",
            "content": "class Session:\n    def request(self, method, url):\n        return self.send(PreparedRequest())\n",
            "start_line": 10,
            "end_line": 13,
            "symbol_name": "Session.request",
            "language": "python"
        }
    ]
    repo_id = "test-rag-citations-repo"
    vector_store.safe_replace_collection(repo_id, test_chunks)

    try:
        res = engine.ask_question(repo_id, "How does Session.request send requests?", top_k=1)
        assert "answer" in res
        sources = res.get("sources", [])
        assert len(sources) > 0
        s0 = sources[0]
        assert s0["file_path"] == "requests/sessions.py"
        assert s0["start_line"] == 10
        assert s0["end_line"] == 13
    finally:
        vector_store.delete_collection(repo_id)


# =====================================================================
# 16. Reanalysis without duplicate chunks or vectors
# =====================================================================
def test_16_reanalysis_idempotency_no_duplicates(db_session, test_users):
    u1 = test_users["user1"]
    temp_dir = tempfile.mkdtemp(prefix="repolens_reanalysis_")
    uid = uuid.uuid4().hex[:8]
    repo_id = f"reanalysis-{uid}"
    job1_id = f"reanalysis-job1-{uid}"
    job2_id = f"reanalysis-job2-{uid}"
    try:
        with open(os.path.join(temp_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write("def run():\n    return 'running'\n")

        repo = Repository(
            id=repo_id,
            user_id=u1.id,
            url="https://github.com/mock-org/reanalysis-repo",
            owner="mock-org",
            name="reanalysis-repo",
            status="analyzing"
        )
        job1 = AnalysisJob(
            id=job1_id,
            repository_id=repo.id,
            status="pending",
            current_stage="validating",
            progress_percent=0
        )
        db_session.add(repo)
        db_session.add(job1)
        db_session.commit()

        # Run 1
        with patch("app.services.ingestion_service.download_or_clone_repository", return_value=temp_dir):
            with patch("app.services.ingestion_service.cleanup_repository_dir"):
                run_ingestion_pipeline(job1.id, repo.id)

        count_files_1 = db_session.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).count()
        count_chunks_1 = db_session.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).count()
        assert count_files_1 == 1
        assert count_chunks_1 == 1

        # Run 2 (Reanalysis of same repository)
        job2 = AnalysisJob(
            id=job2_id,
            repository_id=repo.id,
            status="pending",
            current_stage="validating",
            progress_percent=0
        )
        db_session.add(job2)
        db_session.commit()

        with patch("app.services.ingestion_service.download_or_clone_repository", return_value=temp_dir):
            with patch("app.services.ingestion_service.cleanup_repository_dir"):
                run_ingestion_pipeline(job2.id, repo.id)

        count_files_2 = db_session.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).count()
        count_chunks_2 = db_session.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).count()

        # Counts must NOT have doubled
        assert count_files_2 == 1
        assert count_chunks_2 == 1

        vector_store = get_vector_store()
        col = vector_store.get_or_create_collection(repo.id)
        assert col.count() == 1
        vector_store.delete_collection(repo.id)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 17. URL normalization and edge case parsing
# =====================================================================
def test_17_github_url_normalization():
    assert validate_github_url("https://github.com/psf/requests") is True
    assert validate_github_url("https://github.com/psf/requests/") is True
    assert validate_github_url("https://github.com/psf/requests.git") is True
    assert validate_github_url("https://github.com/psf/requests/tree/main") is True
    assert normalize_github_url("https://github.com/psf/requests/") == "https://github.com/psf/requests"
    assert normalize_github_url("https://github.com/psf/requests.git") == "https://github.com/psf/requests"


# =====================================================================
# 18. Submitting already completed repository reuses existing analysis
# =====================================================================
def test_18_submitting_completed_repo_reuses_without_reanalysis(client, test_users, db_session):
    u1 = test_users["user1"]
    headers1 = test_users["headers1"]

    uid = uuid.uuid4().hex[:8]
    completed_repo = Repository(
        id=f"completed-reuse-{uid}",
        user_id=u1.id,
        url=f"https://github.com/mock-org/completed-repo-{uid}",
        owner="mock-org",
        name=f"completed-repo-{uid}",
        primary_language="Python",
        file_count=20,
        chunk_count=97,
        status="completed"
    )
    completed_job = AnalysisJob(
        id=f"completed-job-{uid}",
        repository_id=completed_repo.id,
        status="completed",
        current_stage="completed",
        progress_percent=100,
        message="Analysis complete!"
    )
    db_session.add(completed_repo)
    db_session.add(completed_job)
    db_session.commit()

    # Submit the exact same repository URL WITHOUT reanalyze flag
    with patch("app.services.repo_service.get_github_repo_metadata", return_value={
        "owner": "mock-org",
        "name": f"completed-repo-{uid}",
        "default_branch": "main",
        "primary_language": "Python",
        "description": "Mock completed repo"
    }):
        resp = client.post("/repositories", json={"url": completed_repo.url}, headers=headers1)

    assert resp.status_code == 201
    data = resp.json()

    # Must reuse existing repository without creating duplicate or re-running pipeline
    assert data["is_already_completed"] is True
    assert data["repository"]["id"] == completed_repo.id
    assert data["repository"]["status"] == "completed"
    assert data["repository"]["file_count"] == 20
    assert data["repository"]["chunk_count"] == 97

    # Verify analysis-status returns 100% completed
    status_resp = client.get(f"/repositories/{completed_repo.id}/analysis-status", headers=headers1)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "completed"
    assert status_data["progress_percent"] == 100


# =====================================================================
# 19. Explicit reanalysis flag triggers re-ingestion job
# =====================================================================
def test_19_explicit_reanalysis_forces_pipeline(client, test_users, db_session):
    u1 = test_users["user1"]
    headers1 = test_users["headers1"]

    uid = uuid.uuid4().hex[:8]
    completed_repo = Repository(
        id=f"force-reanalysis-{uid}",
        user_id=u1.id,
        url=f"https://github.com/mock-org/force-repo-{uid}",
        owner="mock-org",
        name=f"force-repo-{uid}",
        primary_language="Python",
        file_count=5,
        chunk_count=10,
        status="completed"
    )
    job0 = AnalysisJob(
        id=f"job0-{uid}",
        repository_id=completed_repo.id,
        status="completed",
        current_stage="completed",
        progress_percent=100
    )
    db_session.add(completed_repo)
    db_session.add(job0)
    db_session.commit()

    with patch("app.services.repo_service.get_github_repo_metadata", return_value={
        "owner": "mock-org",
        "name": f"force-repo-{uid}",
        "default_branch": "main",
        "primary_language": "Python",
        "description": "Mock force repo"
    }):
        # Submit with reanalyze: True
        resp = client.post("/repositories", json={"url": completed_repo.url, "reanalyze": True}, headers=headers1)

    assert resp.status_code == 201
    data = resp.json()
    assert data["is_already_completed"] is False
    assert data["status"] == "analyzing"
    assert data["job_id"] is not None
    assert data["job_id"] != job0.id
