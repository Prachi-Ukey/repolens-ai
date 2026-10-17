import pytest
import os
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.models.job import AnalysisJob
from app.github.parser import process_repository_files, is_ignored_path

# Use an isolated test database with StaticPool for reliability tests
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    prior_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db
    try:
        yield
    finally:
        if prior_override is not None:
            app.dependency_overrides[get_db] = prior_override
        else:
            app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

client = TestClient(app)


def test_successful_registration_and_login():
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"developer_{unique_suffix}@repolens.ai"
    username = f"dev_{unique_suffix}"
    password = "StrongPassword123!"

    # 1. Register new account
    reg_res = client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": password
    })
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["email"] == email
    assert reg_data["username"] == username
    assert "id" in reg_data
    assert "password" not in reg_data
    assert "hashed_password" not in reg_data

    # 2. Login with username
    login_user_res = client.post("/auth/login", json={
        "username_or_email": username,
        "password": password
    })
    assert login_user_res.status_code == 200
    token_data = login_user_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    access_token = token_data["access_token"]

    # 3. Login with email
    login_email_res = client.post("/auth/login", json={
        "username_or_email": email,
        "password": password
    })
    assert login_email_res.status_code == 200
    assert "access_token" in login_email_res.json()

    # 4. Access authenticated /auth/me route
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == email
    assert me_data["username"] == username


def test_invalid_credentials_and_duplicate_registration():
    unique_suffix = uuid.uuid4().hex[:8]
    email = f"dup_{unique_suffix}@repolens.ai"
    username = f"dup_{unique_suffix}"
    password = "CorrectPassword123!"

    # Register initial user
    client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": password
    })

    # Test login with wrong password
    wrong_pass_res = client.post("/auth/login", json={
        "username_or_email": username,
        "password": "WrongPassword999!"
    })
    assert wrong_pass_res.status_code == 401
    assert "Incorrect username/email or password" in wrong_pass_res.json()["detail"]

    # Test login with non-existent user
    non_existent_res = client.post("/auth/login", json={
        "username_or_email": "non_existent_user_999",
        "password": password
    })
    assert non_existent_res.status_code == 401
    assert "Incorrect username/email or password" in non_existent_res.json()["detail"]

    # Test duplicate registration with same email
    dup_email_res = client.post("/auth/register", json={
        "email": email,
        "username": f"other_{unique_suffix}",
        "password": password
    })
    assert dup_email_res.status_code == 400
    assert "Email is already registered" in dup_email_res.json()["detail"]

    # Test duplicate registration with same username
    dup_user_res = client.post("/auth/register", json={
        "email": f"different_{unique_suffix}@repolens.ai",
        "username": username,
        "password": password
    })
    assert dup_user_res.status_code == 400
    assert "Username is already taken" in dup_user_res.json()["detail"]


def test_repository_authorization_isolation():
    # Create User A
    user_a_email = f"usera_{uuid.uuid4().hex[:6]}@repolens.ai"
    user_a_name = f"usera_{uuid.uuid4().hex[:6]}"
    res_a = client.post("/auth/register", json={"email": user_a_email, "username": user_a_name, "password": "PasswordA123!"})
    user_a_id = res_a.json()["id"]
    token_a = client.post("/auth/login", json={"username_or_email": user_a_name, "password": "PasswordA123!"}).json()["access_token"]

    # Create User B
    user_b_email = f"userb_{uuid.uuid4().hex[:6]}@repolens.ai"
    user_b_name = f"userb_{uuid.uuid4().hex[:6]}"
    res_b = client.post("/auth/register", json={"email": user_b_email, "username": user_b_name, "password": "PasswordB123!"})
    user_b_id = res_b.json()["id"]
    token_b = client.post("/auth/login", json={"username_or_email": user_b_name, "password": "PasswordB123!"}).json()["access_token"]

    # Directly insert a repo owned by User A in test DB
    repo_a_id = str(uuid.uuid4())
    db = TestingSessionLocal()
    repo_a = Repository(
        id=repo_a_id,
        user_id=user_a_id,
        url="https://github.com/owner/repo-a",
        owner="owner",
        name="repo-a",
        default_branch="main",
        primary_language="python",
        status="completed",
        file_count=5,
        chunk_count=20
    )
    db.add(repo_a)
    db.commit()
    db.close()

    # User A should be able to fetch their repository
    get_a_res = client.get(f"/repositories/{repo_a_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert get_a_res.status_code == 200
    assert get_a_res.json()["id"] == repo_a_id

    # User B should NOT be able to access User A's repository
    get_b_res = client.get(f"/repositories/{repo_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert get_b_res.status_code in (404, 403)


def test_ignored_paths_filtering():
    assert is_ignored_path("node_modules/react/index.js") is True
    assert is_ignored_path(".git/HEAD") is True
    assert is_ignored_path("venv/lib/site-packages/os.py") is True
    assert is_ignored_path("dist/bundle.js") is True
    assert is_ignored_path("build/output.css") is True
    assert is_ignored_path("package-lock.json") is True
    assert is_ignored_path("images/banner.png") is True
    assert is_ignored_path("weights.pt") is True
    assert is_ignored_path("src/components/Header.jsx") is False
    assert is_ignored_path("app/main.py") is False


def test_file_processing_limits(tmp_path):
    # Create a dummy repo directory with many small files
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    for i in range(15):
        (repo_dir / f"test_{i}.py").write_text(f"def func_{i}():\n    return {i}\n")

    # Override limit via environment variable
    old_max = os.environ.get("MAX_REPO_FILES")
    try:
        os.environ["MAX_REPO_FILES"] = "5"
        discovered, chunks = process_repository_files(str(repo_dir))
        assert len(discovered) == 5
        assert len(chunks) >= 5
    finally:
        if old_max is not None:
            os.environ["MAX_REPO_FILES"] = old_max
        else:
            os.environ.pop("MAX_REPO_FILES", None)


def test_cors_whitelisting_headers():
    response = client.options(
        "/auth/login",
        headers={
            "Origin": "https://repolens-ai-kappa.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://repolens-ai-kappa.vercel.app"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_local_embedding_batch_size_configuration():
    from app.ai.embeddings import LocalEmbedding

    # Default batch size is 16 for 512 MB memory footprint
    emb_default = LocalEmbedding()
    assert emb_default._batch_size == 16

    # Explicit batch size is respected
    emb_custom = LocalEmbedding(batch_size=8)
    assert emb_custom._batch_size == 8

    # Environment variable override is respected
    old_env = os.environ.get("EMBEDDING_BATCH_SIZE")
    try:
        os.environ["EMBEDDING_BATCH_SIZE"] = "24"
        emb_env = LocalEmbedding()
        assert emb_env._batch_size == 24
    finally:
        if old_env is not None:
            os.environ["EMBEDDING_BATCH_SIZE"] = old_env
        else:
            os.environ.pop("EMBEDDING_BATCH_SIZE", None)


def test_embedding_progress_callback_resilient_to_callback_failure():
    from app.ai.vector_store import get_vector_store

    vs = get_vector_store()
    repo_id = f"cb-fail-test-{uuid.uuid4().hex[:8]}"
    chunks = [
        {
            "chunk_id": f"chunk-{i}",
            "content": f"def test_func_{i}(): return {i}",
            "file_path": "test.py",
            "language": "python",
            "start_line": i * 5,
            "end_line": i * 5 + 4,
            "symbol_name": f"test_func_{i}",
        }
        for i in range(20)
    ]

    callback_invoked = []

    def failing_callback(processed: int, total: int):
        callback_invoked.append((processed, total))
        raise RuntimeError("Simulated transient callback failure (e.g. DB connection blip)")

    # Even though the callback raises an exception, the collection replacement must succeed cleanly
    success = vs.safe_replace_collection(repo_id, chunks, progress_callback=failing_callback)
    assert success is True
    assert len(callback_invoked) >= 1

    # Cleanup
    try:
        vs.delete_collection(repo_id)
    except Exception:
        pass


def test_failed_embedding_batch_cleans_up_staging_and_preserves_target():
    from unittest.mock import patch
    from app.ai.vector_store import get_vector_store

    vs = get_vector_store()
    repo_id = f"batch-fail-test-{uuid.uuid4().hex[:8]}"
    initial_chunks = [
        {"chunk_id": "c1", "content": "initial content", "file_path": "init.py", "language": "python", "start_line": 1, "end_line": 2}
    ]
    # Create initial collection
    vs.safe_replace_collection(repo_id, initial_chunks)
    col = vs.client.get_collection(vs._get_collection_name(repo_id))
    assert col.count() == 1

    replacement_chunks = [
        {"chunk_id": f"rep-{i}", "content": f"replacement code {i}", "file_path": "rep.py", "language": "python", "start_line": i, "end_line": i + 1}
        for i in range(10)
    ]

    # Mock embed_documents to fail during batch generation
    with patch.object(vs.embedding_provider, "embed_documents", side_effect=RuntimeError("Simulated batch embedding OOM/crash")):
        success = vs.safe_replace_collection(repo_id, replacement_chunks)
        assert success is False

    # Target collection must still be intact with its initial count
    col = vs.client.get_collection(vs._get_collection_name(repo_id))
    assert col.count() == 1

    # Ensure no lingering staging collections
    col_names = [c.name for c in vs.client.list_collections()]
    assert not any(name.startswith(f"staging_repo_{repo_id.replace('-', '_')}") for name in col_names)

    # Cleanup
    try:
        vs.delete_collection(repo_id)
    except Exception:
        pass


def test_active_job_exceeding_five_minutes_not_failed():
    from datetime import datetime, timezone, timedelta
    from app.routes.repository import get_analysis_status

    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    user = User(id=user_id, username=f"user_{uuid.uuid4().hex[:6]}", email=f"user_{uuid.uuid4().hex[:6]}@example.com", hashed_password="pw")
    repo = Repository(id=repo_id, user_id=user_id, url=f"https://github.com/test/{repo_id}", owner="test", name=repo_id, default_branch="main", primary_language="python", status="analyzing")

    # Heartbeat was updated recently (within 5 minutes)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    job = AnalysisJob(
        id=str(uuid.uuid4()),
        repository_id=repo_id,
        status="processing",
        current_stage="embedding",
        progress_percent=95,
        message="Generating embeddings...",
        created_at=now - timedelta(minutes=6),
        updated_at=now - timedelta(seconds=10),
    )
    db.add_all([user, repo, job])
    db.commit()

    resp = get_analysis_status(repo_id=repo_id, db=db, current_user=user)
    assert resp.status == "processing"
    assert resp.is_stalled is False
    assert resp.progress_percent == 95

    # Database must remain completely untouched
    db.refresh(job)
    assert job.status == "processing"
    db.close()


def test_interrupted_job_reports_stalled_advisory_without_db_mutation():
    from datetime import datetime, timezone, timedelta
    from app.routes.repository import get_analysis_status

    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    user = User(id=user_id, username=f"user_{uuid.uuid4().hex[:6]}", email=f"user_{uuid.uuid4().hex[:6]}@example.com", hashed_password="pw")
    repo = Repository(id=repo_id, user_id=user_id, url=f"https://github.com/test/{repo_id}", owner="test", name=repo_id, default_branch="main", primary_language="python", status="analyzing")

    # Stalled job: heartbeat stopped 8 minutes ago
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stale_time = now - timedelta(minutes=8)
    job = AnalysisJob(
        id=str(uuid.uuid4()),
        repository_id=repo_id,
        status="processing",
        current_stage="embedding",
        progress_percent=90,
        message="Generating embeddings and indexing in ChromaDB...",
        created_at=stale_time,
        updated_at=stale_time,
    )
    db.add_all([user, repo, job])
    db.commit()

    resp = get_analysis_status(repo_id=repo_id, db=db, current_user=user)
    # Advisory flag must be True
    assert resp.is_stalled is True
    assert "No progress reported" in resp.message

    # Crucial invariant: The database record must remain completely UNMUTATED (read-only polling)
    db.refresh(job)
    assert job.status == "processing"
    db.refresh(repo)
    assert repo.status == "analyzing"
    db.close()


@pytest.mark.asyncio
async def test_retry_conflict_on_active_job():
    from unittest.mock import patch
    from fastapi import HTTPException
    from app.services.repo_service import create_repository

    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    norm_url = f"https://github.com/test/repo-{repo_id[:8]}"
    user = User(id=user_id, username=f"user_{uuid.uuid4().hex[:6]}", email=f"user_{uuid.uuid4().hex[:6]}@example.com", hashed_password="pw")
    repo = Repository(id=repo_id, user_id=user_id, url=norm_url, owner="test", name=f"repo-{repo_id[:8]}", default_branch="main", primary_language="python", status="analyzing")

    # Active job in processing
    job = AnalysisJob(
        id=str(uuid.uuid4()),
        repository_id=repo_id,
        status="processing",
        current_stage="embedding",
        progress_percent=90,
    )
    db.add_all([user, repo, job])
    db.commit()

    # Attempting to submit or retry an active repository must raise HTTP 409 Conflict
    with patch("app.services.repo_service.get_github_repo_metadata", return_value={
        "owner": "test",
        "name": f"repo-{repo_id[:8]}",
        "default_branch": "main",
        "primary_language": "Python",
        "description": "Mock active repo"
    }):
        with pytest.raises(HTTPException) as exc_info:
            await create_repository(db, user, norm_url, reanalyze=True)

    assert exc_info.value.status_code == 409
    assert "already in progress" in str(exc_info.value.detail)
    db.close()


@pytest.mark.asyncio
async def test_preservation_of_completed_analyses():
    from unittest.mock import patch
    from app.services.repo_service import create_repository

    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    norm_url = f"https://github.com/test/repo-comp-{repo_id[:8]}"
    user = User(id=user_id, username=f"user_{uuid.uuid4().hex[:6]}", email=f"user_{uuid.uuid4().hex[:6]}@example.com", hashed_password="pw")
    repo = Repository(
        id=repo_id,
        user_id=user_id,
        url=norm_url,
        owner="test",
        name=f"repo-comp-{repo_id[:8]}",
        default_branch="main",
        primary_language="python",
        status="completed",
        file_count=5,
        chunk_count=15,
    )
    job = AnalysisJob(
        id=str(uuid.uuid4()),
        repository_id=repo_id,
        status="completed",
        current_stage="completed",
        progress_percent=100,
    )
    db.add_all([user, repo, job])
    db.commit()

    # Submitting completed repo without reanalyze must reuse it without launching a pipeline
    with patch("app.services.repo_service.get_github_repo_metadata", return_value={
        "owner": "test",
        "name": f"repo-comp-{repo_id[:8]}",
        "default_branch": "main",
        "primary_language": "Python",
        "description": "Mock completed repo"
    }):
        r, j, is_already_completed = await create_repository(db, user, norm_url, reanalyze=False)

    assert is_already_completed is True
    assert r.id == repo_id
    assert j.id == job.id
    db.close()


def test_embedding_progress_db_commit_failure_recovers_and_completes():
    from unittest.mock import patch
    from app.services.ingestion_service import run_ingestion_pipeline, update_job_status
    from app.ai.vector_store import get_vector_store

    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    user = User(id=user_id, username=f"u_{uuid.uuid4().hex[:6]}", email=f"u_{uuid.uuid4().hex[:6]}@example.com", hashed_password="pw")
    repo = Repository(
        id=repo_id,
        user_id=user_id,
        url="https://github.com/test-owner/test-progress-recover",
        owner="test-owner",
        name="test-progress-recover",
        default_branch="main",
        primary_language="Python",
        status="pending",
    )
    job = AnalysisJob(
        id=job_id,
        repository_id=repo_id,
        status="pending",
        current_stage="pending",
        progress_percent=0,
    )
    db.add_all([user, repo, job])
    db.commit()
    db.close()

    orig_update_job_status = update_job_status
    progress_failures = []

    def mock_update_job_status(s_db, s_job, stage, percent, msg, status="processing"):
        if "Generating embeddings and indexing in ChromaDB (" in msg:
            progress_failures.append(True)
            raise RuntimeError("Simulated DB lock/commit failure during progress callback")
        return orig_update_job_status(s_db, s_job, stage, percent, msg, status=status)

    with patch("app.services.ingestion_service.validate_github_url", return_value=True), \
         patch("app.services.ingestion_service.download_or_clone_repository", return_value="dummy_dir"), \
         patch("app.services.ingestion_service.cleanup_repository_dir", return_value=None), \
         patch("app.services.ingestion_service.process_repository_files", return_value=(
             [{
                 "file_path": "main.py",
                 "file_name": "main.py",
                 "language": "python",
                 "size_bytes": 25,
                 "line_count": 2,
                 "sha_hash": "dummy_sha",
                 "content": "def test_recovery():\n    return 'clean'\n"
             }],
             [{
                 "chunk_id": f"chunk-prog-{uuid.uuid4().hex[:6]}",
                 "content": "def test_recovery():\n    return 'clean'\n",
                 "file_path": "main.py",
                 "language": "python",
                 "start_line": 1,
                 "end_line": 2,
                 "symbol_name": "test_recovery",
                 "sha_hash": "dummy_chunk_sha",
             }]
         )), \
         patch("app.services.ingestion_service.SessionLocal", side_effect=TestingSessionLocal), \
         patch("app.services.ingestion_service.update_job_status", side_effect=mock_update_job_status):

        run_ingestion_pipeline(job_id, repo_id)

    assert len(progress_failures) > 0, "Expected at least one progress callback invocation to fail"

    verify_db = TestingSessionLocal()
    completed_job = verify_db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    completed_repo = verify_db.query(Repository).filter(Repository.id == repo_id).first()

    assert completed_job is not None
    assert completed_job.status == "completed", f"Expected job status 'completed', got '{completed_job.status}' (details: {completed_job.error_details})"
    assert completed_job.progress_percent == 100
    assert completed_repo is not None
    assert completed_repo.status == "completed"
    assert completed_repo.chunk_count == 1
    verify_db.close()

    try:
        vs = get_vector_store()
        vs.delete_collection(repo_id)
    except Exception:
        pass
