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
    yield
    Base.metadata.drop_all(bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
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
