import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "RepoLens AI"
    assert data["status"] == "online"

def test_unauthenticated_protected_route():
    response = client.get("/repositories")
    assert response.status_code == 401


def test_health_endpoint_reports_active_embedding_configuration():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "embedding_provider" in data
    assert "embedding_dimension" in data
    assert "embedding_model" in data


def test_health_endpoint_reports_gemini_when_configured(monkeypatch):
    import app.config as config_mod
    import app.ai.embeddings as emb_mod
    from app.config import settings

    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 768)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-api-key-12345")
    monkeypatch.setattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setattr(settings, "SECRET_KEY", "super-secret-key-that-must-never-leak-123")

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["env"] == "production"
    assert data["embedding_provider"] == "gemini"
    assert data["configured_embedding_provider"] == "gemini"
    assert data["resolved_embedding_provider"] == "gemini"
    assert data["embedding_dimension"] == 768
    assert data["embedding_model"] == "gemini-embedding-001"

    # Confirm secrets are not exposed in health response
    response_str = response.text
    assert "test-api-key-12345" not in response_str
    assert "super-secret-key" not in response_str
    assert "SECRET_KEY" not in data
    assert "GEMINI_API_KEY" not in data


def test_health_endpoint_reports_unhealthy_when_embedding_provider_fails(monkeypatch):
    import app.ai.embeddings as emb_mod
    from app.config import settings

    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-secret-key-that-must-not-leak")

    def failing_get_embedding_provider():
        raise RuntimeError("Quota exceeded: secret-key-in-error-message")

    monkeypatch.setattr(emb_mod, "get_embedding_provider", failing_get_embedding_provider)

    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] in ("unhealthy", "degraded")
    assert data["status"] != "ok"
    assert data["configured_embedding_provider"] == "gemini"
    assert data["resolved_embedding_provider"] is None
    assert data["embedding_provider"] is None
    assert "test-secret-key-that-must-not-leak" not in response.text
    assert "secret-key-in-error-message" not in response.text
