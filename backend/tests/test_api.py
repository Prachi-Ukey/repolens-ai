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
