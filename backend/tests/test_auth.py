import pytest
from app.utils.security import get_password_hash, verify_password, create_access_token, decode_access_token

def test_password_hashing():
    raw_pass = "SecurePass123!"
    hashed = get_password_hash(raw_pass)
    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPass", hashed) is False

def test_jwt_token_flow():
    data = {"sub": "user-uuid-123", "username": "testdev"}
    token = create_access_token(data)
    assert isinstance(token, str)

    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user-uuid-123"
    assert decoded["username"] == "testdev"
