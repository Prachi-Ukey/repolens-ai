"""
Security and Configuration Tests for RepoLens AI.

Tests fail-fast production enforcement of SECRET_KEY,
development fallbacks, and CORS origin configuration.
"""

import pytest
from app.config import Settings, DEV_FALLBACK_SECRET_KEY, KNOWN_INSECURE_SECRETS
from app.utils.security import create_access_token, decode_access_token


class TestDevelopmentSecurityConfiguration:
    """Test convenient, safe local development fallback behavior."""

    def test_dev_fallback_when_secret_key_is_none(self):
        s = Settings(APP_ENV="development", SECRET_KEY=None)
        assert s.SECRET_KEY == DEV_FALLBACK_SECRET_KEY
        assert len(s.SECRET_KEY) >= 16

    def test_dev_fallback_when_secret_key_is_empty(self):
        s = Settings(APP_ENV="development", SECRET_KEY="   ")
        assert s.SECRET_KEY == DEV_FALLBACK_SECRET_KEY

    def test_dev_fallback_when_secret_key_is_placeholder(self):
        s = Settings(
            APP_ENV="development",
            SECRET_KEY="CHANGE_ME_GENERATE_A_SECURE_RANDOM_SECRET"
        )
        assert s.SECRET_KEY == DEV_FALLBACK_SECRET_KEY

    def test_dev_fallback_when_secret_key_is_old_default(self):
        s = Settings(
            APP_ENV="development",
            SECRET_KEY="repolens-super-secret-jwt-key-change-in-production-2026"
        )
        assert s.SECRET_KEY == DEV_FALLBACK_SECRET_KEY

    def test_dev_custom_secret_is_preserved(self):
        custom = "my-custom-dev-secret-value-12345"
        s = Settings(APP_ENV="development", SECRET_KEY=custom)
        assert s.SECRET_KEY == custom


class TestProductionSecurityFailFast:
    """Test strict fail-fast validation in production mode (APP_ENV=production)."""

    def test_production_fails_fast_when_secret_key_is_none(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(APP_ENV="production", SECRET_KEY=None)

    def test_production_fails_fast_when_secret_key_is_empty(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(APP_ENV="production", SECRET_KEY="   ")

    def test_production_fails_fast_with_placeholder(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(
                APP_ENV="production",
                SECRET_KEY="CHANGE_ME_GENERATE_A_SECURE_RANDOM_SECRET"
            )

    def test_production_fails_fast_with_change_me_prefix(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(
                APP_ENV="production",
                SECRET_KEY="CHANGE_ME_custom_string_here"
            )

    def test_production_fails_fast_with_old_hardcoded_default(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(
                APP_ENV="production",
                SECRET_KEY="repolens-super-secret-jwt-key-change-in-production-2026"
            )

    @pytest.mark.parametrize("insecure_key", [
        "secret",
        "secret-key",
        "change-me",
        "changeme",
        "your-secret-key",
        "your_secret_key_here",
        DEV_FALLBACK_SECRET_KEY,
    ])
    def test_production_fails_fast_with_known_insecure_keys(self, insecure_key):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(APP_ENV="production", SECRET_KEY=insecure_key)

    def test_production_fails_fast_with_short_key(self):
        with pytest.raises(ValueError, match="Production security configuration error"):
            Settings(APP_ENV="production", SECRET_KEY="short-secret-9")

    def test_production_error_message_never_leaks_supplied_secret_value(self):
        sensitive_attempt = "super-sensitive-raw-secret-attempt-12345"
        with pytest.raises(ValueError) as exc_info:
            Settings(
                APP_ENV="production",
                SECRET_KEY="CHANGE_ME_" + sensitive_attempt
            )
        err_msg = str(exc_info.value)
        assert "Production security configuration error" in err_msg
        assert sensitive_attempt not in err_msg
        assert "Received invalid" not in err_msg

        # Also test with a short secret containing sensitive string
        short_sensitive = "my-secret-pw"
        with pytest.raises(ValueError) as exc_info_short:
            Settings(
                APP_ENV="production",
                SECRET_KEY=short_sensitive
            )
        err_msg_short = str(exc_info_short.value)
        assert "Production security configuration error" in err_msg_short
        assert short_sensitive not in err_msg_short
        assert "Received invalid" not in err_msg_short

    def test_production_succeeds_with_strong_secret(self):
        strong_secret = "a8f3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1"
        s = Settings(APP_ENV="production", SECRET_KEY=strong_secret)
        assert s.SECRET_KEY == strong_secret
        assert s.APP_ENV == "production"


class TestAuthAndCorsConfiguration:
    """Test token generation and CORS parsing."""

    def test_jwt_token_flow_with_current_settings(self):
        payload = {"sub": "user-test-uuid", "role": "developer"}
        token = create_access_token(payload)
        assert isinstance(token, str)
        decoded = decode_access_token(token)
        assert decoded is not None
        assert decoded["sub"] == "user-test-uuid"
        assert decoded["role"] == "developer"

    def test_cors_origins_parsing_production(self):
        origins_str = "https://repolens-ai.vercel.app, https://repolens-preview.vercel.app"
        s = Settings(APP_ENV="production", SECRET_KEY="strong-valid-secret-key-123456", CORS_ORIGINS=origins_str)
        parsed = [o.strip() for o in s.CORS_ORIGINS.split(",") if o.strip()]
        assert len(parsed) == 2
        assert "https://repolens-ai.vercel.app" in parsed
        assert "https://repolens-preview.vercel.app" in parsed
