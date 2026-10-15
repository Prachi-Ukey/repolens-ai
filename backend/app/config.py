import os
import sys

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.append(user_site)

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from typing import Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_env = os.path.join(BACKEND_DIR, ".env")
_default_chroma = os.path.join(BACKEND_DIR, "chroma_data")
_default_db_path = os.path.join(BACKEND_DIR, "repolens.db").replace("\\", "/")
_default_db = f"sqlite:///{_default_db_path}"

# Documented development-only fallback key.
# This key is strictly forbidden in production (APP_ENV=production).
DEV_FALLBACK_SECRET_KEY = "dev-insecure-secret-key-only-for-local-development-2026"

KNOWN_INSECURE_SECRETS = {
    "",
    "repolens-super-secret-jwt-key-change-in-production-2026",
    "CHANGE_ME_GENERATE_A_SECURE_RANDOM_SECRET",
    "change-me",
    "changeme",
    "secret",
    "secret-key",
    "your-secret-key",
    "your_secret_key_here",
    DEV_FALLBACK_SECRET_KEY,
}

class Settings(BaseSettings):
    APP_NAME: str = "RepoLens AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Security & Auth
    # Production requires a cryptographically random SECRET_KEY (min 16 chars).
    # Local development uses DEV_FALLBACK_SECRET_KEY if not configured.
    SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    CORS_ORIGINS: str = "*"

    # Database
    DATABASE_URL: str = _default_db

    # AI Configuration
    LLM_PROVIDER: str = "ollama"  # mock, ollama, openai, gemini
    EMBEDDING_PROVIDER: str = "mock"  # mock, local, openai, gemini
    EMBEDDING_DIMENSION: int = 384
    OLLAMA_MODEL: str = "qwen2.5-coder:1.5b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # GitHub API
    GITHUB_TOKEN: Optional[str] = None

    # Vector Storage
    CHROMA_PERSIST_DIRECTORY: str = _default_chroma

    def __init__(self, **values):
        super().__init__(**values)
        if self.CHROMA_PERSIST_DIRECTORY.startswith("./") or self.CHROMA_PERSIST_DIRECTORY == "chroma_data":
            rel = self.CHROMA_PERSIST_DIRECTORY.replace("./", "")
            self.CHROMA_PERSIST_DIRECTORY = os.path.abspath(os.path.join(BACKEND_DIR, rel))
        if self.DATABASE_URL == "sqlite:///./repolens.db":
            self.DATABASE_URL = _default_db

        is_production = str(self.APP_ENV).strip().lower() == "production"
        current_secret = (self.SECRET_KEY or "").strip()

        if is_production:
            if (
                not current_secret
                or current_secret in KNOWN_INSECURE_SECRETS
                or current_secret.upper().startswith("CHANGE_ME")
                or len(current_secret) < 16
            ):
                raise ValueError(
                    "Production security configuration error: In production (APP_ENV=production), "
                    "a secure, non-default SECRET_KEY must be provided via environment variables. "
                    "Found invalid, default, or placeholder secret key. "
                    "Please set SECRET_KEY to a cryptographically random string with at least 16 characters "
                    "(e.g., generated with `openssl rand -hex 32`)."
                )

            # Validate required production AI provider API keys
            emb_provider = str(self.EMBEDDING_PROVIDER).strip().lower()
            if emb_provider == "gemini" and not (self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "GEMINI_API_KEY must be configured when EMBEDDING_PROVIDER='gemini'. "
                    "Found missing or empty API key. Please supply GEMINI_API_KEY in your environment variables."
                )
            elif emb_provider == "openai" and not (self.OPENAI_API_KEY and self.OPENAI_API_KEY.strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "OPENAI_API_KEY must be configured when EMBEDDING_PROVIDER='openai'. "
                    "Found missing or empty API key. Please supply OPENAI_API_KEY in your environment variables."
                )

            llm_prov = str(self.LLM_PROVIDER).strip().lower()
            if llm_prov == "gemini" and not (self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "GEMINI_API_KEY must be configured when LLM_PROVIDER='gemini'. "
                    "Found missing or empty API key. Please supply GEMINI_API_KEY in your environment variables."
                )
            elif llm_prov == "openai" and not (self.OPENAI_API_KEY and self.OPENAI_API_KEY.strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "OPENAI_API_KEY must be configured when LLM_PROVIDER='openai'. "
                    "Found missing or empty API key. Please supply OPENAI_API_KEY in your environment variables."
                )
        else:
            # Development-only fallback:
            if not current_secret or current_secret in KNOWN_INSECURE_SECRETS or current_secret.upper().startswith("CHANGE_ME"):
                self.SECRET_KEY = DEV_FALLBACK_SECRET_KEY

    class Config:
        env_file = _default_env if os.path.exists(_default_env) else ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
