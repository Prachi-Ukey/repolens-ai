import os
import sys

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.append(user_site)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

try:
    from pydantic import field_validator
    def pre_validator(*fields):
        return field_validator(*fields, mode="before")
except ImportError:
    from pydantic import validator
    def pre_validator(*fields):
        return validator(*fields, pre=True, allow_reuse=True)

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
    EMBEDDING_PROVIDER: str = "local"  # mock, local, openai, gemini
    EMBEDDING_DIMENSION: int = 384
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    ONNX_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2-qint8"
    ONNX_MODEL_DIR: Optional[str] = None
    MOCK_EMBEDDING_MODEL: Optional[str] = None
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

    @pre_validator("EMBEDDING_DIMENSION")
    @classmethod
    def _parse_embedding_dimension(cls, v):
        if isinstance(v, str):
            cleaned = v.strip().strip("'\"")
            return int(cleaned)
        return v

    @pre_validator("EMBEDDING_PROVIDER")
    @classmethod
    def _parse_embedding_provider(cls, v):
        if isinstance(v, str):
            return v.strip().strip("'\"").lower()
        return v

    @pre_validator("APP_ENV", "LLM_PROVIDER")
    @classmethod
    def _normalize_lower_string(cls, v):
        if isinstance(v, str):
            return v.strip().strip("'\"").lower()
        return v

    @pre_validator(
        "SECRET_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "CORS_ORIGINS",
    )
    @classmethod
    def _strip_quotes_and_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip().strip("'\"")
        return v

    def __init__(self, **values):
        super().__init__(**values)
        if self.CHROMA_PERSIST_DIRECTORY.startswith("./") or self.CHROMA_PERSIST_DIRECTORY == "chroma_data":
            rel = self.CHROMA_PERSIST_DIRECTORY.replace("./", "")
            self.CHROMA_PERSIST_DIRECTORY = os.path.abspath(os.path.join(BACKEND_DIR, rel))
        if self.DATABASE_URL in ("sqlite:///./repolens.db", "sqlite:///repolens.db"):
            self.DATABASE_URL = _default_db

        # When Gemini is configured, always ensure 768 dimensions
        if self.EMBEDDING_PROVIDER == "gemini":
            self.EMBEDDING_DIMENSION = 768
        elif self.EMBEDDING_PROVIDER in ("onnx_local", "onnx"):
            self.EMBEDDING_DIMENSION = 384

        is_production = self.APP_ENV == "production"
        current_secret = self.SECRET_KEY or ""

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

            # Validate required production AI provider API keys without silent auto-switching
            emb_provider = self.EMBEDDING_PROVIDER
            if emb_provider == "local":
                has_st = False
                try:
                    import sentence_transformers
                    has_st = True
                except ImportError:
                    has_st = False

                if not has_st:
                    raise ValueError(
                        "Production configuration error: In production (APP_ENV=production), "
                        "EMBEDDING_PROVIDER is 'local', but sentence-transformers is not installed. "
                        "To use Gemini embeddings, set EMBEDDING_PROVIDER='gemini' and configure GEMINI_API_KEY. "
                        "To use local embeddings, install sentence-transformers."
                    )

            elif emb_provider == "gemini":
                if not self.GEMINI_API_KEY or not str(self.GEMINI_API_KEY).strip():
                    raise ValueError(
                        "Production configuration error: In production (APP_ENV=production), "
                        "GEMINI_API_KEY must be configured when EMBEDDING_PROVIDER='gemini'. "
                        "Found missing or empty API key. Please supply GEMINI_API_KEY in your environment variables."
                    )
                self.EMBEDDING_DIMENSION = 768
            elif emb_provider == "openai":
                if not self.OPENAI_API_KEY or not str(self.OPENAI_API_KEY).strip():
                    raise ValueError(
                        "Production configuration error: In production (APP_ENV=production), "
                        "OPENAI_API_KEY must be configured when EMBEDDING_PROVIDER='openai'. "
                        "Found missing or empty API key. Please supply OPENAI_API_KEY in your environment variables."
                    )
            elif emb_provider in ("onnx_local", "onnx"):
                has_ort = False
                try:
                    import onnxruntime
                    import tokenizers
                    has_ort = True
                except ImportError:
                    has_ort = False

                if not has_ort:
                    raise ValueError(
                        "Production configuration error: In production (APP_ENV=production), "
                        f"EMBEDDING_PROVIDER is '{emb_provider}', but onnxruntime and tokenizers are not installed. "
                        "Install onnxruntime and tokenizers via `pip install onnxruntime tokenizers`."
                    )
            elif emb_provider not in ("mock",):
                raise ValueError(
                    f"Production configuration error: Unsupported EMBEDDING_PROVIDER '{emb_provider}' in production. "
                    "Allowed production providers: 'gemini', 'openai', 'local', 'onnx_local', 'onnx'."
                )

            llm_prov = self.LLM_PROVIDER
            if llm_prov == "gemini" and (not self.GEMINI_API_KEY or not str(self.GEMINI_API_KEY).strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "GEMINI_API_KEY must be configured when LLM_PROVIDER='gemini'. "
                    "Found missing or empty API key. Please supply GEMINI_API_KEY in your environment variables."
                )
            elif llm_prov == "openai" and (not self.OPENAI_API_KEY or not str(self.OPENAI_API_KEY).strip()):
                raise ValueError(
                    "Production configuration error: In production (APP_ENV=production), "
                    "OPENAI_API_KEY must be configured when LLM_PROVIDER='openai'. "
                    "Found missing or empty API key. Please supply OPENAI_API_KEY in your environment variables."
                )
        else:
            # Development-only fallback:
            if not current_secret or current_secret in KNOWN_INSECURE_SECRETS or current_secret.upper().startswith("CHANGE_ME"):
                self.SECRET_KEY = DEV_FALLBACK_SECRET_KEY

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=_default_env if os.path.exists(_default_env) else ".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )
    else:
        class Config:
            env_file = _default_env if os.path.exists(_default_env) else ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"
            case_sensitive = False

settings = Settings()
