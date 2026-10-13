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
_default_db = f"sqlite:///{os.path.join(BACKEND_DIR, 'repolens.db').replace('\\', '/')}"

class Settings(BaseSettings):
    APP_NAME: str = "RepoLens AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Security & Auth
    SECRET_KEY: str = "repolens-super-secret-jwt-key-change-in-production-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Database
    DATABASE_URL: str = _default_db

    # AI Configuration
    LLM_PROVIDER: str = "mock"  # mock, openai, gemini
    EMBEDDING_PROVIDER: str = "mock"  # mock, openai, gemini
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

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

    class Config:
        env_file = _default_env if os.path.exists(_default_env) else ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
