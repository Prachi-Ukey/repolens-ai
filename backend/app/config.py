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

class Settings(BaseSettings):
    APP_NAME: str = "RepoLens AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Security & Auth
    SECRET_KEY: str = "repolens-super-secret-jwt-key-change-in-production-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Database
    DATABASE_URL: str = "sqlite:///./repolens.db"

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
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
