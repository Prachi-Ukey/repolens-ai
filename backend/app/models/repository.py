import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

def get_default_embedding_model() -> str:
    """Returns sensible default embedding model based on current application config."""
    try:
        from app.ai.embeddings import get_active_embedding_model_name
        return get_active_embedding_model_name()
    except Exception:
        return "all-MiniLM-L6-v2"


def get_default_embedding_dimension() -> int:
    """Returns sensible default embedding dimension based on current application config."""
    try:
        from app.config import settings
        return getattr(settings, "EMBEDDING_DIMENSION", 384)
    except Exception:
        return 384


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    url = Column(String, nullable=False)
    owner = Column(String, nullable=False)
    name = Column(String, nullable=False)
    default_branch = Column(String, default="main")
    primary_language = Column(String, default="Unknown")
    file_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    embedding_model = Column(String, default=get_default_embedding_model, nullable=True)
    embedding_dimension = Column(Integer, default=get_default_embedding_dimension, nullable=True)
    status = Column(String, default="pending") # pending, analyzing, completed, failed
    description = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True) # AI overview summary
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="repositories")
    files = relationship("RepositoryFile", back_populates="repository", cascade="all, delete-orphan")
    chunks = relationship("CodeChunk", back_populates="repository", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="repository", cascade="all, delete-orphan")
    analysis_jobs = relationship("AnalysisJob", back_populates="repository", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        if "embedding_model" not in kwargs or kwargs["embedding_model"] is None:
            kwargs["embedding_model"] = get_default_embedding_model()
        if "embedding_dimension" not in kwargs or (
            kwargs["embedding_dimension"] is None and kwargs.get("embedding_model") != "unknown"
        ):
            kwargs["embedding_dimension"] = get_default_embedding_dimension()
        super().__init__(**kwargs)
