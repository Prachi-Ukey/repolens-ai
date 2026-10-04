import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

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
