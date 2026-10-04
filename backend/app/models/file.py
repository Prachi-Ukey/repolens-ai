import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class RepositoryFile(Base):
    __tablename__ = "repository_files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    repository_id = Column(String, ForeignKey("repositories.id"), nullable=False)
    file_path = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    language = Column(String, default="text")
    size_bytes = Column(Integer, default=0)
    line_count = Column(Integer, default=0)
    sha_hash = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    repository = relationship("Repository", back_populates="files")
    chunks = relationship("CodeChunk", back_populates="file", cascade="all, delete-orphan")
