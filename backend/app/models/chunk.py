import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class CodeChunk(Base):
    __tablename__ = "code_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    repository_id = Column(String, ForeignKey("repositories.id"), nullable=False)
    file_id = Column(String, ForeignKey("repository_files.id"), nullable=False)
    file_path = Column(String, nullable=False, index=True)
    language = Column(String, default="text")
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    symbol_name = Column(String, nullable=True) # function / class / section name
    chunk_id = Column(String, nullable=False) # sha or custom id
    content = Column(Text, nullable=False)
    sha_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    repository = relationship("Repository", back_populates="chunks")
    file = relationship("RepositoryFile", back_populates="chunks")
