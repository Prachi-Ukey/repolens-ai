from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime

class RepositoryCreate(BaseModel):
    url: str

class RepositoryResponse(BaseModel):
    id: str
    user_id: str
    url: str
    owner: str
    name: str
    default_branch: str
    primary_language: str
    file_count: int
    chunk_count: int
    status: str
    description: Optional[str] = None
    summary_json: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class RepositoryFileResponse(BaseModel):
    id: str
    repository_id: str
    file_path: str
    file_name: str
    language: str
    size_bytes: int
    line_count: int
    sha_hash: Optional[str] = None
    content: Optional[str] = None

    class Config:
        from_attributes = True

class AnalysisStatusResponse(BaseModel):
    job_id: str
    repository_id: str
    status: str
    current_stage: str
    progress_percent: int
    message: Optional[str] = None
    error_details: Optional[str] = None
