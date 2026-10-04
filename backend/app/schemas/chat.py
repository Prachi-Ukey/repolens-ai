from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class QuestionRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    top_k: int = 5

class SourceCitation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    snippet: str

class QuestionResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: List[SourceCitation]
    grounded: bool = True

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    sender: str
    content: str
    sources_json: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: str
    repository_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True
