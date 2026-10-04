from pydantic import BaseModel
from typing import Optional

class DocGenRequest(BaseModel):
    doc_type: str # "readme", "api", "architecture", "onboarding"

class DocGenResponse(BaseModel):
    repository_id: str
    doc_type: str
    content: str
