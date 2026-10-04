from pydantic import BaseModel
from typing import List, Optional

class InsightItem(BaseModel):
    severity: str # "high", "medium", "low", "info"
    category: str # "security", "complexity", "maintenance", "documentation"
    file_path: str
    line_number: Optional[int] = None
    title: str
    explanation: str
    suggestion: str

class CodeInsightsResponse(BaseModel):
    repository_id: str
    total_issues: int
    insights: List[InsightItem]

class DependencyItem(BaseModel):
    name: str
    version: str
    type: str # "production", "development", "transitive"
    ecosystem: str # "npm", "pip", "maven", etc.
    outdated: bool = False
    details: Optional[str] = None

class DependencyAnalysisResponse(BaseModel):
    repository_id: str
    ecosystems_found: List[str]
    total_dependencies: int
    dependencies: List[DependencyItem]
