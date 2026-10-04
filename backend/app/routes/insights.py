from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.insights import CodeInsightsResponse, DependencyAnalysisResponse
from app.services.auth_service import get_current_user
from app.services.repo_service import get_repository_by_id
from app.services.insights_service import analyze_code_insights, analyze_dependencies

router = APIRouter(prefix="/repositories/{repo_id}", tags=["Code Insights & Dependencies"])

@router.get("/insights", response_model=CodeInsightsResponse)
def get_insights(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return analyze_code_insights(db, repo_id)

@router.get("/dependencies", response_model=DependencyAnalysisResponse)
def get_dependencies(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return analyze_dependencies(db, repo_id)
