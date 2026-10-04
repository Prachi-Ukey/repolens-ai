from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.docgen import DocGenRequest, DocGenResponse
from app.services.auth_service import get_current_user
from app.services.repo_service import get_repository_by_id
from app.services.doc_service import generate_repository_docs

router = APIRouter(prefix="/repositories/{repo_id}", tags=["Documentation Generator"])

@router.post("/generate-documentation", response_model=DocGenResponse)
def generate_docs(
    repo_id: str,
    request: DocGenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return generate_repository_docs(db, repo_id, request.doc_type)
