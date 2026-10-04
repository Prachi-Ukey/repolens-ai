from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.database import get_db
from app.models.user import User
from app.models.file import RepositoryFile
from app.models.job import AnalysisJob
from app.schemas.repository import RepositoryCreate, RepositoryResponse, RepositoryFileResponse, AnalysisStatusResponse
from app.services.auth_service import get_current_user
from app.services.repo_service import create_repository, get_user_repositories, get_repository_by_id, get_repository_files, build_file_tree, generate_architecture_graph
from app.services.ingestion_service import run_ingestion_pipeline
from app.ai.vector_store import get_vector_store

router = APIRouter(prefix="/repositories", tags=["Repositories"])

@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def add_repository(
    repo_data: RepositoryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        repo, job = await create_repository(db, current_user, repo_data.url)
        # Dispatch background ingestion pipeline
        background_tasks.add_task(run_ingestion_pipeline, job.id, repo.id)
        return {
            "repository": RepositoryResponse.model_validate(repo),
            "job_id": job.id,
            "message": "Repository queued for analysis."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[RepositoryResponse])
def list_repositories(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_user_repositories(db, current_user.id)

@router.get("/{repo_id}", response_model=RepositoryResponse)
def get_repository(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return repo

@router.delete("/{repo_id}")
def delete_repository(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")

    # Clean Chroma collection
    vector_store = get_vector_store()
    vector_store.delete_collection(repo.id)

    db.delete(repo)
    db.commit()
    return {"message": "Repository deleted successfully."}

@router.get("/{repo_id}/analysis-status", response_model=AnalysisStatusResponse)
def get_analysis_status(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(AnalysisJob).filter(AnalysisJob.repository_id == repo_id).order_by(AnalysisJob.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found for repository.")
    return AnalysisStatusResponse(
        job_id=job.id,
        repository_id=job.repository_id,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent,
        message=job.message,
        error_details=job.error_details
    )

@router.get("/{repo_id}/files")
def get_files(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")

    files = get_repository_files(db, repo_id)
    tree = build_file_tree(files)
    return {
        "repository_id": repo_id,
        "total_files": len(files),
        "tree": tree,
        "flat_files": [RepositoryFileResponse.model_validate(f) for f in files]
    }

@router.get("/{repo_id}/files/{file_id}", response_model=RepositoryFileResponse)
def get_file_content(repo_id: str, file_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    file_obj = db.query(RepositoryFile).filter(
        RepositoryFile.id == file_id,
        RepositoryFile.repository_id == repo_id
    ).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found.")
    return file_obj

@router.get("/{repo_id}/architecture")
def get_architecture(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    files = get_repository_files(db, repo_id)
    graph = generate_architecture_graph(files)
    return graph
