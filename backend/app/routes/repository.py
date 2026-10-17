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
        repo, job, is_already_completed = await create_repository(
            db, current_user, repo_data.url, reanalyze=bool(repo_data.reanalyze)
        )
        if not is_already_completed and job:
            # Dispatch background ingestion pipeline
            background_tasks.add_task(run_ingestion_pipeline, job.id, repo.id)
            message = "Repository queued for analysis."
        else:
            message = "Repository already analyzed and ready."

        return {
            "repository": RepositoryResponse.model_validate(repo),
            "job_id": job.id if job else None,
            "status": repo.status,
            "is_already_completed": is_already_completed,
            "message": message
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
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    job = db.query(AnalysisJob).filter(AnalysisJob.repository_id == repo.id).order_by(AnalysisJob.created_at.desc(), AnalysisJob.updated_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found for repository.")

    job_status = job.status
    job_stage = job.current_stage
    job_percent = job.progress_percent
    job_msg = job.message
    if repo.status == "completed" and job_status != "failed":
        job_status = "completed"
        job_stage = "completed"
        job_percent = 100
        job_msg = job_msg or "Analysis complete!"

    return AnalysisStatusResponse(
        job_id=job.id,
        repository_id=job.repository_id,
        status=job_status,
        current_stage=job_stage,
        progress_percent=job_percent,
        message=job_msg,
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

@router.get("/{repo_id}/files/{file_id:path}", response_model=RepositoryFileResponse)
def get_file_content(repo_id: str, file_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    from sqlalchemy import or_
    file_obj = db.query(RepositoryFile).filter(
        RepositoryFile.repository_id == repo.id,
        or_(RepositoryFile.id == file_id, RepositoryFile.file_path == file_id)
    ).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found.")
    return file_obj

@router.get("/{repo_id}/architecture")
def get_architecture(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    graph = generate_architecture_graph(repo_id, db)
    return graph

@router.get("/{repo_id}/overview")
def get_repository_overview(repo_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = get_repository_by_id(db, repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")

    readme_file = db.query(RepositoryFile).filter(
        RepositoryFile.repository_id == repo_id,
        RepositoryFile.file_name.ilike("readme%")
    ).first()

    readme_preview = ""
    if readme_file and readme_file.content:
        readme_preview = readme_file.content[:3000]

    return {
        "id": repo.id,
        "name": repo.name,
        "owner": repo.owner,
        "url": repo.url,
        "description": repo.description or f"Public repository {repo.owner}/{repo.name}",
        "primary_language": repo.primary_language,
        "file_count": repo.file_count,
        "chunk_count": repo.chunk_count,
        "status": repo.status,
        "summary_json": repo.summary_json,
        "readme_preview": readme_preview
    }
