import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.user import User
from app.models.job import AnalysisJob
from app.github.client import parse_github_url, get_github_repo_metadata

async def create_repository(db: Session, user: User, url: str) -> Tuple[Repository, AnalysisJob]:
    owner, name = parse_github_url(url)
    metadata = await get_github_repo_metadata(owner, name)

    # Check if repo exists for user
    existing_repo = db.query(Repository).filter(
        Repository.user_id == user.id,
        Repository.url == url
    ).first()

    if existing_repo:
        repo = existing_repo
        repo.status = "analyzing"
    else:
        repo = Repository(
            user_id=user.id,
            url=url,
            owner=metadata["owner"],
            name=metadata["name"],
            default_branch=metadata["default_branch"],
            primary_language=metadata["primary_language"],
            description=metadata["description"],
            status="analyzing"
        )
        db.add(repo)
        db.flush()

    # Create analysis job
    job = AnalysisJob(
        repository_id=repo.id,
        status="pending",
        current_stage="validating",
        progress_percent=0,
        message="Queued for analysis..."
    )
    db.add(job)
    db.commit()
    db.refresh(repo)
    db.refresh(job)

    return repo, job

def get_user_repositories(db: Session, user_id: str) -> List[Repository]:
    return db.query(Repository).filter(Repository.user_id == user_id).order_by(Repository.updated_at.desc()).all()

def get_repository_by_id(db: Session, repo_id: str, user_id: str) -> Optional[Repository]:
    return db.query(Repository).filter(Repository.id == repo_id, Repository.user_id == user_id).first()

def get_repository_files(db: Session, repo_id: str) -> List[RepositoryFile]:
    return db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo_id).all()

def build_file_tree(files: List[RepositoryFile]) -> List[Dict[str, Any]]:
    """
    Transforms flat repository files into a recursive tree structure for the frontend file explorer.
    """
    root = []

    for f in files:
        parts = f.file_path.split("/")
        current_level = root

        for idx, part in enumerate(parts):
            is_file = (idx == len(parts) - 1)
            existing_node = next((node for node in current_level if node["name"] == part), None)

            if existing_node:
                current_level = existing_node["children"] if "children" in existing_node else []
            else:
                if is_file:
                    new_node = {
                        "name": part,
                        "path": f.file_path,
                        "type": "file",
                        "id": f.id,
                        "language": f.language,
                        "size": f.size_bytes,
                        "line_count": f.line_count
                    }
                else:
                    new_node = {
                        "name": part,
                        "path": "/".join(parts[:idx+1]),
                        "type": "directory",
                        "children": []
                    }
                current_level.append(new_node)
                if not is_file:
                    current_level = new_node["children"]

    return root

def generate_architecture_graph(files: List[RepositoryFile]) -> Dict[str, Any]:
    """
    Analyzes project files to synthesize nodes and edges for architecture visualization.
    """
    nodes = [
        {"id": "user", "label": "Client / User Browser", "type": "frontend", "category": "Client"},
    ]
    edges = []

    has_frontend = False
    has_backend = False
    has_db = False
    has_auth = False
    has_api = False

    paths = [f.file_path.lower() for f in files]

    for p in paths:
        if any(k in p for k in ["react", "components", "pages", "view", "app.jsx", "app.tsx", "package.json"]):
            has_frontend = True
        if any(k in p for k in ["server", "backend", "api", "routes", "controllers", "main.py", "app.py"]):
            has_backend = True
            has_api = True
        if any(k in p for k in ["db", "database", "models", "schema", "prisma", "alembic", "sql"]):
            has_db = True
        if any(k in p for k in ["auth", "jwt", "login", "passport", "session"]):
            has_auth = True

    if has_frontend:
        nodes.append({"id": "frontend", "label": "React Frontend", "type": "frontend", "category": "UI Layer"})
        edges.append({"source": "user", "target": "frontend", "label": "HTTP Requests"})

    if has_api or has_backend:
        nodes.append({"id": "api", "label": "FastAPI / Node Backend API", "type": "backend", "category": "API Layer"})
        if has_frontend:
            edges.append({"source": "frontend", "target": "api", "label": "REST / JSON API"})
        else:
            edges.append({"source": "user", "target": "api", "label": "REST API"})

    if has_auth:
        nodes.append({"id": "auth", "label": "JWT Auth Service", "type": "auth", "category": "Security"})
        edges.append({"source": "api", "target": "auth", "label": "Token Validation"})

    if has_db:
        nodes.append({"id": "db", "label": "PostgreSQL / SQLite Database", "type": "database", "category": "Data Layer"})
        edges.append({"source": "api", "target": "db", "label": "SQL Queries / ORM"})

    # Fallback nodes if project is small
    if len(nodes) == 1:
        nodes.append({"id": "app", "label": "Application Codebase", "type": "backend", "category": "Core"})
        edges.append({"source": "user", "target": "app", "label": "Execution"})

    return {"nodes": nodes, "edges": edges}
