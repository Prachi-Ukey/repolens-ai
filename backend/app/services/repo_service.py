import json
import os
from collections import defaultdict
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

def get_node_level(category: str, label: str) -> int:
    cat = category.lower()
    lbl = label.lower()
    if "presentation" in cat or "frontend" in cat or "ui" in cat or "client" in cat:
        return 1
    if "application" in cat or "entry" in cat or "app" in lbl:
        return 2
    if "api" in cat or "route" in cat or "security" in cat or "auth" in cat:
        return 3
    if "service" in cat or "business" in cat or "domain" in cat or "module" in cat:
        return 4
    if "processing" in cat or "logic" in cat:
        return 5
    if "ai" in cat or "ml" in cat or "model" in cat or "predict" in lbl or "yolo" in lbl:
        return 6
    if "data" in cat or "database" in cat or "storage" in cat or "sql" in lbl:
        return 7
    return 4

def generate_architecture_graph(repo_id: str, db: Session) -> Dict[str, Any]:
    from app.ai.vector_store import ChromaManager
    import os
    import re
    from collections import defaultdict
    
    chroma_mgr = ChromaManager()
    collection = chroma_mgr.get_or_create_collection(repo_id)
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])
    
    # 1. Global framework detection
    frameworks = {
        "flask": ("Flask Application", "backend", "Application"),
        "fastapi": ("FastAPI Application", "backend", "Application"),
        "express": ("Express API", "backend", "Application"),
        "react": ("React Frontend", "frontend", "Presentation"),
        "django": ("Django Application", "backend", "Application"),
        "sqlalchemy": ("SQLAlchemy ORM", "database", "Data"),
        "pymongo": ("MongoDB", "database", "Data"),
        "mongoose": ("MongoDB", "database", "Data"),
        "yolo": ("YOLO Model", "backend", "AI / ML"),
        "ultralytics": ("YOLO Model", "backend", "AI / ML"),
        "socket.io": ("Socket.IO", "backend", "API"),
        "pytorch": ("PyTorch Model", "backend", "AI / ML"),
        "tensorflow": ("TensorFlow Model", "backend", "AI / ML"),
    }
    
    detected_frameworks = {}
    for meta in metadatas:
        if not meta: continue
        imports = meta.get("imported_symbols", "").lower()
        for kw, info in frameworks.items():
            if kw in imports and kw not in detected_frameworks:
                detected_frameworks[kw] = info

    # 2. Extract components
    components = {}
    def add_comp(c_id, label, tpe, cat, file, func=None):
        if c_id not in components:
            components[c_id] = {
                "id": c_id, "label": label, "type": tpe, "category": cat,
                "files": set(), "functions": set()
            }
        if file:
            components[c_id]["files"].add(file)
        if func:
            components[c_id]["functions"].add(func)

    for kw, (label, tpe, cat) in detected_frameworks.items():
        add_comp(f"fw_{kw}", label, tpe, cat, "")
        
    def clean_name(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        s3 = s2.replace('_', ' ').replace('-', ' ')
        words = s3.split()
        stop = {"get", "set", "calculate", "fetch", "update", "delete", "create", "handler", "util", "helper", "view", "edit", "route", "api", "service"}
        filtered = [w for w in words if w.lower() not in stop]
        return " ".join(filtered).title() if filtered else s3.title()
        
    def determine_category(n):
        n = n.lower()
        if "auth" in n or "login" in n or "session" in n: return "auth", "Security"
        if "db" in n or "data" in n or "model" in n or "sql" in n: return "database", "Data"
        if "ui" in n or "react" in n or "page" in n: return "frontend", "Presentation"
        if "api" in n or "dashboard" in n or "route" in n: return "backend", "API"
        if "predict" in n or "detect" in n or "yolo" in n or "ml" in n or "ai" in n: return "backend", "AI / ML"
        if "app" in n or "main" in n or "server" in n: return "backend", "Application"
        if "price" in n or "cost" in n or "recommend" in n or "damage" in n: return "backend", "Business Logic"
        return "backend", "Service"
        
    symbol_to_comp = {}
    ignore_files = ["__init__.py", "setup.py", "test", "util", "helper", "config", "type", "interface"]
    
    for meta in metadatas:
        if not meta: continue
        path = meta.get("file_path", "")
        filename = os.path.basename(path).lower()
        if any(ig in filename for ig in ignore_files):
            continue
            
        symbol = meta.get("symbol_name", "")
        if not symbol: continue
        
        name_base = os.path.splitext(filename)[0]
        strong_terms = ["auth", "dashboard", "predict", "damage", "price", "cost", "user", "chat", "image", "upload", "session", "recommend", "insurance", "analysis"]
        
        comp_name = None
        if any(term in name_base for term in strong_terms):
            comp_name = name_base
        elif any(term in symbol.lower() for term in strong_terms):
            comp_name = symbol
            
        if not comp_name:
            if name_base not in ["app", "main", "server", "index"]:
                comp_name = name_base
            else:
                comp_name = symbol
                
        if comp_name:
            c_label = clean_name(comp_name)
            if len(c_label) < 3 or len(c_label) > 30:
                continue
            c_id = f"comp_{c_label.replace(' ', '_').lower()}"
            tpe, cat = determine_category(c_label)
            add_comp(c_id, c_label, tpe, cat, path, symbol)
            symbol_to_comp[symbol] = c_id

    # 3. Build Edges
    edges_counts = defaultdict(int)
    for meta in metadatas:
        if not meta: continue
        caller = meta.get("symbol_name", "")
        called_str = meta.get("called_symbols", "")
        
        caller_comp = symbol_to_comp.get(caller)
        if caller_comp and called_str:
            for callee in called_str.split(","):
                callee = callee.strip()
                callee_comp = symbol_to_comp.get(callee)
                if callee_comp and callee_comp != caller_comp:
                    edges_counts[(caller_comp, callee_comp)] += 1
                    
    app_fw = next((f"fw_{kw}" for kw in detected_frameworks if detected_frameworks[kw][2] == "Application"), None)
    if app_fw:
        for c_id, c_data in components.items():
            if c_data["category"] in ["API", "Security", "Presentation"] and c_id != app_fw:
                edges_counts[(app_fw, c_id)] += 1

    db_fw = next((f"fw_{kw}" for kw in detected_frameworks if detected_frameworks[kw][2] == "Data"), None)
    if db_fw:
        for c_id, c_data in components.items():
            if c_data["category"] in ["Service", "Business Logic", "Security", "Application"] and c_id != db_fw:
                edges_counts[(c_id, db_fw)] += 1

    # 4. Filter top components
    comp_scores = defaultdict(int)
    for c_id in components:
        comp_scores[c_id] += len(components[c_id]["functions"])
        if c_id.startswith("fw_"):
            comp_scores[c_id] += 10
            
    for (src, dst), count in edges_counts.items():
        comp_scores[src] += count * 2
        comp_scores[dst] += count * 2
        
    sorted_comps = sorted(comp_scores.keys(), key=lambda k: comp_scores[k], reverse=True)
    top_comps = set(sorted_comps[:15])
    
    # 5. Format output
    nodes = []
    for c_id in top_comps:
        c_data = components[c_id]
        f_list = list(c_data["files"])
        file_disp = f_list[0] if f_list else ""
        if len(f_list) > 1: file_disp += f" (+{len(f_list)-1})"
        funcs = list(c_data["functions"])
        
        nodes.append({
            "id": c_id,
            "label": c_data["label"],
            "type": c_data["type"],
            "category": c_data["category"],
            "level": get_node_level(c_data["category"], c_data["label"]),
            "file": file_disp,
            "functions": funcs[:5] + (["..."] if len(funcs) > 5 else [])
        })
        
    edges = []
    for (src, dst), count in edges_counts.items():
        if src in top_comps and dst in top_comps:
            edges.append({"source": src, "target": dst, "label": ""})
            
    nodes.append({
        "id": "user",
        "label": "User / Client",
        "type": "frontend",
        "category": "Client",
        "level": 1,
        "file": "",
        "functions": []
    })
    
    entry_nodes = [n["id"] for n in nodes if n["category"] in ["Presentation", "Application"] and n["id"] != "user"]
    if not entry_nodes and len(nodes) > 1:
        # fallback, connect user to level 2 or 3
        entry_nodes = [n["id"] for n in nodes if n["level"] in [2, 3]]
    for e in entry_nodes:
        edges.append({"source": "user", "target": e, "label": "interacts"})

    return {"nodes": nodes, "edges": edges}
