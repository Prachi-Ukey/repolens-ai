from sqlalchemy.orm import Session
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.ai.llm import get_llm_provider
from app.schemas.docgen import DocGenResponse

def generate_repository_docs(db: Session, repository_id: str, doc_type: str) -> DocGenResponse:
    repo = db.query(Repository).filter(Repository.id == repository_id).first()
    if not repo:
        raise ValueError("Repository not found.")

    files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repository_id).all()
    file_list_str = "\n".join([f"- `{f.file_path}` ({f.language}, {f.line_count} lines)" for f in files[:20]])

    llm = get_llm_provider()

    system_prompt = "You are RepoLens AI documentation generator. Generate professional, clear markdown documentation."

    user_prompt = f"""Generate a comprehensive, production-grade {doc_type.upper()} document for repository `{repo.owner}/{repo.name}`.

Repository Metadata:
- Primary Language: {repo.primary_language}
- Total Files: {repo.file_count}
- Description: {repo.description}

Discovered Key Files:
{file_list_str}

Format the response as clean Markdown with headings, bullet points, code blocks, and clear developer instructions.
"""
    # Use LLM or structured fallback generator
    try:
        content = llm.generate_response(system_prompt, user_prompt)
    except Exception:
        content = f"# Documentation for {repo.owner}/{repo.name}\n\nGenerated {doc_type} documentation based on {repo.file_count} analyzed repository files."

    return DocGenResponse(
        repository_id=repository_id,
        doc_type=doc_type,
        content=content
    )
