import os
import json
import asyncio
from typing import Any, Dict
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.models.job import AnalysisJob
from app.github.client import validate_github_url, parse_github_url, get_github_repo_metadata
from app.github.repository import download_or_clone_repository, cleanup_repository_dir
from app.github.parser import process_repository_files
from app.ai.vector_store import get_vector_store
import logging

logger = logging.getLogger("repolens.services.ingestion")

def update_job_status(db: Session, job: AnalysisJob, stage: str, percent: int, msg: str, status: str = "processing"):
    job.current_stage = stage
    job.progress_percent = percent
    job.message = msg
    job.status = status
    db.commit()
    logger.info(f"Job {job.id} [{stage}] ({percent}%): {msg}")

def run_ingestion_pipeline(job_id: str, repository_id: str):
    """
    Synchronous ingestion runner executed in background worker thread.
    """
    db = SessionLocal()
    temp_dir = None
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        repo = db.query(Repository).filter(Repository.id == repository_id).first()

        if not job or not repo:
            logger.error("Job or repository not found for ingestion execution.")
            return

        # Stage 1: Validating
        update_job_status(db, job, "validating", 10, "Validating repository URL...")
        if not validate_github_url(repo.url):
            raise ValueError(f"Invalid GitHub URL: {repo.url}")

        owner, name = parse_github_url(repo.url)
        
        # Stage 2: Downloading
        update_job_status(db, job, "downloading", 25, f"Cloning repository {owner}/{name}...")
        temp_dir = download_or_clone_repository(repo.url)

        # Stage 3: Detecting Structure
        update_job_status(db, job, "detecting_structure", 45, "Detecting project structure & language files...")
        discovered_files, chunks = process_repository_files(temp_dir)

        # Stage 4: Parsing files & saving to Database
        update_job_status(db, job, "parsing_files", 65, f"Processed {len(discovered_files)} code files...")
        
        # Clear existing files & chunks for re-indexing
        db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).delete()
        db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).delete()

        file_obj_map = {}
        for f_data in discovered_files:
            file_obj = RepositoryFile(
                repository_id=repo.id,
                file_path=f_data["file_path"],
                file_name=f_data["file_name"],
                language=f_data["language"],
                size_bytes=f_data["size_bytes"],
                line_count=f_data["line_count"],
                sha_hash=f_data["sha_hash"],
                content=f_data["content"]
            )
            db.add(file_obj)
            db.flush() # Get file_obj.id
            file_obj_map[f_data["file_path"]] = file_obj.id

        # Stage 5: Chunking
        update_job_status(db, job, "chunking", 80, f"Created {len(chunks)} code chunks...")
        chunk_objs = []
        for c in chunks:
            file_id = file_obj_map.get(c["file_path"])
            if file_id:
                chunk_obj = CodeChunk(
                    repository_id=repo.id,
                    file_id=file_id,
                    file_path=c["file_path"],
                    language=c["language"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    symbol_name=c["symbol_name"],
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    sha_hash=c["sha_hash"]
                )
                chunk_objs.append(chunk_obj)
        db.bulk_save_objects(chunk_objs)

        # Stage 6: Embedding & Vector Storage
        update_job_status(db, job, "embedding", 90, "Generating embeddings and indexing in ChromaDB...")
        vector_store = get_vector_store()
        # Safe rebuild: Delete existing collection for this repo to purge stale/ignored chunks
        vector_store.delete_collection(repo.id)
        vector_store.upsert_chunks(repo.id, chunks)

        # Generate overview summary
        summary = {
            "overview": f"Repository {owner}/{name} contains {len(discovered_files)} source files and {len(chunks)} code chunks.",
            "primary_language": repo.primary_language,
            "total_files": len(discovered_files),
            "total_chunks": len(chunks),
            "top_files": [f["file_path"] for f in discovered_files[:10]]
        }
        repo.summary_json = json.dumps(summary)
        repo.file_count = len(discovered_files)
        repo.chunk_count = len(chunks)
        repo.status = "completed"

        update_job_status(db, job, "completed", 100, "Analysis complete!", status="completed")
        db.commit()

    except Exception as e:
        logger.error(f"Ingestion pipeline error: {e}", exc_info=True)
        if 'db' in locals() and 'job' in locals() and job:
            job.status = "failed"
            job.error_details = str(e)
            job.message = f"Analysis failed: {str(e)}"
            db.commit()
        if 'repo' in locals() and repo:
            repo.status = "failed"
            db.commit()
    finally:
        if temp_dir:
            cleanup_repository_dir(temp_dir)
        db.close()


def reindex_existing_repository(repository_id: str) -> Dict[str, Any]:
    """
    Safely reindexes an existing repository using stored files.
    Purges ignored files (e.g., dependency lockfiles, build artifacts) from DB,
    re-chunks valid files, resets the Chroma collection for this repository only,
    and upserts fresh chunks. Does NOT affect any other repository or user.
    """
    from app.github.parser import is_ignored_path, parse_file_into_chunks

    db = SessionLocal()
    try:
        repo = db.query(Repository).filter(Repository.id == repository_id).first()
        if not repo:
            raise ValueError(f"Repository '{repository_id}' not found.")

        files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).all()
        if not files:
            raise ValueError(f"No stored files found for repository '{repository_id}'.")

        logger.info(f"Reindexing repository {repo.name} ({repo.id}) with {len(files)} stored files...")

        # 1. Filter out ignored files (e.g. package-lock.json, build artifacts)
        valid_files = []
        ignored_file_ids = []
        for f in files:
            if is_ignored_path(f.file_path):
                ignored_file_ids.append(f.id)
                logger.info(f"Purging ignored file from index: {f.file_path}")
            else:
                valid_files.append(f)

        if ignored_file_ids:
            db.query(CodeChunk).filter(CodeChunk.file_id.in_(ignored_file_ids)).delete(synchronize_session=False)
            db.query(RepositoryFile).filter(RepositoryFile.id.in_(ignored_file_ids)).delete(synchronize_session=False)
            db.commit()

        # 2. Clear existing code chunks for this repository
        db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).delete()
        db.commit()

        # 3. Re-chunk valid files
        all_chunks = []
        chunk_objs = []
        for f in valid_files:
            file_chunks = parse_file_into_chunks(
                file_path=f.file_path,
                content=f.content,
                language=f.language
            )
            all_chunks.extend(file_chunks)
            for c in file_chunks:
                chunk_obj = CodeChunk(
                    repository_id=repo.id,
                    file_id=f.id,
                    file_path=c["file_path"],
                    language=c["language"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    symbol_name=c["symbol_name"],
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    sha_hash=c["sha_hash"]
                )
                chunk_objs.append(chunk_obj)

        db.bulk_save_objects(chunk_objs)
        db.commit()

        # 4. Safely rebuild Chroma collection for this specific repository ONLY
        vector_store = get_vector_store()
        logger.info(f"Deleting stale Chroma collection for repository {repo.id}...")
        vector_store.delete_collection(repo.id)
        logger.info(f"Upserting {len(all_chunks)} fresh chunks to Chroma collection for repository {repo.id}...")
        vector_store.upsert_chunks(repo.id, all_chunks)

        # 5. Update repository stats
        repo.file_count = len(valid_files)
        repo.chunk_count = len(all_chunks)
        db.commit()

        logger.info(f"Successfully reindexed repository {repo.name}: {len(valid_files)} files, {len(all_chunks)} chunks.")
        return {
            "repository_id": repo.id,
            "name": repo.name,
            "files_indexed": len(valid_files),
            "files_purged": len(ignored_file_ids),
            "chunks_indexed": len(all_chunks)
        }
    finally:
        db.close()
