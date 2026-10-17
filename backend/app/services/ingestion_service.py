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

        # Stage 4: Parsing files
        update_job_status(db, job, "parsing_files", 65, f"Processed {len(discovered_files)} code files...")

        # Stage 5: Chunking
        update_job_status(db, job, "chunking", 80, f"Created {len(chunks)} code chunks...")

        # Handle empty repository or repository with zero indexable chunks safely
        if len(chunks) == 0:
            vector_store = get_vector_store()
            try:
                vector_store.delete_collection(repo.id)
            except Exception:
                pass
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).delete()
            db.query(RepositoryFile).filter(RepositoryFile.repository_id == repo.id).delete()
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

            total_discovered_files = len(discovered_files)
            top_files_list = [f["file_path"] for f in discovered_files[:10]]
            repo.embedding_model = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "all-MiniLM-L6-v2"))
            repo.embedding_dimension = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))
            summary = {
                "overview": f"Repository {owner}/{name} contains {total_discovered_files} files and 0 indexable code chunks.",
                "primary_language": repo.primary_language,
                "total_files": total_discovered_files,
                "total_chunks": 0,
                "top_files": top_files_list
            }
            repo.summary_json = json.dumps(summary)
            repo.file_count = total_discovered_files
            repo.chunk_count = 0
            repo.status = "completed"
            update_job_status(db, job, "completed", 100, "Analysis complete. (No indexable code chunks found)", status="completed")
            db.commit()
            return

        # Stage 6: Embedding & Vector Storage
        # Staging and collection replacement are performed BEFORE modifying database records.
        # If vector generation or Chroma indexing fails, existing database files and chunks
        # remain intact, preserving full consistency with the retained vector collection.
        update_job_status(db, job, "embedding", 90, "Generating embeddings and indexing in ChromaDB...")
        vector_store = get_vector_store()

        def on_embedding_progress(processed: int, total: int):
            try:
                pct = 90 + int((processed / max(total, 1)) * 9)
                update_job_status(
                    db,
                    job,
                    "embedding",
                    pct,
                    f"Generating embeddings and indexing in ChromaDB ({processed}/{total})...",
                )
            except Exception as prog_err:
                logger.warning(
                    "Error updating embedding progress for job %s: %s",
                    job_id,
                    prog_err,
                )
                try:
                    db.rollback()
                except Exception as rb_err:
                    logger.warning(
                        "Error rolling back session after failed progress update for job %s: %s",
                        job_id,
                        rb_err,
                    )

        # Safe replacement: Staging and validation before replacing collection
        success = vector_store.safe_replace_collection(
            repo.id,
            chunks,
            progress_callback=on_embedding_progress,
            deferred_cleanup=True,
        )
        if not success:
            raise RuntimeError(f"Failed to generate embeddings and index vector collection for repository {repo.name}.")

        # Vector replacement succeeded: atomically swap previous database records for new records
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
            db.flush() # Populate file_obj.id
            file_obj_map[f_data["file_path"]] = file_obj.id

        # Insert chunks in bounded batches to avoid allocating all ORM instances at once
        CHUNK_SAVE_BATCH_SIZE = 100
        for i in range(0, len(chunks), CHUNK_SAVE_BATCH_SIZE):
            batch_slice = chunks[i : i + CHUNK_SAVE_BATCH_SIZE]
            chunk_objs = []
            for c in batch_slice:
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
                        sha_hash=c.get("sha_hash")
                    )
                    chunk_objs.append(chunk_obj)
            if chunk_objs:
                db.bulk_save_objects(chunk_objs)
                del chunk_objs

        file_obj_map.clear()
        del file_obj_map

        total_discovered_files = len(discovered_files)
        top_files_list = [f["file_path"] for f in discovered_files[:10]]

        repo.embedding_model = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "all-MiniLM-L6-v2"))
        repo.embedding_dimension = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))

        # Generate overview summary
        summary = {
            "overview": f"Repository {owner}/{name} contains {total_discovered_files} source files and {len(chunks)} code chunks.",
            "primary_language": repo.primary_language,
            "total_files": total_discovered_files,
            "total_chunks": len(chunks),
            "top_files": top_files_list
        }
        repo.summary_json = json.dumps(summary)
        repo.file_count = total_discovered_files
        repo.chunk_count = len(chunks)
        repo.status = "completed"

        update_job_status(db, job, "completed", 100, "Analysis complete!", status="completed")
        db.commit()
        vector_store.finalize_replacement(repo.id)

    except Exception as e:
        logger.error(f"Ingestion pipeline error: {e}", exc_info=True)
        if 'vector_store' in locals() and vector_store and 'repo' in locals() and repo:
            if getattr(vector_store, "has_pending_replacement", None) and vector_store.has_pending_replacement(repo.id):
                recovery_ok = vector_store.rollback_replacement(repo.id)
                if not recovery_ok:
                    logger.critical(
                        "CRITICAL: Failed to rollback vector collection for repo '%s' after ingestion failure.",
                        repo.id,
                    )
        if 'db' in locals():
            try:
                db.rollback()
            except Exception:
                pass
            try:
                j = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
                if j:
                    j.status = "failed"
                    j.error_details = str(e)
                    j.message = f"Analysis failed: {str(e)}"
                r = db.query(Repository).filter(Repository.id == repository_id).first()
                if r:
                    r.status = "failed"
                db.commit()
            except Exception as commit_err:
                logger.error(f"Failed to record failure status in db: {commit_err}")
    finally:
        if temp_dir:
            cleanup_repository_dir(temp_dir)
        import gc
        gc.collect()
        db.close()


def reconcile_abandoned_jobs(db: Session, max_stale_seconds: int = 300) -> int:
    """
    Recovers analysis jobs left in 'processing' or 'pending' state across server restarts or crashes.
    To avoid interfering with any concurrently running workers, only marks jobs as failed
    if their last heartbeat (updated_at) was older than max_stale_seconds (default 5 minutes).
    """
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_stale_seconds)
    cutoff_naive = cutoff.replace(tzinfo=None)

    stale_jobs = db.query(AnalysisJob).filter(
        AnalysisJob.status.in_(["processing", "pending"]),
        AnalysisJob.updated_at < cutoff_naive
    ).all()

    recovered_count = len(stale_jobs)
    for job in stale_jobs:
        job.status = "failed"
        job.current_stage = "failed"
        job.message = "Analysis was interrupted due to a server restart or timeout. Please re-analyze the repository."
        job.error_details = "Job heartbeat timed out (server restart or memory limit exceeded)."

        repo = db.query(Repository).filter(Repository.id == job.repository_id).first()
        if repo and repo.status in ("analyzing", "pending"):
            repo.status = "failed"

    if recovered_count > 0:
        db.commit()
        logger.warning(
            "Reconciled %s abandoned analysis job(s) from previous server sessions.",
            recovered_count,
        )

    return recovered_count


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

        # 2. Check if local source files are available on disk and prepare refreshed content in memory
        refreshed_data = {}
        for f in valid_files:
            norm_p = f.file_path.replace("\\", "/").strip().lstrip("./")
            candidates = [
                norm_p,
                os.path.join("..", norm_p),
                os.path.join(".", norm_p),
                os.path.abspath(norm_p),
                os.path.abspath(os.path.join("..", norm_p)),
            ]
            if norm_p.startswith("backend/"):
                sub_p = norm_p[len("backend/"):]
                candidates.extend([sub_p, os.path.join(".", sub_p), os.path.abspath(sub_p)])

            disk_content = None
            for cand in candidates:
                if os.path.isfile(cand):
                    try:
                        with open(cand, "r", encoding="utf-8", errors="replace") as fp_in:
                            disk_content = fp_in.read()
                        break
                    except Exception:
                        pass
            if disk_content is not None and disk_content != f.content:
                logger.info(f"Prepared refreshed content for {f.file_path} from disk ({len((f.content or '').splitlines())} -> {len(disk_content.splitlines())} lines)")
                refreshed_data[f.id] = {
                    "content": disk_content,
                    "line_count": len(disk_content.splitlines()),
                    "size_bytes": len(disk_content.encode("utf-8")),
                }

        # 3. Parse valid files into chunks in memory
        all_chunks = []
        for f in valid_files:
            content_to_use = refreshed_data[f.id]["content"] if f.id in refreshed_data else f.content
            file_chunks = parse_file_into_chunks(
                file_path=f.file_path,
                content=content_to_use or "",
                language=f.language
            )
            for c in file_chunks:
                c_with_meta = dict(c)
                c_with_meta["file_id"] = f.id
                all_chunks.append(c_with_meta)

        # 4. Handle empty repository or zero indexable chunks safely
        vector_store = get_vector_store()
        if len(all_chunks) == 0:
            try:
                vector_store.delete_collection(repo.id)
            except Exception:
                pass
            if ignored_file_ids:
                db.query(CodeChunk).filter(CodeChunk.file_id.in_(ignored_file_ids)).delete(synchronize_session=False)
                db.query(RepositoryFile).filter(RepositoryFile.id.in_(ignored_file_ids)).delete(synchronize_session=False)
            db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).delete()
            for f in valid_files:
                if f.id in refreshed_data:
                    f.content = refreshed_data[f.id]["content"]
                    f.line_count = refreshed_data[f.id]["line_count"]
                    f.size_bytes = refreshed_data[f.id]["size_bytes"]
            repo.embedding_model = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "all-MiniLM-L6-v2"))
            repo.embedding_dimension = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))
            repo.file_count = len(valid_files)
            repo.chunk_count = 0
            db.commit()
            return {
                "repository_id": repo.id,
                "name": repo.name,
                "files_indexed": len(valid_files),
                "files_purged": len(ignored_file_ids),
                "chunks_indexed": 0
            }

        # 5. Safely rebuild Chroma collection BEFORE touching database records
        # If vector generation or staging fails, previous database records remain intact.
        logger.info(f"Safely replacing Chroma collection for repository {repo.id} with {len(all_chunks)} chunks...")
        success = vector_store.safe_replace_collection(repo.id, all_chunks, deferred_cleanup=True)
        if not success:
            raise RuntimeError(f"Failed to safely reindex Chroma collection for repository '{repo.id}'.")

        # 6. Vector replacement succeeded: atomically swap previous database records for new records
        if ignored_file_ids:
            db.query(CodeChunk).filter(CodeChunk.file_id.in_(ignored_file_ids)).delete(synchronize_session=False)
            db.query(RepositoryFile).filter(RepositoryFile.id.in_(ignored_file_ids)).delete(synchronize_session=False)

        for f in valid_files:
            if f.id in refreshed_data:
                f.content = refreshed_data[f.id]["content"]
                f.line_count = refreshed_data[f.id]["line_count"]
                f.size_bytes = refreshed_data[f.id]["size_bytes"]

        db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).delete()

        CHUNK_SAVE_BATCH_SIZE = 100
        for i in range(0, len(all_chunks), CHUNK_SAVE_BATCH_SIZE):
            batch_slice = all_chunks[i : i + CHUNK_SAVE_BATCH_SIZE]
            chunk_objs = [
                CodeChunk(
                    repository_id=repo.id,
                    file_id=c["file_id"],
                    file_path=c["file_path"],
                    language=c["language"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    symbol_name=c["symbol_name"],
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    sha_hash=c.get("sha_hash")
                )
                for c in batch_slice
            ]
            if chunk_objs:
                db.bulk_save_objects(chunk_objs)
                del chunk_objs

        # Update repository stats and embedding metadata
        repo.embedding_model = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "all-MiniLM-L6-v2"))
        repo.embedding_dimension = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))
        repo.file_count = len(valid_files)
        repo.chunk_count = len(all_chunks)
        db.commit()

        # 7. Database transaction succeeded: explicitly finalize vector replacement
        vector_store.finalize_replacement(repo.id)

        logger.info(f"Successfully reindexed repository {repo.name}: {len(valid_files)} files, {len(all_chunks)} chunks.")
        return {
            "repository_id": repo.id,
            "name": repo.name,
            "files_indexed": len(valid_files),
            "files_purged": len(ignored_file_ids),
            "chunks_indexed": len(all_chunks)
        }
    except Exception as e:
        logger.error(
            "Re-indexing failed during database update for repository '%s': %s. Rolling back DB and restoring vector collection...",
            repository_id,
            e,
        )
        if 'db' in locals():
            try:
                db.rollback()
            except Exception:
                pass

        if 'vector_store' in locals() and vector_store:
            if getattr(vector_store, "has_pending_replacement", None) and vector_store.has_pending_replacement(repository_id):
                recovery_ok = vector_store.rollback_replacement(repository_id)
                if not recovery_ok:
                    logger.critical(
                        "CRITICAL: Vector collection recovery failed for repository '%s' after database update failure! "
                        "The vector store and database are in an inconsistent state.",
                        repository_id,
                    )
                    raise RuntimeError(
                        f"Re-indexing failed during database update ({e}) AND vector store recovery failed. Data is inconsistent."
                    ) from e
                logger.info(
                    "Successfully rolled back vector collection for repository '%s' to match restored database records.",
                    repository_id,
                )
        raise
    finally:
        # Trigger garbage collection of unreferenced Python objects; note that OS/glibc allocator may retain heap arenas.
        import gc
        gc.collect()
        db.close()


def reindex_single_repository_file(repository_id: str, file_path: str) -> Dict[str, Any]:
    """
    Safely reindexes a single file within a repository without touching any other files,
    collections, conversations, or repositories.
    """
    from app.github.parser import parse_file_into_chunks

    db = SessionLocal()
    try:
        repo = db.query(Repository).filter(Repository.id == repository_id).first()
        if not repo:
            raise ValueError(f"Repository '{repository_id}' not found.")

        clean_fp = file_path.replace("\\", "/").strip().lstrip("./")
        rf = db.query(RepositoryFile).filter(
            RepositoryFile.repository_id == repo.id,
            RepositoryFile.file_path.ilike(f"%{clean_fp}")
        ).first()
        if not rf:
            raise ValueError(f"File '{file_path}' not found in repository '{repository_id}'.")

        # Refresh from disk if available
        candidates = [
            clean_fp,
            os.path.join("..", clean_fp),
            os.path.join(".", clean_fp),
            os.path.abspath(clean_fp),
            os.path.abspath(os.path.join("..", clean_fp)),
        ]
        if clean_fp.startswith("backend/"):
            sub_p = clean_fp[len("backend/"):]
            candidates.extend([sub_p, os.path.join(".", sub_p), os.path.abspath(sub_p)])

        disk_content = None
        for cand in candidates:
            if os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8", errors="replace") as fp_in:
                        disk_content = fp_in.read()
                    break
                except Exception:
                    pass

        if disk_content is not None:
            rf.content = disk_content
            rf.line_count = len(disk_content.splitlines())
            rf.size_bytes = len(disk_content.encode("utf-8"))

        fresh_chunks = parse_file_into_chunks(
            file_path=rf.file_path,
            content=rf.content,
            language=rf.language or "python"
        )

        db.query(CodeChunk).filter(
            CodeChunk.repository_id == repo.id,
            CodeChunk.file_id == rf.id
        ).delete()

        chunk_objs = []
        for c in fresh_chunks:
            c_obj = CodeChunk(
                repository_id=repo.id,
                file_id=rf.id,
                file_path=c["file_path"],
                language=c["language"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                symbol_name=c["symbol_name"],
                chunk_id=c["chunk_id"],
                content=c["content"],
                sha_hash=c["sha_hash"]
            )
            chunk_objs.append(c_obj)
        db.bulk_save_objects(chunk_objs)
        db.commit()

        # Update Chroma collection with fresh chunks for this file
        vector_store = get_vector_store()
        try:
            col = vector_store.get_or_create_collection(repo.id)
            col.delete(where={"file_path": rf.file_path})
        except Exception as e_del:
            logger.warning(f"Could not delete old Chroma chunks for {rf.file_path}: {e_del}")
        vector_store.upsert_chunks(repo.id, fresh_chunks)

        total_chunks = db.query(CodeChunk).filter(CodeChunk.repository_id == repo.id).count()
        repo.chunk_count = total_chunks
        db.commit()

        return {
            "repository_id": repo.id,
            "file_path": rf.file_path,
            "chunks_indexed": len(fresh_chunks),
            "line_count": rf.line_count
        }
    finally:
        db.close()
