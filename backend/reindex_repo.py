import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models.repository import Repository
from app.services.ingestion_service import reindex_existing_repository
from app.ai.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_repo")

def verify_dimensions(db, vector_store):
    repos = db.query(Repository).all()
    target_dim = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))
    provider_name = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "unknown"))
    print(f"\nActive Embedding Provider: {provider_name} (target dimension: {target_dim})\n")
    print(f"{'Repository Name':<24} | {'Repo ID':<38} | {'DB Dim':<6} | {'Chr Dim':<7} | {'DB Model':<22} | {'Chr Model':<22} | {'Status'}")
    print("-" * 145)
    for r in repos:
        col = vector_store.get_or_create_collection(r.id)
        dim = vector_store.get_collection_dimension(col)
        db_dim = getattr(r, "embedding_dimension", None)
        meta = col.metadata or {}
        col_model = meta.get("model")
        db_model = getattr(r, "embedding_model", None)

        is_compat = vector_store.is_collection_compatible(col, repository_id=r.id, db_repo=r)
        if is_compat:
            status = "COMPATIBLE"
        elif (dim is not None and dim != target_dim) or (db_dim is not None and db_dim != target_dim):
            status = "DIM_MISMATCH"
        elif (col_model and col_model != provider_name) or (db_model and db_model != provider_name):
            status = "MODEL_MISMATCH"
        else:
            status = "UNVERIFIED"
        print(
            f"{r.name[:24]:<24} | {r.id:<38} | {str(db_dim):<6} | {str(dim):<7} | "
            f"{str(db_model)[:22]:<22} | {str(col_model)[:22]:<22} | {status}"
        )

def rebuild_incompatible(db, vector_store):
    repos = db.query(Repository).all()
    target_dim = getattr(vector_store, "embedding_dimension", getattr(vector_store.embedding_provider, "dimension", 384))
    active_model = getattr(vector_store, "embedding_model_name", getattr(vector_store.embedding_provider, "model_name", "unknown"))
    for r in repos:
        col = vector_store.get_or_create_collection(r.id)
        if not vector_store.is_collection_compatible(col, repository_id=r.id, db_repo=r):
            dim = vector_store.get_collection_dimension(col)
            meta = col.metadata or {}
            col_model = meta.get("model")
            logger.info(
                f"Rebuilding incompatible collection for {r.name} ({r.id}) "
                f"[Chroma: dim={dim}, model={col_model} | DB: dim={r.embedding_dimension}, model={r.embedding_model} "
                f"-> Target: dim={target_dim}, model={active_model}]..."
            )
            success = vector_store.rebuild_collection_from_db(r.id)
            if success:
                logger.info(f"Successfully migrated {r.name} ({r.id}) to {active_model} ({target_dim}d).")
            else:
                logger.error(f"Failed to migrate {r.name} ({r.id}).")

def main():
    parser = argparse.ArgumentParser(description="Safely reindex a repository collection without touching other repositories.")
    parser.add_argument("repo_id", nargs="?", help="Repository ID to reindex")
    parser.add_argument("--file", help="Reindex only a specific file path within the specified repository ID")
    parser.add_argument("--all", action="store_true", help="Reindex all repositories in DB")
    parser.add_argument("--verify-dimensions", action="store_true", help="Check vector dimension compatibility for all collections")
    parser.add_argument("--rebuild-incompatible", action="store_true", help="Rebuild only collections with incompatible dimensions")
    args = parser.parse_args()

    db = SessionLocal()
    vector_store = get_vector_store()
    try:
        if args.verify_dimensions:
            verify_dimensions(db, vector_store)
        elif args.rebuild_incompatible:
            rebuild_incompatible(db, vector_store)
        elif args.all:
            repos = db.query(Repository).all()
            for r in repos:
                logger.info(f"Reindexing {r.name} ({r.id})...")
                res = reindex_existing_repository(r.id)
                print(res)
        elif args.repo_id:
            if args.file:
                from app.services.ingestion_service import reindex_single_repository_file
                res = reindex_single_repository_file(args.repo_id, args.file)
                print(res)
            else:
                res = reindex_existing_repository(args.repo_id)
                print(res)
        else:
            repos = db.query(Repository).all()
            print("Available repositories in DB:")
            for r in repos:
                print(f"  {r.id}: {r.name} ({r.file_count} files, {r.chunk_count} chunks)")
            print("\nUsage: python reindex_repo.py <repo_id> OR python reindex_repo.py --all OR python reindex_repo.py --verify-dimensions")
    finally:
        db.close()

if __name__ == "__main__":
    main()
