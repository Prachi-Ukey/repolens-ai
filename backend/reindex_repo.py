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
    target_dim = getattr(vector_store.embedding_provider, "dimension", 384)
    provider_name = getattr(vector_store.embedding_provider, "model_name", "unknown")
    print(f"\nActive Embedding Provider: {provider_name} (target dimension: {target_dim})\n")
    print(f"{'Repository Name':<30} | {'Repo ID':<38} | {'Stored Dim':<10} | {'Target Dim':<10} | {'Status'}")
    print("-" * 105)
    for r in repos:
        col = vector_store.get_or_create_collection(r.id)
        dim = vector_store.get_collection_dimension(col)
        status = "COMPATIBLE" if dim == target_dim or dim is None else "MISMATCH"
        print(f"{r.name[:30]:<30} | {r.id:<38} | {str(dim):<10} | {target_dim:<10} | {status}")

def rebuild_incompatible(db, vector_store):
    repos = db.query(Repository).all()
    target_dim = getattr(vector_store.embedding_provider, "dimension", 384)
    for r in repos:
        col = vector_store.get_or_create_collection(r.id)
        dim = vector_store.get_collection_dimension(col)
        if dim is not None and dim != target_dim:
            logger.info(f"Rebuilding incompatible collection for {r.name} ({r.id}) [expected {dim}, target {target_dim}]...")
            vector_store.rebuild_collection_from_db(r.id)

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
