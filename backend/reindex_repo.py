import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models.repository import Repository
from app.services.ingestion_service import reindex_existing_repository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_repo")

def main():
    parser = argparse.ArgumentParser(description="Safely reindex a repository collection without touching other repositories.")
    parser.add_argument("repo_id", nargs="?", help="Repository ID to reindex")
    parser.add_argument("--all", action="store_true", help="Reindex all repositories in DB")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.all:
            repos = db.query(Repository).all()
            for r in repos:
                logger.info(f"Reindexing {r.name} ({r.id})...")
                res = reindex_existing_repository(r.id)
                print(res)
        elif args.repo_id:
            res = reindex_existing_repository(args.repo_id)
            print(res)
        else:
            repos = db.query(Repository).all()
            print("Available repositories in DB:")
            for r in repos:
                print(f"  {r.id}: {r.name} ({r.file_count} files, {r.chunk_count} chunks)")
            print("\nUsage: python reindex_repo.py <repo_id> OR python reindex_repo.py --all")
    finally:
        db.close()

if __name__ == "__main__":
    main()
