import os
import shutil
import tempfile
import git
from typing import Tuple, Optional
import logging

logger = logging.getLogger("repolens.github.repository")

def download_or_clone_repository(clone_url: str, target_dir: Optional[str] = None) -> str:
    """
    Clones a public GitHub repository into a target directory or temporary directory.
    Returns the path to the local directory.
    """
    if not target_dir:
        target_dir = tempfile.mkdtemp(prefix="repolens_repo_")

    try:
        logger.info(f"Cloning repository from {clone_url} into {target_dir}...")
        git.Repo.clone_from(clone_url, target_dir, depth=1)
        logger.info("Clone completed successfully.")
        return target_dir
    except Exception as e:
        logger.error(f"Failed to clone repository {clone_url}: {e}")
        # Clean up directory on failure
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        raise RuntimeError(f"Could not clone repository: {str(e)}")

def cleanup_repository_dir(dir_path: str):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            logger.info(f"Cleaned up directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up directory {dir_path}: {e}")
