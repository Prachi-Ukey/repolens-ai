import re
import httpx
from typing import Tuple, Dict, Any, Optional
from app.config import settings
import logging

logger = logging.getLogger("repolens.github.client")

GITHUB_URL_REGEX = re.compile(
    r"^https?://(?:www\.)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)(?:\.git|/)?$"
)

def validate_github_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    return bool(GITHUB_URL_REGEX.match(url.strip()))

def parse_github_url(url: str) -> Tuple[str, str]:
    match = GITHUB_URL_REGEX.match(url.strip())
    if not match:
        raise ValueError(f"Invalid GitHub repository URL: '{url}'")
    owner, repo = match.groups()
    return owner, repo

async def get_github_repo_metadata(owner: str, repo: str) -> Dict[str, Any]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RepoLens-AI-App"
    }
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(api_url, headers=headers)
            if response.status_code == 404:
                raise ValueError(f"GitHub repository '{owner}/{repo}' not found or is private.")
            elif response.status_code == 403:
                # Rate limit exceeded or auth required
                logger.warning("GitHub API rate limit hit, returning fallback metadata")
                return {
                    "owner": owner,
                    "name": repo,
                    "default_branch": "main",
                    "description": f"Public repository {owner}/{repo}",
                    "primary_language": "Unknown"
                }
            response.raise_for_status()
            data = response.json()
            return {
                "owner": data.get("owner", {}).get("login", owner),
                "name": data.get("name", repo),
                "default_branch": data.get("default_branch", "main"),
                "description": data.get("description", ""),
                "primary_language": data.get("language") or "Unknown"
            }
        except httpx.HTTPError as err:
            logger.error(f"HTTP error fetching GitHub repo metadata: {err}")
            # Fallback
            return {
                "owner": owner,
                "name": repo,
                "default_branch": "main",
                "description": f"Repository {owner}/{repo}",
                "primary_language": "Unknown"
            }
