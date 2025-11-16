"""GitHub API client for fetching PR diffs and metadata."""
import json
import time
from typing import Dict, Any, Tuple, Optional
import httpx
from .config import validate_owner_repo, validate_pr_number, redact_secret


class GitHubAPIError(Exception):
    """GitHub API error."""

    def __init__(self, status_code: int, message: str, url: str):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"GitHub API error {status_code} for {url}: {message}")


def fetch_pr_diff(
    owner: str, repo: str, pr: int, token: Optional[str] = None, timeout: float = 60.0
) -> str:
    """
    Fetch PR diff from GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        timeout: Request timeout in seconds
        
    Returns:
        Diff text as string
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}"
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to fetch PR diff: {e}")


def fetch_pr_metadata(
    owner: str, repo: str, pr: int, token: Optional[str] = None, timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Fetch PR metadata and files list from GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        timeout: Request timeout in seconds
        
    Returns:
        Combined metadata dict with 'files' key
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)
    
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Fetch PR metadata
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}"
    # Fetch files list
    files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}/files"
    
    try:
        with httpx.Client(timeout=timeout) as client:
            # Fetch both in parallel
            pr_response = client.get(pr_url, headers=headers)
            pr_response.raise_for_status()
            meta = pr_response.json()
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            
            files_response = client.get(files_url, headers=headers)
            files_response.raise_for_status()
            files = files_response.json()
            
            meta["files"] = files
            return meta
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            pr_url if "pr_response" not in locals() else files_url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to fetch PR metadata: {e}")


def fetch_pr(
    owner: str, repo: str, pr: int, token: Optional[str] = None, sleep_s: float = 0.7
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch both PR diff and metadata with rate limiting.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        sleep_s: Sleep between requests in seconds
        
    Returns:
        Tuple of (diff_text, metadata_dict)
    """
    diff_text = fetch_pr_diff(owner, repo, pr, token)
    time.sleep(sleep_s)
    metadata = fetch_pr_metadata(owner, repo, pr, token)
    return diff_text, metadata

