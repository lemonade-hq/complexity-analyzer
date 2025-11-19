"""GitHub API client for fetching PR diffs and metadata."""
import re
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import httpx
from .config import validate_owner_repo, validate_pr_number


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
def search_closed_prs(
    org: str,
    since: datetime,
    until: datetime,
    token: Optional[str] = None,
    sleep_s: float = 0.7,
    timeout: float = 60.0,
) -> List[str]:
    """
    Search for closed PRs in an organization within a date range.
    
    Uses GitHub Search API to find PRs closed between since and until dates.
    
    Args:
        org: Organization name
        token: GitHub token (required for private repos)
        since: Start date (inclusive)
        until: End date (inclusive)
        sleep_s: Sleep between API requests in seconds
        timeout: Request timeout in seconds
        
    Returns:
        List of PR URLs (e.g., ["https://github.com/org/repo/pull/123", ...])
        
    Raises:
        GitHubAPIError: If API call fails
        ValueError: If org name is invalid
    """
    # Validate org name
    pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not pattern.match(org):
        raise ValueError(f"Invalid organization name: {org}")
    
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Format dates for GitHub search (YYYY-MM-DD)
    since_str = since.strftime("%Y-%m-%d")
    until_str = until.strftime("%Y-%m-%d")
    
    # GitHub search query: org:orgname is:pr is:closed closed:YYYY-MM-DD..YYYY-MM-DD
    query = f"org:{org} is:pr is:closed closed:{since_str}..{until_str}"
    
    url = "https://api.github.com/search/issues"
    params = {"q": query, "per_page": 100, "page": 1}
    
    pr_urls: List[str] = []
    
    try:
        with httpx.Client(timeout=timeout) as client:
            while True:
                response = client.get(url, headers=headers, params=params)
                
                # Check rate limit
                if response.status_code == 403:
                    rate_limit_remaining = response.headers.get("X-RateLimit-Remaining", "0")
                    if rate_limit_remaining == "0":
                        reset_time = response.headers.get("X-RateLimit-Reset", "0")
                        raise GitHubAPIError(
                            403,
                            f"Rate limit exceeded. Reset at: {reset_time}",
                            url,
                        )
                
                response.raise_for_status()
                data = response.json()
                
                items = data.get("items", [])
                for item in items:
                    pr_url = item.get("html_url", "")
                    if pr_url:
                        pr_urls.append(pr_url)
                
                # Check if there are more pages
                if len(items) < params["per_page"]:
                    break
                
                params["page"] += 1
                time.sleep(sleep_s)
                
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to search PRs: {e}")
    
    return pr_urls
