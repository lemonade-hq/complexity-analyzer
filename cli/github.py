"""GitHub API client for fetching PR diffs and metadata."""
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Callable
import httpx
from .config import validate_owner_repo, validate_pr_number, redact_secret


class GitHubAPIError(Exception):
    """GitHub API error."""

    def __init__(self, status_code: int, message: str, url: str):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"GitHub API error {status_code} for {url}: {message}")


def wait_for_rate_limit(
    token: Optional[str] = None,
    api_type: str = "core",
    min_remaining: int = 1,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: float = 60.0,
) -> None:
    """
    Check rate limit and wait if necessary before making API requests.
    
    Args:
        token: GitHub token (optional)
        api_type: Type of API to check - "core" or "search" (default: "core")
        min_remaining: Minimum remaining requests required (default: 1)
        progress_callback: Optional callback for progress messages
        timeout: Request timeout in seconds
    """
    try:
        rate_limit_info = check_rate_limit(token=token, timeout=timeout)
        api_info = rate_limit_info.get(api_type, {})
        remaining = api_info.get("remaining", 0)
        reset_timestamp = api_info.get("reset", 0)
        
        if remaining < min_remaining and reset_timestamp > 0:
            current_time = int(time.time())
            wait_seconds = max(0, reset_timestamp - current_time) + 1  # Add 1 second buffer
            
            if wait_seconds > 0:
                reset_time = datetime.fromtimestamp(reset_timestamp)
                msg = (
                    f"Rate limit low ({remaining} remaining for {api_type} API). "
                    f"Waiting {wait_seconds} seconds until reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')}..."
                )
                if progress_callback:
                    progress_callback(msg)
                else:
                    import warnings
                    warnings.warn(msg)
                
                time.sleep(wait_seconds)
    except Exception:
        # If we can't check rate limit, continue anyway (don't block on rate limit check failure)
        pass


def fetch_pr_diff(
    owner: str,
    repo: str,
    pr: int,
    token: Optional[str] = None,
    timeout: float = 60.0,
    check_rate_limit_first: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Fetch PR diff from GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        timeout: Request timeout in seconds
        check_rate_limit_first: If True, check and wait for rate limit before making request
        progress_callback: Optional callback for progress messages
        
    Returns:
        Diff text as string
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)
    
    # Check rate limit before making request
    if check_rate_limit_first:
        wait_for_rate_limit(
            token=token,
            api_type="core",
            min_remaining=1,
            progress_callback=progress_callback,
            timeout=timeout,
        )
    
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
    owner: str,
    repo: str,
    pr: int,
    token: Optional[str] = None,
    timeout: float = 60.0,
    check_rate_limit_first: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Fetch PR metadata and files list from GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        timeout: Request timeout in seconds
        check_rate_limit_first: If True, check and wait for rate limit before making request
        progress_callback: Optional callback for progress messages
        
    Returns:
        Combined metadata dict with 'files' key
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)
    
    # Check rate limit before making request
    if check_rate_limit_first:
        wait_for_rate_limit(
            token=token,
            api_type="core",
            min_remaining=2,  # Need 2 requests (PR metadata + files)
            progress_callback=progress_callback,
            timeout=timeout,
        )
    
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
    owner: str,
    repo: str,
    pr: int,
    token: Optional[str] = None,
    sleep_s: float = 0.7,
    check_rate_limit_first: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch both PR diff and metadata with rate limiting.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        sleep_s: Sleep between requests in seconds
        check_rate_limit_first: If True, check and wait for rate limit before making requests
        progress_callback: Optional callback for progress messages
        
    Returns:
        Tuple of (diff_text, metadata_dict)
    """
    diff_text = fetch_pr_diff(
        owner, repo, pr, token, check_rate_limit_first=check_rate_limit_first, progress_callback=progress_callback
    )
    time.sleep(sleep_s)
    metadata = fetch_pr_metadata(
        owner, repo, pr, token, check_rate_limit_first=check_rate_limit_first, progress_callback=progress_callback
    )
    return diff_text, metadata


def check_rate_limit(
    token: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Check GitHub API rate limit status.
    
    Args:
        token: GitHub token (optional, but authenticated requests have higher limits)
        timeout: Request timeout in seconds
        
    Returns:
        Dict with rate limit information including 'limit', 'remaining', 'reset', and 'used'
        
    Raises:
        GitHubAPIError: If API call fails
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = "https://api.github.com/rate_limit"
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract core rate limit info
            core = data.get("resources", {}).get("core", {})
            search = data.get("resources", {}).get("search", {})
            
            return {
                "core": {
                    "limit": core.get("limit", 0),
                    "remaining": core.get("remaining", 0),
                    "reset": core.get("reset", 0),
                    "used": core.get("used", 0),
                },
                "search": {
                    "limit": search.get("limit", 0),
                    "remaining": search.get("remaining", 0),
                    "reset": search.get("reset", 0),
                    "used": search.get("used", 0),
                },
            }
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to check rate limit: {e}")


def search_closed_prs(
    org: str,
    since: datetime,
    until: datetime,
    token: Optional[str] = None,
    sleep_s: float = 2.0,
    timeout: float = 60.0,
    on_pr_found: Optional[Callable[[str], None]] = None,
    max_retries: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None,
    client: Optional[httpx.Client] = None,
) -> List[str]:
    """
    Search for closed PRs in an organization within a date range.
    
    Uses GitHub Search API to find PRs closed between since and until dates.
    Handles secondary rate limits with exponential backoff.
    
    Args:
        org: Organization name
        token: GitHub token (required for private repos)
        since: Start date (inclusive)
        until: End date (inclusive)
        sleep_s: Sleep between API requests in seconds (default: 2.0 to avoid secondary limits)
        timeout: Request timeout in seconds
        on_pr_found: Optional callback function called for each PR URL as it's found
        max_retries: Maximum number of retries for rate limit errors
        progress_callback: Optional callback for progress messages (e.g., rate limit warnings)
        client: Optional httpx.Client to reuse connections
        
    Returns:
        List of PR URLs (e.g., ["https://github.com/org/repo/pull/123", ...])
        
    Raises:
        GitHubAPIError: If API call fails after retries
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
    # GitHub Search API limit is 100 items per page
    # Total limit is 1000 items (will need to refine search if exceeded)
    per_page = 100
    params = {"q": query, "per_page": per_page, "page": 1}
    
    pr_urls: List[str] = []
    
    should_close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout)
        should_close_client = True
    
    try:
        # Check rate limit before starting search
        wait_for_rate_limit(
            token=token,
            api_type="search",
            min_remaining=1,
            progress_callback=progress_callback,
            timeout=timeout,
        )
        
        while True:
            retry_count = 0
            response = None
            
            # Retry loop for handling rate limits
            while retry_count < max_retries:
                try:
                    response = client.get(url, headers=headers, params=params)
                    
                    # Check for rate limit errors
                    if response.status_code == 403:
                        response_text = response.text.lower()
                        rate_limit_remaining = response.headers.get("X-RateLimit-Remaining", "0")
                        
                        # Check if it's a secondary rate limit
                        is_secondary_limit = (
                            "secondary rate limit" in response_text or
                            "exceeded a secondary rate limit" in response_text
                        )
                        
                        # Primary rate limit (X-RateLimit-Remaining = 0)
                        if rate_limit_remaining == "0" and not is_secondary_limit:
                            reset_time = response.headers.get("X-RateLimit-Reset")
                            if reset_time:
                                try:
                                    reset_timestamp = int(reset_time)
                                    wait_seconds = max(0, reset_timestamp - int(time.time()))
                                    # Add small buffer to ensure reset has occurred
                                    wait_seconds = max(wait_seconds, 1)
                                    msg = f"Primary rate limit exceeded. Waiting {wait_seconds} seconds until reset..."
                                    if progress_callback:
                                        progress_callback(msg)
                                    else:
                                        import warnings
                                        warnings.warn(msg)
                                    time.sleep(wait_seconds + 1)
                                    retry_count += 1
                                    continue
                                except (ValueError, TypeError):
                                    # Fall through to generic handling if header parsing fails
                                    pass
                            # If no reset header, fall through to generic handling
                        
                        # Secondary rate limit - use shorter exponential backoff
                        # GitHub secondary limits typically resolve quickly (seconds to minutes)
                        if is_secondary_limit:
                            # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                            wait_seconds = 10 * (2 ** retry_count)
                            msg = f"Secondary rate limit hit. Waiting {wait_seconds} seconds before retry {retry_count + 1}/{max_retries}..."
                            if progress_callback:
                                progress_callback(msg)
                            else:
                                import warnings
                                warnings.warn(msg)
                            time.sleep(wait_seconds)
                            retry_count += 1
                            continue
                        
                        # If we can't handle it and no more retries, raise error
                        if retry_count >= max_retries - 1:
                            raise GitHubAPIError(
                                403,
                                response.text[:500],
                                url,
                            )
                        
                        # Unknown 403 error - try shorter backoff first
                        # If it's actually a primary limit without headers, fall back to longer wait
                        wait_seconds = 10 * (2 ** retry_count) if retry_count < 3 else 60
                        msg = f"Rate limit error (403). Waiting {wait_seconds} seconds before retry {retry_count + 1}/{max_retries}..."
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            import warnings
                            warnings.warn(msg)
                        time.sleep(wait_seconds)
                        retry_count += 1
                        continue
                    
                    # Success - break out of retry loop
                    break
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 403 and retry_count < max_retries - 1:
                        # Will be handled in the retry loop above
                        continue
                    raise
            
            # If we exhausted retries, raise error
            if retry_count >= max_retries:
                raise GitHubAPIError(
                    403,
                    f"Rate limit exceeded after {max_retries} retries",
                    url,
                )
            
            # Process successful response
            response.raise_for_status()
            data = response.json()
            
            # Check total count on first page to fail fast if we exceed 1000 limit
            # This saves us from fetching 10 pages only to fail at the end
            if params["page"] == 1 and data.get("total_count", 0) > 1000:
                raise GitHubAPIError(
                    422,
                    "1000 search results limit exceeded (total_count > 1000)",
                    url,
                )
            
            items = data.get("items", [])
            for item in items:
                pr_url = item.get("html_url", "")
                if pr_url:
                    pr_urls.append(pr_url)
                    # Call callback immediately if provided
                    if on_pr_found:
                        try:
                            on_pr_found(pr_url)
                        except Exception as e:
                            # Log but don't fail the entire process if callback fails
                            import warnings
                            warnings.warn(f"Callback failed for PR {pr_url}: {e}")
            
            # Check if there are more pages
            if len(items) < per_page:
                break
            
            # GitHub Search API is limited to 1000 results (10 pages of 100)
            if params["page"] >= 10:
                break
            
            params["page"] += 1
            
            # Check rate limit before next request
            wait_for_rate_limit(
                token=token,
                api_type="search",
                min_remaining=1,
                progress_callback=progress_callback,
                timeout=timeout,
            )
            
            # Adaptive sleep based on rate limits
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining and int(remaining) > 10:
                # If we have plenty of quota, sleep less than the default conservative amount
                time.sleep(min(sleep_s, 0.5))
            else:
                time.sleep(sleep_s)
                
    except httpx.HTTPStatusError as e:
        # Check for 422 error about 1000 result limit
        if e.response.status_code == 422:
            error_text = e.response.text.lower()
            if "only the first 1000 search results" in error_text or "1000 search results" in error_text:
                # This is the 1000 result limit - we need to split the date range
                raise GitHubAPIError(
                    422,
                    "GitHub Search API limit: Only first 1000 results available. Date range too large.",
                    url,
                )
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to search PRs: {e}")
    finally:
        if should_close_client:
            client.close()
    
    return pr_urls

