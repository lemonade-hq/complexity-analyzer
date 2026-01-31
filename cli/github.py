"""GitHub API client for fetching PR diffs and metadata."""

import re
import time
import threading
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Callable
import httpx
from .config import validate_owner_repo, validate_pr_number


class GitHubAPIError(Exception):
    """GitHub API error."""

    def __init__(self, status_code: int, message: str, url: str):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"GitHub API error {status_code} for {url}: {message}")


class TokenRotator:
    """
    Manages a pool of GitHub tokens and rotates between them when rate limits are hit.

    Thread-safe implementation that tracks rate limit status for each token and
    automatically selects the best available token for API requests.

    Usage:
        rotator = TokenRotator(["token1", "token2", "token3"])
        token = rotator.get_token()  # Gets the best available token

        # After a rate limit error:
        rotator.mark_rate_limited(token, reset_timestamp)
        token = rotator.get_token()  # Gets the next available token

        # Update rate limit info from response headers:
        rotator.update_rate_limit(token, remaining=50, reset=1234567890)
    """

    def __init__(self, tokens: List[str], api_type: str = "core"):
        """
        Initialize the token rotator.

        Args:
            tokens: List of GitHub tokens to rotate through
            api_type: API type to track ("core" or "search")
        """
        if not tokens:
            raise ValueError("At least one token is required")

        # Remove duplicates while preserving order
        seen = set()
        self._tokens = []
        for token in tokens:
            if token and token not in seen:
                seen.add(token)
                self._tokens.append(token)

        if not self._tokens:
            raise ValueError("At least one non-empty token is required")

        self._api_type = api_type
        self._lock = threading.Lock()
        self._current_index = 0

        # Track rate limit status for each token
        # {token: {"remaining": int, "reset": int, "rate_limited_until": int}}
        self._token_status: Dict[str, Dict[str, int]] = {
            token: {"remaining": -1, "reset": 0, "rate_limited_until": 0} for token in self._tokens
        }

    @property
    def token_count(self) -> int:
        """Return the number of tokens in the pool."""
        return len(self._tokens)

    def get_token(self) -> str:
        """
        Get the best available token.

        Returns a token that is not rate limited, or if all tokens are rate limited,
        returns the token that will be available soonest.

        Returns:
            The best available GitHub token
        """
        with self._lock:
            current_time = int(time.time())

            # First, try to find a token that is not rate limited
            best_token = None
            best_remaining = -1
            soonest_reset = float("inf")
            soonest_reset_token = self._tokens[0]

            for token in self._tokens:
                status = self._token_status[token]
                rate_limited_until = status.get("rate_limited_until", 0)

                # Check if rate limit has expired
                if rate_limited_until > 0 and current_time >= rate_limited_until:
                    # Rate limit expired, reset the status
                    status["rate_limited_until"] = 0
                    status["remaining"] = -1  # Unknown, will be updated on next request

                # If token is not rate limited
                if status.get("rate_limited_until", 0) <= current_time:
                    remaining = status.get("remaining", -1)

                    # Prefer token with most remaining requests, or unknown (-1)
                    if remaining == -1 or remaining > best_remaining:
                        best_token = token
                        best_remaining = remaining

                # Track which token resets soonest (in case all are rate limited)
                reset_time = status.get("rate_limited_until", 0)
                if reset_time > 0 and reset_time < soonest_reset:
                    soonest_reset = reset_time
                    soonest_reset_token = token

            if best_token:
                return best_token

            # All tokens are rate limited, return the one that resets soonest
            return soonest_reset_token

    def mark_rate_limited(self, token: str, reset_timestamp: int) -> None:
        """
        Mark a token as rate limited until the given reset timestamp.

        Args:
            token: The token that hit a rate limit
            reset_timestamp: Unix timestamp when the rate limit resets
        """
        with self._lock:
            if token in self._token_status:
                self._token_status[token]["rate_limited_until"] = reset_timestamp
                self._token_status[token]["remaining"] = 0

    def update_rate_limit(self, token: str, remaining: int, reset: int) -> None:
        """
        Update rate limit information for a token from response headers.

        Args:
            token: The token used for the request
            remaining: Number of remaining requests (from X-RateLimit-Remaining header)
            reset: Reset timestamp (from X-RateLimit-Reset header)
        """
        with self._lock:
            if token in self._token_status:
                self._token_status[token]["remaining"] = remaining
                self._token_status[token]["reset"] = reset

                # If remaining is 0, mark as rate limited
                if remaining <= 0 and reset > 0:
                    self._token_status[token]["rate_limited_until"] = reset

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all tokens.

        Returns:
            Dict mapping token (redacted) to status info
        """
        with self._lock:
            current_time = int(time.time())
            result = {}
            for i, token in enumerate(self._tokens):
                status = self._token_status[token]
                rate_limited_until = status.get("rate_limited_until", 0)

                # Redact token for display (show first 4 chars only)
                redacted = token[:4] + "..." if len(token) > 4 else "***"

                result[f"token_{i+1} ({redacted})"] = {
                    "remaining": status.get("remaining", -1),
                    "reset": status.get("reset", 0),
                    "rate_limited": rate_limited_until > current_time,
                    "rate_limited_until": (
                        rate_limited_until if rate_limited_until > current_time else None
                    ),
                }
            return result

    def wait_for_any_available(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        max_wait: int = 3600,
    ) -> str:
        """
        Wait until at least one token is available, then return it.

        Args:
            progress_callback: Optional callback for progress messages
            max_wait: Maximum seconds to wait (default: 1 hour)

        Returns:
            An available token

        Raises:
            RuntimeError: If no token becomes available within max_wait
        """
        start_time = time.time()

        while True:
            current_time = int(time.time())

            with self._lock:
                # Find the soonest reset time
                soonest_reset = float("inf")
                any_available = False

                for token in self._tokens:
                    status = self._token_status[token]
                    rate_limited_until = status.get("rate_limited_until", 0)

                    if rate_limited_until <= current_time:
                        any_available = True
                        break

                    if rate_limited_until < soonest_reset:
                        soonest_reset = rate_limited_until

            if any_available:
                return self.get_token()

            # Calculate wait time
            wait_seconds = max(1, int(soonest_reset - current_time) + 1)

            if time.time() - start_time + wait_seconds > max_wait:
                raise RuntimeError(f"All tokens rate limited for more than {max_wait} seconds")

            if progress_callback:
                reset_time = datetime.fromtimestamp(soonest_reset)
                progress_callback(
                    f"All {len(self._tokens)} tokens rate limited. "
                    f"Waiting {wait_seconds}s until reset at {reset_time.strftime('%H:%M:%S')}..."
                )

            time.sleep(min(wait_seconds, 60))  # Sleep in chunks of max 60s


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
        owner,
        repo,
        pr,
        token,
        check_rate_limit_first=check_rate_limit_first,
        progress_callback=progress_callback,
    )
    time.sleep(sleep_s)
    metadata = fetch_pr_metadata(
        owner,
        repo,
        pr,
        token,
        check_rate_limit_first=check_rate_limit_first,
        progress_callback=progress_callback,
    )
    return diff_text, metadata


def fetch_pr_with_rotation(
    owner: str,
    repo: str,
    pr: int,
    token_rotator: TokenRotator,
    sleep_s: float = 0.7,
    max_retries: int = 10,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: float = 60.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch PR diff and metadata with automatic token rotation on rate limits.

    Uses the TokenRotator to automatically switch to another token when a rate
    limit is hit, enabling higher throughput when multiple tokens are available.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token_rotator: TokenRotator instance managing multiple tokens
        sleep_s: Sleep between requests in seconds
        max_retries: Maximum number of retry attempts across all tokens
        progress_callback: Optional callback for progress messages
        timeout: Request timeout in seconds

    Returns:
        Tuple of (diff_text, metadata_dict)

    Raises:
        GitHubAPIError: If all tokens are rate limited and max retries exceeded
        RuntimeError: If request fails for non-rate-limit reasons
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    retry_count = 0

    while retry_count < max_retries:
        # Get the best available token
        token = token_rotator.get_token()

        try:
            # Fetch diff
            diff_text = fetch_pr_diff(
                owner,
                repo,
                pr,
                token,
                check_rate_limit_first=False,  # We handle rate limits via rotation
                progress_callback=progress_callback,
                timeout=timeout,
            )

            time.sleep(sleep_s)

            # Fetch metadata
            metadata = fetch_pr_metadata(
                owner,
                repo,
                pr,
                token,
                check_rate_limit_first=False,
                progress_callback=progress_callback,
                timeout=timeout,
            )

            return diff_text, metadata

        except GitHubAPIError as e:
            # Check if this is a rate limit error
            if e.status_code == 403:
                error_text = str(e.message).lower()
                is_rate_limit = (
                    "rate limit" in error_text
                    or "api rate limit exceeded" in error_text
                    or "secondary rate limit" in error_text
                )

                if is_rate_limit:
                    # Try to extract reset timestamp from error message or use default
                    reset_timestamp = int(time.time()) + 60  # Default: 1 minute

                    # Check if there's a reset time in the response
                    # The actual reset time should come from headers, but we can estimate
                    if "secondary rate limit" in error_text:
                        # Secondary limits are shorter - use exponential backoff
                        reset_timestamp = int(time.time()) + (10 * (2 ** min(retry_count, 4)))
                    else:
                        # Primary rate limit - usually 1 hour window
                        reset_timestamp = int(time.time()) + 60

                    token_rotator.mark_rate_limited(token, reset_timestamp)

                    if progress_callback:
                        redacted = token[:4] + "..." if len(token) > 4 else "***"
                        progress_callback(
                            f"Token {redacted} rate limited for {owner}/{repo}#{pr}. "
                            f"Rotating to next token (attempt {retry_count + 1}/{max_retries})..."
                        )

                    retry_count += 1

                    # If we have multiple tokens, immediately try the next one
                    if token_rotator.token_count > 1:
                        continue

                    # If only one token, wait for it to reset
                    try:
                        token = token_rotator.wait_for_any_available(
                            progress_callback=progress_callback,
                            max_wait=300,  # Max 5 minutes wait
                        )
                        continue
                    except RuntimeError:
                        raise GitHubAPIError(
                            403,
                            f"All tokens rate limited after {retry_count} retries",
                            e.url,
                        )
                else:
                    # Non-rate-limit 403 error
                    raise
            else:
                # Non-403 error, re-raise
                raise

    # Exhausted all retries
    raise GitHubAPIError(
        403,
        f"Rate limit exceeded after {max_retries} retries across all tokens",
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}",
    )


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


def get_pr_labels(
    owner: str,
    repo: str,
    pr: int,
    token: Optional[str] = None,
    timeout: float = 60.0,
) -> List[str]:
    """
    Get all labels on a PR.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        timeout: Request timeout in seconds

    Returns:
        List of label names
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # PRs use the issues endpoint for labels
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr}/labels"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            labels = response.json()
            return [label["name"] for label in labels]
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to get PR labels: {e}")


def add_pr_label(
    owner: str,
    repo: str,
    pr: int,
    label: str,
    token: str,
    timeout: float = 60.0,
) -> None:
    """
    Add a label to a PR.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        label: Label name to add
        token: GitHub token (required - must have write access)
        timeout: Request timeout in seconds
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    if not token:
        raise ValueError("GitHub token is required to add labels")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr}/labels"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json={"labels": [label]})
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to add PR label: {e}")


def remove_pr_label(
    owner: str,
    repo: str,
    pr: int,
    label: str,
    token: str,
    timeout: float = 60.0,
) -> None:
    """
    Remove a label from a PR.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        label: Label name to remove
        token: GitHub token (required - must have write access)
        timeout: Request timeout in seconds
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    if not token:
        raise ValueError("GitHub token is required to remove labels")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    # URL-encode the label name for the path
    import urllib.parse

    encoded_label = urllib.parse.quote(label, safe="")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr}/labels/{encoded_label}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.delete(url, headers=headers)
            # 404 is OK - label might not exist
            if response.status_code == 404:
                return
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return  # Label doesn't exist, that's fine
        raise GitHubAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to remove PR label: {e}")


def update_complexity_label(
    owner: str,
    repo: str,
    pr: int,
    complexity: int,
    token: str,
    label_prefix: str = "complexity:",
    timeout: float = 60.0,
) -> str:
    """
    Update the complexity label on a PR.

    Removes any existing complexity labels and adds a new one with the given score.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        complexity: Complexity score (1-10)
        token: GitHub token (required - must have write access)
        label_prefix: Prefix for complexity labels (default: "complexity:")
        timeout: Request timeout in seconds

    Returns:
        The label name that was applied
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    if not 1 <= complexity <= 10:
        raise ValueError(f"Complexity must be between 1 and 10, got: {complexity}")

    # Get current labels
    current_labels = get_pr_labels(owner, repo, pr, token, timeout)

    # Remove any existing complexity labels
    for label in current_labels:
        if label.startswith(label_prefix):
            remove_pr_label(owner, repo, pr, label, token, timeout)

    # Add new complexity label
    new_label = f"{label_prefix}{complexity}"
    add_pr_label(owner, repo, pr, new_label, token, timeout)

    return new_label


def has_complexity_label(
    owner: str,
    repo: str,
    pr: int,
    token: Optional[str] = None,
    label_prefix: str = "complexity:",
    timeout: float = 60.0,
) -> Optional[str]:
    """
    Check if a PR already has a complexity label.

    Args:
        owner: Repository owner
        repo: Repository name
        pr: PR number
        token: GitHub token (optional for public repos)
        label_prefix: Prefix for complexity labels (default: "complexity:")
        timeout: Request timeout in seconds

    Returns:
        The existing complexity label name if found, None otherwise
    """
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    labels = get_pr_labels(owner, repo, pr, token, timeout)
    for label in labels:
        if label.startswith(label_prefix):
            return label
    return None


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
                            "secondary rate limit" in response_text
                            or "exceeded a secondary rate limit" in response_text
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
                            wait_seconds = 10 * (2**retry_count)
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
                        wait_seconds = 10 * (2**retry_count) if retry_count < 3 else 60
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
            if (
                "only the first 1000 search results" in error_text
                or "1000 search results" in error_text
            ):
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
