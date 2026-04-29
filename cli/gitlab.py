"""GitLab API client for fetching MR diffs and metadata."""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from .constants import DEFAULT_TIMEOUT, DEFAULT_SLEEP_SECONDS, GITLAB_DIFFS_PER_PAGE, GITLAB_PER_PAGE
from .github import TokenRotator

# Re-use redact_token from utils
from .utils import redact_token, ssl_verify_enabled

logger = logging.getLogger("complexity-cli")

# Pattern for valid GitLab project paths (alphanumeric, hyphens, underscores, dots, slashes)
_VALID_PROJECT_PATH_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


class GitLabAPIError(Exception):
    """GitLab API error."""

    def __init__(self, status_code: int, message: str, url: str):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"GitLab API error {status_code} for {url}: {message}")


def validate_project_path(project_path: str) -> None:
    """Validate a GitLab project path."""
    if not project_path:
        raise ValueError("GitLab project path cannot be empty")
    if not _VALID_PROJECT_PATH_RE.match(project_path):
        raise ValueError(f"Invalid GitLab project path: {project_path}")


def validate_mr_iid(mr_iid: int) -> None:
    """Validate a GitLab MR IID."""
    if mr_iid <= 0:
        raise ValueError(f"MR IID must be positive, got: {mr_iid}")


def build_gitlab_headers(token: Optional[str] = None) -> Dict[str, str]:
    """
    Build headers for GitLab API requests.

    Args:
        token: Optional GitLab private token for authentication

    Returns:
        Dict of headers for HTTP requests
    """
    headers: Dict[str, str] = {
        "Accept": "application/json",
    }
    if token:
        headers["PRIVATE-TOKEN"] = token
    return headers


def _encode_project_path(project_path: str) -> str:
    """URL-encode a GitLab project path (e.g. 'group/subgroup/repo' -> 'group%2Fsubgroup%2Frepo')."""
    return quote(project_path, safe="")


def _diff_from_gitlab_diffs(diffs: List[Dict[str, Any]]) -> str:
    """
    Reconstruct a unified diff string from GitLab diff entries.

    Each entry from the /diffs endpoint includes a ``diff`` field with
    the unified diff hunks including --- / +++ headers. This function wraps
    each in the standard ``diff --git`` header so the result is parseable
    by ``parse_diff_sections()``.

    Args:
        diffs: List of diff entries from the GitLab diffs API

    Returns:
        Reconstructed unified diff string
    """
    parts: List[str] = []
    for d in diffs:
        patch = d.get("diff")
        if not patch:
            continue
        if d.get("truncated"):
            filename_warn = d.get("new_path") or d.get("old_path", "unknown")
            logger.warning("GitLab diff truncated for file: %s", filename_warn)
        filename = d.get("new_path") or d.get("old_path", "unknown")
        old_path = d.get("old_path", filename)
        # The GitLab diff field already contains --- / +++ lines,
        # so we just prepend the diff --git header
        parts.append(f"diff --git a/{old_path} b/{filename}\n{patch}")
    return "\n".join(parts)


def _fetch_mr_diffs_raw(
    project_path: str,
    mr_iid: int,
    token: Optional[str] = None,
    base_url: str = "https://gitlab.com",
    timeout: float = DEFAULT_TIMEOUT,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch raw diff entries from GitLab /diffs endpoint with pagination.

    Args:
        project_path: Full project path (e.g. 'gitlab-org/gitlab')
        mr_iid: Merge request IID (internal ID)
        token: GitLab private token
        base_url: GitLab instance base URL
        timeout: Request timeout in seconds
        client: Optional httpx.Client to reuse

    Returns:
        List of diff entry dicts from the API
    """
    encoded_path = _encode_project_path(project_path)
    url = f"{base_url}/api/v4/projects/{encoded_path}/merge_requests/{mr_iid}/diffs"
    headers = build_gitlab_headers(token)

    def _do_fetch(c: httpx.Client) -> List[Dict[str, Any]]:
        all_diffs: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = {"per_page": GITLAB_DIFFS_PER_PAGE, "page": page}
            response = c.get(url, headers=headers, params=params)
            response.raise_for_status()
            page_data = response.json()
            if not page_data:
                break
            all_diffs.extend(page_data)
            if len(page_data) < GITLAB_DIFFS_PER_PAGE:
                break
            page += 1
        return all_diffs

    try:
        if client:
            return _do_fetch(client)
        with httpx.Client(timeout=timeout, verify=ssl_verify_enabled()) as c:
            return _do_fetch(c)
    except httpx.HTTPStatusError as e:
        raise GitLabAPIError(
            e.response.status_code,
            e.response.text[:500],
            url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to fetch MR diffs: {e}")


def _fetch_mr_details(
    project_path: str,
    mr_iid: int,
    token: Optional[str] = None,
    base_url: str = "https://gitlab.com",
    timeout: float = DEFAULT_TIMEOUT,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """
    Fetch MR detail metadata from GitLab API.

    Args:
        project_path: Full project path (e.g. 'gitlab-org/gitlab')
        mr_iid: Merge request IID (internal ID)
        token: GitLab private token
        base_url: GitLab instance base URL
        timeout: Request timeout in seconds
        client: Optional httpx.Client to reuse

    Returns:
        Raw MR metadata dict from the API
    """
    encoded_path = _encode_project_path(project_path)
    mr_url = f"{base_url}/api/v4/projects/{encoded_path}/merge_requests/{mr_iid}"
    headers = build_gitlab_headers(token)

    def _do_fetch(c: httpx.Client) -> Dict[str, Any]:
        mr_response = c.get(mr_url, headers=headers)
        mr_response.raise_for_status()
        return mr_response.json()

    try:
        if client:
            return _do_fetch(client)
        with httpx.Client(timeout=timeout, verify=ssl_verify_enabled()) as c:
            return _do_fetch(c)
    except httpx.HTTPStatusError as e:
        raise GitLabAPIError(
            e.response.status_code,
            e.response.text[:500],
            mr_url,
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to fetch MR metadata: {e}")


def _normalize_gitlab_diffs(diffs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize GitLab diff entries to GitHub file objects.

    GitLab diff entries have 'old_path', 'new_path', 'diff' fields.
    GitHub file objects have 'filename', 'patch', 'status', etc.
    """
    files = []
    for d in diffs:
        filename = d.get("new_path") or d.get("old_path", "unknown")
        patch = d.get("diff", "")

        # Map GitLab status to GitHub status
        if d.get("new_file"):
            status = "added"
        elif d.get("deleted_file"):
            status = "removed"
        elif d.get("renamed_file"):
            status = "renamed"
        else:
            status = "modified"

        files.append({
            "filename": filename,
            "patch": patch,
            "status": status,
            "previous_filename": d.get("old_path") if d.get("renamed_file") else None,
        })
    return files


def fetch_mr(
    project_path: str,
    mr_iid: int,
    token: Optional[str] = None,
    base_url: str = "https://gitlab.com",
    sleep_s: float = DEFAULT_SLEEP_SECONDS,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch both MR diff and metadata (diffs fetched once, shared).

    Args:
        project_path: Full project path (e.g. 'gitlab-org/gitlab')
        mr_iid: Merge request IID (internal ID)
        token: GitLab private token
        base_url: GitLab instance base URL
        sleep_s: Sleep between requests in seconds
        progress_callback: Optional callback for progress messages
        timeout: Request timeout in seconds

    Returns:
        Tuple of (diff_text, metadata_dict)
    """
    validate_project_path(project_path)
    validate_mr_iid(mr_iid)

    with httpx.Client(timeout=timeout, verify=ssl_verify_enabled()) as client:
        # Fetch diffs once
        diffs_raw = _fetch_mr_diffs_raw(project_path, mr_iid, token, base_url, timeout, client)
        diff_text = _diff_from_gitlab_diffs(diffs_raw)

        time.sleep(sleep_s)

        # Fetch MR details
        mr_data = _fetch_mr_details(project_path, mr_iid, token, base_url, timeout, client)

    # Normalize files from the already-fetched diffs
    files = _normalize_gitlab_diffs(diffs_raw)

    metadata = {
        "title": mr_data.get("title", ""),
        "html_url": mr_data.get("web_url", ""),
        "number": mr_data.get("iid"),
        "state": mr_data.get("state", ""),
        "user": {"login": mr_data.get("author", {}).get("username", "")},
        "files": files,
    }

    return diff_text, metadata


def _parse_retry_after(error: GitLabAPIError) -> int:
    """Extract Retry-After from a 429 error message, defaulting to 60s."""
    # The error message contains the response text; try to parse Retry-After
    # from the response. Since we don't have access to headers in the error,
    # fall back to 60s.
    return 60


def fetch_mr_with_rotation(
    project_path: str,
    mr_iid: int,
    token_rotator: TokenRotator,
    base_url: str = "https://gitlab.com",
    sleep_s: float = DEFAULT_SLEEP_SECONDS,
    max_retries: int = 10,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch MR diff and metadata with automatic token rotation on rate limits.

    Args:
        project_path: Full project path
        mr_iid: Merge request IID
        token_rotator: TokenRotator instance managing multiple tokens
        base_url: GitLab instance base URL
        sleep_s: Sleep between requests in seconds
        max_retries: Maximum number of retry attempts across all tokens
        progress_callback: Optional callback for progress messages
        timeout: Request timeout in seconds

    Returns:
        Tuple of (diff_text, metadata_dict)

    Raises:
        GitLabAPIError: If all tokens are exhausted and max retries exceeded
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        token = token_rotator.get_token()

        try:
            return fetch_mr(
                project_path, mr_iid, token, base_url, sleep_s, progress_callback, timeout
            )
        except GitLabAPIError as e:
            last_error = e
            if e.status_code == 429:
                # Rate limited — mark token and retry
                wait_seconds = _parse_retry_after(e)
                reset_timestamp = int(time.time()) + wait_seconds
                token_rotator.mark_rate_limited(token, reset_timestamp)
                if progress_callback:
                    redacted = redact_token(token)
                    progress_callback(
                        f"GitLab token {redacted} rate limited (attempt {attempt + 1}/{max_retries}). Rotating..."
                    )

                # Wait for a token to become available
                token_rotator.wait_for_any_available(
                    progress_callback=progress_callback,
                )
                continue
            else:
                # Non-rate-limit error, don't retry
                raise

    # All retries exhausted
    if last_error:
        raise last_error
    raise GitLabAPIError(0, "Max retries exceeded", "")
