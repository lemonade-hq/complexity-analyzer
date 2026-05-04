"""Shared utilities for the CLI."""

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .constants import GITHUB_API_VERSION, TOKEN_VISIBLE_CHARS

logger = logging.getLogger("complexity-cli")

# Regex to parse GitHub PR URL
_OWNER_REPO_RE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")

# Regex to parse GitLab MR URL (any domain with /-/merge_requests/ pattern)
_GITLAB_MR_RE = re.compile(r"https?://([^/\s]+)/(.+?)/-/merge_requests/(\d+)")

# Well-known GitLab domains that are safe to send tokens to.
# Self-hosted domains will trigger a warning log.
_KNOWN_GITLAB_DOMAINS: Set[str] = {"gitlab.com", "gitlab.io"}


def parse_mr_url(url: str) -> Tuple[str, str, int, str, str]:
    """
    Parse a PR/MR URL, auto-detecting GitHub vs GitLab.

    Args:
        url: GitHub PR or GitLab MR URL

    Returns:
        Tuple of (owner_or_project, repo_or_empty, number, provider, base_url)
        - For GitHub: (owner, repo, pr_number, "github", "https://github.com")
        - For GitLab: (project_path, "", mr_iid, "gitlab", "https://gitlab.com")

    Raises:
        ValueError: If URL format is invalid or unrecognized
    """
    url = url.strip()

    # Try GitHub first
    m = _OWNER_REPO_RE.match(url)
    if m:
        owner, repo, pr_str = m.group(1), m.group(2), m.group(3)
        return owner, repo, int(pr_str), "github", "https://github.com"

    # Try GitLab (any non-github domain with /-/merge_requests/)
    m = _GITLAB_MR_RE.match(url)
    if m:
        domain = m.group(1)
        project_path = m.group(2)
        mr_iid = m.group(3)
        # Determine the base URL from the domain
        # Use https by default; check if original URL used http
        scheme = "https"
        if url.startswith("http://"):
            scheme = "http"
        base_url = f"{scheme}://{domain}"

        # Warn about self-hosted domains where tokens will be sent
        if domain not in _KNOWN_GITLAB_DOMAINS:
            logger.warning(
                "GitLab token will be sent to non-standard domain: %s. "
                "Ensure this is a trusted GitLab instance.",
                domain,
            )

        return project_path, "", int(mr_iid), "gitlab", base_url

    raise ValueError(f"Invalid PR URL: {url}")


def parse_pr_url(url: str) -> Tuple[str, str, int]:
    """
    Parse owner, repo, and PR number from GitHub PR URL.

    Args:
        url: GitHub PR URL (e.g., "https://github.com/owner/repo/pull/123")

    Returns:
        Tuple of (owner, repo, pr_number)

    Raises:
        ValueError: If URL format is invalid
    """
    m = _OWNER_REPO_RE.match(url.strip())
    if not m:
        raise ValueError(f"Invalid PR URL: {url}")
    owner, repo, pr_str = m.group(1), m.group(2), m.group(3)
    return owner, repo, int(pr_str)


def build_github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """
    Build headers for GitHub API requests.

    Args:
        token: Optional GitHub token for authentication

    Returns:
        Dict of headers for HTTP requests
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_github_diff_headers(token: Optional[str] = None) -> Dict[str, str]:
    """
    Build headers for GitHub API diff requests.

    Args:
        token: Optional GitHub token for authentication

    Returns:
        Dict of headers for diff requests
    """
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def setup_github_tokens(
    cli_tokens: Optional[str] = None,
    env_tokens_getter: Optional[callable] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Set up GitHub tokens from CLI argument or environment.

    Args:
        cli_tokens: Comma-separated tokens from CLI argument
        env_tokens_getter: Function to get tokens from environment (defaults to config.get_github_tokens)

    Returns:
        Tuple of (token_list, single_token) where single_token is the first token or None
    """
    from .config import get_github_tokens

    getter = env_tokens_getter or get_github_tokens

    token_list: List[str] = []
    if cli_tokens:
        # Parse comma-separated tokens from CLI
        token_list = [t.strip() for t in cli_tokens.split(",") if t.strip()]
    else:
        # Fall back to environment variables
        token_list = getter()

    single_token = token_list[0] if token_list else None
    return token_list, single_token


def redact_token(token: str, visible_chars: int = TOKEN_VISIBLE_CHARS) -> str:
    """
    Redact a token for display, showing only first few characters.

    Args:
        token: The token to redact
        visible_chars: Number of characters to show (default: 4)

    Returns:
        Redacted string like "ghp_..." or "****"
    """
    if not token:
        return "***"
    if len(token) <= visible_chars:
        return "*" * len(token)
    return token[:visible_chars] + "..."


def ssl_verify_enabled() -> bool:
    """Check whether SSL verification is enabled.

    Returns False when the SSL_NO_VERIFY environment variable is set to a
    truthy value (1, true, yes).
    """
    val = os.environ.get("SSL_NO_VERIFY", "").lower()
    return val not in ("1", "true", "yes")
