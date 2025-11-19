"""Configuration parsing and validation."""

import os
import re
from typing import Optional


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment.

    Checks GH_TOKEN first (GitHub CLI convention), then GITHUB_TOKEN.
    """
    return os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


def validate_owner_repo(owner: str, repo: str) -> None:
    """Validate owner and repo names."""
    pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not pattern.match(owner):
        raise ValueError(f"Invalid owner name: {owner}")
    if not pattern.match(repo):
        raise ValueError(f"Invalid repo name: {repo}")


def validate_pr_number(pr: int) -> None:
    """Validate PR number."""
    if pr <= 0:
        raise ValueError(f"PR number must be positive, got: {pr}")


def redact_secret(value: str, visible_chars: int = 4) -> str:
    """Redact a secret value for logging."""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)
