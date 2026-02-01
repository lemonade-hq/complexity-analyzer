"""Configuration parsing and validation."""

import os
import re
from typing import List, Optional

from .constants import TOKEN_VISIBLE_CHARS


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment.

    Checks GH_TOKEN first (GitHub CLI convention), then GITHUB_TOKEN.
    """
    return os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")


def get_github_tokens() -> List[str]:
    """Get multiple GitHub tokens from environment.

    Checks GH_TOKENS first (comma-separated), then falls back to single token.
    Tokens can be comma-separated or newline-separated.

    Environment variables (in order of priority):
    - GH_TOKENS: Comma or newline-separated list of tokens
    - GITHUB_TOKENS: Comma or newline-separated list of tokens
    - GH_TOKEN: Single token (fallback)
    - GITHUB_TOKEN: Single token (fallback)

    Returns:
        List of GitHub tokens (empty list if none found)
    """
    # Check for multiple tokens first
    tokens_str = os.getenv("GH_TOKENS") or os.getenv("GITHUB_TOKENS")
    if tokens_str:
        # Support both comma and newline separators
        tokens = []
        for line in tokens_str.replace("\n", ",").split(","):
            token = line.strip()
            if token:
                tokens.append(token)
        if tokens:
            return tokens

    # Fall back to single token
    single_token = get_github_token()
    if single_token:
        return [single_token]

    return []


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


def redact_secret(value: str, visible_chars: int = TOKEN_VISIBLE_CHARS) -> str:
    """Redact a secret value for logging."""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)
