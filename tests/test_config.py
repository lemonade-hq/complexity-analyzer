"""Tests for config module."""
import pytest
from cli.config import validate_owner_repo, validate_pr_number


def test_validate_owner_repo_valid():
    """Test valid owner/repo names."""
    validate_owner_repo("owner", "repo")
    validate_owner_repo("owner-name", "repo_name")
    validate_owner_repo("owner.name", "repo-123")


def test_validate_owner_repo_invalid():
    """Test invalid owner/repo names."""
    with pytest.raises(ValueError):
        validate_owner_repo("owner/repo", "repo")
    with pytest.raises(ValueError):
        validate_owner_repo("owner", "repo@name")


def test_validate_pr_number():
    """Test PR number validation."""
    validate_pr_number(1)
    validate_pr_number(123)
    with pytest.raises(ValueError):
        validate_pr_number(0)
    with pytest.raises(ValueError):
        validate_pr_number(-1)

