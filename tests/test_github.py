"""Tests for GitHub module."""
import pytest
from unittest.mock import patch, Mock
from cli.github import fetch_pr_diff, fetch_pr_metadata, GitHubAPIError


@patch("cli.github.httpx.Client")
def test_fetch_pr_diff_success(mock_client_class):
    """Test successful PR diff fetch."""
    mock_response = Mock()
    mock_response.text = "diff content"
    mock_response.raise_for_status = Mock()
    
    mock_client = Mock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = mock_response
    mock_client_class.return_value = mock_client
    
    result = fetch_pr_diff("owner", "repo", 123, token="token")
    assert result == "diff content"


@patch("cli.github.httpx.Client")
def test_fetch_pr_diff_error(mock_client_class):
    """Test PR diff fetch error handling."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    
    mock_client = Mock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_response.raise_for_status.side_effect = Exception("404")
    mock_client.get.return_value = mock_response
    mock_client_class.return_value = mock_client
    
    with pytest.raises(Exception):
        fetch_pr_diff("owner", "repo", 123)


def test_validate_owner_repo():
    """Test owner/repo validation."""
    with pytest.raises(ValueError):
        fetch_pr_diff("owner/repo", "repo", 123)
    with pytest.raises(ValueError):
        fetch_pr_diff("owner", "repo@name", 123)


def test_validate_pr_number():
    """Test PR number validation."""
    with pytest.raises(ValueError):
        fetch_pr_diff("owner", "repo", 0)
    with pytest.raises(ValueError):
        fetch_pr_diff("owner", "repo", -1)

