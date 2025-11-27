"""Tests for GitHub module."""

import pytest
from datetime import datetime
from unittest.mock import patch, Mock
from cli.github import fetch_pr_diff, search_closed_prs


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


@patch("cli.github.httpx.Client")
def test_search_closed_prs_success(mock_client_class):
    """Test successful PR search."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "items": [
            {"html_url": "https://github.com/org/repo/pull/123"},
            {"html_url": "https://github.com/org/repo/pull/124"},
        ],
        "total_count": 2,
    }
    mock_response.headers = {"X-RateLimit-Remaining": "100"}
    mock_response.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = search_closed_prs(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        token="token",
    )

    assert len(result) == 2
    assert "https://github.com/org/repo/pull/123" in result
    assert "https://github.com/org/repo/pull/124" in result


@patch("cli.github.httpx.Client")
def test_search_closed_prs_pagination(mock_client_class):
    """Test PR search with pagination."""
    mock_response_page1 = Mock()
    mock_response_page1.json.return_value = {
        "items": [{"html_url": f"https://github.com/org/repo/pull/{i}"} for i in range(100)],
    }
    mock_response_page1.headers = {"X-RateLimit-Remaining": "100"}
    mock_response_page1.raise_for_status = Mock()

    mock_response_page2 = Mock()
    mock_response_page2.json.return_value = {
        "items": [{"html_url": "https://github.com/org/repo/pull/100"}],
    }
    mock_response_page2.headers = {"X-RateLimit-Remaining": "99"}
    mock_response_page2.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.side_effect = [mock_response_page1, mock_response_page2]
    mock_client_class.return_value = mock_client

    result = search_closed_prs(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        token="token",
    )

    assert len(result) == 101


def test_search_closed_prs_invalid_org():
    """Test PR search with invalid org name."""
    with pytest.raises(ValueError):
        search_closed_prs(
            org="invalid/org",
            since=datetime(2024, 1, 1),
            until=datetime(2024, 1, 31),
        )
