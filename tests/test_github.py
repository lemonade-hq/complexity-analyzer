"""Tests for GitHub module."""

import time
import pytest
from datetime import datetime
from unittest.mock import patch, Mock
from cli.github import (
    fetch_pr_diff,
    search_prs,
    TokenRotator,
    _fetch_all_pr_files,
    _diff_from_files,
)


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
def test_search_prs_success(mock_client_class):
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

    result = search_prs(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        token="token",
    )

    assert len(result) == 2
    assert "https://github.com/org/repo/pull/123" in result
    assert "https://github.com/org/repo/pull/124" in result


@patch("cli.github.wait_for_rate_limit")
@patch("cli.github.httpx.Client")
def test_search_prs_pagination(mock_client_class, mock_wait_rate_limit):
    """Test PR search with pagination."""
    mock_response_page1 = Mock()
    mock_response_page1.json.return_value = {
        "items": [{"html_url": f"https://github.com/org/repo/pull/{i}"} for i in range(100)],
    }
    mock_response_page1.status_code = 200
    mock_response_page1.headers = {"X-RateLimit-Remaining": "100"}
    mock_response_page1.raise_for_status = Mock()

    mock_response_page2 = Mock()
    mock_response_page2.json.return_value = {
        "items": [{"html_url": "https://github.com/org/repo/pull/100"}],
    }
    mock_response_page2.status_code = 200
    mock_response_page2.headers = {"X-RateLimit-Remaining": "99"}
    mock_response_page2.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.get.side_effect = [mock_response_page1, mock_response_page2]
    mock_client_class.return_value = mock_client

    result = search_prs(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        token="token",
    )

    assert len(result) == 101


def test_search_prs_invalid_org():
    """Test PR search with invalid org name."""
    with pytest.raises(ValueError):
        search_prs(
            org="invalid/org",
            since=datetime(2024, 1, 1),
            until=datetime(2024, 1, 31),
        )


# TokenRotator tests


class TestTokenRotator:
    """Tests for the TokenRotator class."""

    def test_init_single_token(self):
        """Test initialization with a single token."""
        rotator = TokenRotator(["token1"])
        assert rotator.token_count == 1
        assert rotator.get_token() == "token1"

    def test_init_multiple_tokens(self):
        """Test initialization with multiple tokens."""
        rotator = TokenRotator(["token1", "token2", "token3"])
        assert rotator.token_count == 3

    def test_init_removes_duplicates(self):
        """Test that duplicate tokens are removed."""
        rotator = TokenRotator(["token1", "token2", "token1", "token3", "token2"])
        assert rotator.token_count == 3

    def test_init_empty_tokens_raises(self):
        """Test that empty token list raises an error."""
        with pytest.raises(ValueError, match="At least one token is required"):
            TokenRotator([])

    def test_init_only_empty_strings_raises(self):
        """Test that list with only empty strings raises an error."""
        with pytest.raises(ValueError, match="At least one non-empty token is required"):
            TokenRotator(["", "", ""])

    def test_init_filters_empty_strings(self):
        """Test that empty strings are filtered out."""
        rotator = TokenRotator(["token1", "", "token2", ""])
        assert rotator.token_count == 2

    def test_get_token_returns_available_token(self):
        """Test that get_token returns an available token."""
        rotator = TokenRotator(["token1", "token2"])
        token = rotator.get_token()
        assert token in ["token1", "token2"]

    def test_mark_rate_limited(self):
        """Test marking a token as rate limited."""
        rotator = TokenRotator(["token1", "token2"])
        reset_time = int(time.time()) + 60  # 1 minute from now

        rotator.mark_rate_limited("token1", reset_time)

        # Should now return token2 since token1 is rate limited
        token = rotator.get_token()
        assert token == "token2"

    def test_rotation_on_rate_limit(self):
        """Test that tokens rotate when one is rate limited."""
        rotator = TokenRotator(["token1", "token2", "token3"])
        reset_time = int(time.time()) + 60

        # Mark first token as rate limited
        rotator.mark_rate_limited("token1", reset_time)
        token = rotator.get_token()
        assert token in ["token2", "token3"]

        # Mark second token as rate limited
        rotator.mark_rate_limited("token2", reset_time)
        token = rotator.get_token()
        assert token == "token3"

    def test_all_tokens_rate_limited_returns_soonest(self):
        """Test behavior when all tokens are rate limited."""
        rotator = TokenRotator(["token1", "token2"])
        current_time = int(time.time())

        # Mark both tokens with different reset times
        rotator.mark_rate_limited("token1", current_time + 120)  # 2 minutes
        rotator.mark_rate_limited("token2", current_time + 60)  # 1 minute

        # Should return the token that resets soonest (token2)
        token = rotator.get_token()
        assert token == "token2"

    def test_update_rate_limit(self):
        """Test updating rate limit info from headers."""
        rotator = TokenRotator(["token1", "token2"])
        reset_time = int(time.time()) + 60

        rotator.update_rate_limit("token1", remaining=0, reset=reset_time)

        # token1 should now be rate limited
        token = rotator.get_token()
        assert token == "token2"

    def test_update_rate_limit_with_remaining(self):
        """Test that tokens with more remaining requests are preferred."""
        rotator = TokenRotator(["token1", "token2"])
        reset_time = int(time.time()) + 60

        # Update rate limits - token2 has more remaining
        rotator.update_rate_limit("token1", remaining=10, reset=reset_time)
        rotator.update_rate_limit("token2", remaining=50, reset=reset_time)

        # Should prefer token2 with more remaining
        token = rotator.get_token()
        assert token == "token2"

    def test_rate_limit_expires(self):
        """Test that expired rate limits are cleared."""
        rotator = TokenRotator(["token1", "token2"])

        # Mark token1 as rate limited with a past reset time
        past_time = int(time.time()) - 10  # 10 seconds ago
        rotator.mark_rate_limited("token1", past_time)

        # token1 should be available again since the rate limit expired
        # The rotator should return either token
        token = rotator.get_token()
        assert token in ["token1", "token2"]

    def test_get_status(self):
        """Test getting status of all tokens."""
        rotator = TokenRotator(["ghp_token1abc", "ghp_token2xyz"])
        reset_time = int(time.time()) + 60

        rotator.update_rate_limit("ghp_token1abc", remaining=50, reset=reset_time)
        rotator.mark_rate_limited("ghp_token2xyz", reset_time)

        status = rotator.get_status()

        # Check that we have 2 tokens in status
        assert len(status) == 2

        # Check that tokens are redacted
        keys = list(status.keys())
        assert "ghp_..." in keys[0]

    def test_thread_safety(self):
        """Test that TokenRotator is thread-safe."""
        import threading

        rotator = TokenRotator(["token1", "token2", "token3"])
        tokens_retrieved = []
        errors = []

        def get_tokens(count):
            try:
                for _ in range(count):
                    token = rotator.get_token()
                    tokens_retrieved.append(token)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=get_tokens, args=(100,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tokens_retrieved) == 1000
        # All tokens should be valid
        for token in tokens_retrieved:
            assert token in ["token1", "token2", "token3"]


# _diff_from_files tests


class TestDiffFromFiles:
    """Tests for the _diff_from_files helper."""

    def test_produces_correct_unified_diff(self):
        """Test that file objects are converted to proper unified diff format."""
        files = [
            {
                "filename": "src/app.py",
                "patch": "@@ -1,3 +1,4 @@\n import os\n+import sys\n \n def main():",
            },
            {
                "filename": "README.md",
                "patch": "@@ -1 +1 @@\n-# Old Title\n+# New Title",
            },
        ]
        result = _diff_from_files(files)

        assert "diff --git a/src/app.py b/src/app.py" in result
        assert "--- a/src/app.py" in result
        assert "+++ b/src/app.py" in result
        assert "+import sys" in result

        assert "diff --git a/README.md b/README.md" in result
        assert "--- a/README.md" in result
        assert "+++ b/README.md" in result
        assert "+# New Title" in result

    def test_skips_files_without_patch(self):
        """Test that binary files (no patch field) are skipped."""
        files = [
            {"filename": "image.png", "status": "added"},
            {
                "filename": "code.py",
                "patch": "@@ -0,0 +1 @@\n+print('hello')",
            },
            {"filename": "binary.bin", "patch": None},
        ]
        result = _diff_from_files(files)

        assert "image.png" not in result
        assert "binary.bin" not in result
        assert "diff --git a/code.py b/code.py" in result

    def test_empty_files_list(self):
        """Test that an empty file list returns an empty string."""
        assert _diff_from_files([]) == ""


# _fetch_all_pr_files tests


class TestFetchAllPrFiles:
    """Tests for the _fetch_all_pr_files paginated helper."""

    @patch("cli.github.httpx.Client")
    def test_single_page(self, mock_client_class):
        """Test fetching files that fit in a single page."""
        files = [{"filename": f"file{i}.py"} for i in range(30)]

        mock_response = Mock()
        mock_response.json.return_value = files
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = _fetch_all_pr_files("owner", "repo", 1, token="tok")
        assert len(result) == 30
        # Only one request needed
        assert mock_client.get.call_count == 1

    @patch("cli.github.time.sleep")
    @patch("cli.github.httpx.Client")
    def test_multi_page_pagination(self, mock_client_class, mock_sleep):
        """Test that multiple pages are fetched until an incomplete page."""
        page1 = [{"filename": f"file{i}.py"} for i in range(100)]
        page2 = [{"filename": f"file{i}.py"} for i in range(100, 150)]

        resp1 = Mock()
        resp1.json.return_value = page1
        resp1.raise_for_status = Mock()

        resp2 = Mock()
        resp2.json.return_value = page2
        resp2.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.side_effect = [resp1, resp2]
        mock_client_class.return_value = mock_client

        result = _fetch_all_pr_files("owner", "repo", 1, token="tok")
        assert len(result) == 150
        assert mock_client.get.call_count == 2

    @patch("cli.github.time.sleep")
    @patch("cli.github.httpx.Client")
    def test_stops_on_empty_page(self, mock_client_class, mock_sleep):
        """Test that pagination stops when an empty page is returned."""
        page1 = [{"filename": f"file{i}.py"} for i in range(100)]
        page2: list = []

        resp1 = Mock()
        resp1.json.return_value = page1
        resp1.raise_for_status = Mock()

        resp2 = Mock()
        resp2.json.return_value = page2
        resp2.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.side_effect = [resp1, resp2]
        mock_client_class.return_value = mock_client

        result = _fetch_all_pr_files("owner", "repo", 1, token="tok")
        assert len(result) == 100


# fetch_pr_diff 406 fallback tests


class TestFetchPrDiff406Fallback:
    """Tests for the 406 fallback in fetch_pr_diff."""

    @patch("cli.github._fetch_all_pr_files")
    @patch("cli.github.httpx.Client")
    def test_falls_back_on_406(self, mock_client_class, mock_fetch_files):
        """Test that a 406 response triggers the files API fallback."""
        import httpx

        # Simulate a 406 HTTPStatusError
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 406
        mock_response.text = "Diff too large"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "406 Not Acceptable",
            request=Mock(),
            response=mock_response,
        )

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Set up the fallback
        mock_fetch_files.return_value = [
            {
                "filename": "big_file.py",
                "patch": "@@ -1,2 +1,3 @@\n import os\n+import sys\n pass",
            },
        ]

        result = fetch_pr_diff("owner", "repo", 42, token="tok")

        mock_fetch_files.assert_called_once_with("owner", "repo", 42, token="tok", timeout=120.0)
        assert "diff --git a/big_file.py b/big_file.py" in result
        assert "+import sys" in result

    @patch("cli.github.httpx.Client")
    def test_non_406_errors_still_raise(self, mock_client_class):
        """Test that non-406 HTTP errors are raised normally."""
        import httpx
        from cli.github import GitHubAPIError

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=Mock(),
            response=mock_response,
        )

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        with pytest.raises(GitHubAPIError) as exc_info:
            fetch_pr_diff("owner", "repo", 42, token="tok")
        assert exc_info.value.status_code == 404
