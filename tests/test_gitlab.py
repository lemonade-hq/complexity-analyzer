"""Tests for GitLab module."""

import pytest
from unittest.mock import patch, Mock

from cli.gitlab import (
    GitLabAPIError,
    build_gitlab_headers,
    fetch_mr,
    fetch_mr_with_rotation,
    validate_project_path,
    validate_mr_iid,
    _diff_from_gitlab_diffs,
    _fetch_mr_diffs_raw,
    _normalize_gitlab_diffs,
    _encode_project_path,
)


class TestBuildGitlabHeaders:
    """Tests for build_gitlab_headers function."""

    def test_with_token(self):
        """Test building headers with token."""
        headers = build_gitlab_headers("test-token-123")
        assert headers["PRIVATE-TOKEN"] == "test-token-123"
        assert headers["Accept"] == "application/json"

    def test_without_token(self):
        """Test building headers without token."""
        headers = build_gitlab_headers(None)
        assert "PRIVATE-TOKEN" not in headers
        assert headers["Accept"] == "application/json"

    def test_empty_token(self):
        """Test building headers with empty token."""
        headers = build_gitlab_headers("")
        assert "PRIVATE-TOKEN" not in headers


class TestValidation:
    """Tests for GitLab input validation."""

    def test_valid_project_path(self):
        """Test that valid project paths pass validation."""
        validate_project_path("group/repo")
        validate_project_path("group/subgroup/repo")
        validate_project_path("my-org/my.repo")
        validate_project_path("org_name/repo-name")

    def test_empty_project_path(self):
        """Test that empty project path raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_project_path("")

    def test_invalid_project_path(self):
        """Test that invalid project path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GitLab project path"):
            validate_project_path("path with spaces")
        with pytest.raises(ValueError, match="Invalid GitLab project path"):
            validate_project_path("path@invalid")

    def test_valid_mr_iid(self):
        """Test that valid MR IIDs pass validation."""
        validate_mr_iid(1)
        validate_mr_iid(123)

    def test_invalid_mr_iid(self):
        """Test that invalid MR IIDs raise ValueError."""
        with pytest.raises(ValueError, match="MR IID must be positive"):
            validate_mr_iid(0)
        with pytest.raises(ValueError, match="MR IID must be positive"):
            validate_mr_iid(-1)


class TestEncodeProjectPath:
    """Tests for _encode_project_path."""

    def test_simple_path(self):
        """Test encoding a simple project path."""
        assert _encode_project_path("group/repo") == "group%2Frepo"

    def test_nested_path(self):
        """Test encoding a nested project path."""
        assert _encode_project_path("group/subgroup/repo") == "group%2Fsubgroup%2Frepo"


class TestDiffFromGitlabDiffs:
    """Tests for _diff_from_gitlab_diffs."""

    def test_produces_correct_unified_diff(self):
        """Test that GitLab diff entries are converted to proper unified diff format."""
        diffs = [
            {
                "old_path": "src/app.py",
                "new_path": "src/app.py",
                "diff": "--- a/src/app.py\n+++ b/src/app.py\n@@ -1,3 +1,4 @@\n import os\n+import sys\n \n def main():",
            },
            {
                "old_path": "README.md",
                "new_path": "README.md",
                "diff": "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-# Old Title\n+# New Title",
            },
        ]
        result = _diff_from_gitlab_diffs(diffs)

        assert "diff --git a/src/app.py b/src/app.py" in result
        assert "+import sys" in result
        assert "diff --git a/README.md b/README.md" in result
        assert "+# New Title" in result

    def test_skips_entries_without_diff(self):
        """Test that entries without diff content are skipped."""
        diffs = [
            {"old_path": "binary.bin", "new_path": "binary.bin", "diff": ""},
            {"old_path": "empty.txt", "new_path": "empty.txt", "diff": None},
            {
                "old_path": "code.py",
                "new_path": "code.py",
                "diff": "@@ -0,0 +1 @@\n+print('hello')",
            },
        ]
        result = _diff_from_gitlab_diffs(diffs)

        assert "binary.bin" not in result
        assert "empty.txt" not in result
        assert "diff --git a/code.py b/code.py" in result

    def test_empty_diffs_list(self):
        """Test that an empty diffs list returns an empty string."""
        assert _diff_from_gitlab_diffs([]) == ""

    def test_truncated_diff_logs_warning(self):
        """Test that truncated diffs produce a warning log."""
        diffs = [
            {
                "old_path": "big_file.py",
                "new_path": "big_file.py",
                "diff": "@@ -1 +1 @@\n-old\n+new",
                "truncated": True,
            },
        ]
        with patch("cli.gitlab.logger") as mock_logger:
            _diff_from_gitlab_diffs(diffs)
            mock_logger.warning.assert_called_once()
            assert "truncated" in mock_logger.warning.call_args[0][0].lower()

    def test_non_truncated_diff_no_warning(self):
        """Test that non-truncated diffs don't produce a warning."""
        diffs = [
            {
                "old_path": "file.py",
                "new_path": "file.py",
                "diff": "@@ -1 +1 @@\n-old\n+new",
                "truncated": False,
            },
        ]
        with patch("cli.gitlab.logger") as mock_logger:
            _diff_from_gitlab_diffs(diffs)
            mock_logger.warning.assert_not_called()


class TestNormalizeGitlabDiffs:
    """Tests for _normalize_gitlab_diffs."""

    def test_modified_file(self):
        """Test normalizing a modified file."""
        diffs = [{"new_path": "file.py", "old_path": "file.py", "diff": "patch content"}]
        result = _normalize_gitlab_diffs(diffs)
        assert len(result) == 1
        assert result[0]["filename"] == "file.py"
        assert result[0]["patch"] == "patch content"
        assert result[0]["status"] == "modified"

    def test_added_file(self):
        """Test normalizing an added file."""
        diffs = [{"new_path": "new.py", "old_path": "new.py", "diff": "patch", "new_file": True}]
        result = _normalize_gitlab_diffs(diffs)
        assert result[0]["status"] == "added"

    def test_deleted_file(self):
        """Test normalizing a deleted file."""
        diffs = [
            {"new_path": "old.py", "old_path": "old.py", "diff": "patch", "deleted_file": True}
        ]
        result = _normalize_gitlab_diffs(diffs)
        assert result[0]["status"] == "removed"

    def test_renamed_file(self):
        """Test normalizing a renamed file."""
        diffs = [
            {
                "new_path": "new_name.py",
                "old_path": "old_name.py",
                "diff": "patch",
                "renamed_file": True,
            }
        ]
        result = _normalize_gitlab_diffs(diffs)
        assert result[0]["status"] == "renamed"
        assert result[0]["filename"] == "new_name.py"
        assert result[0]["previous_filename"] == "old_name.py"

    def test_empty_list(self):
        """Test normalizing an empty list."""
        assert _normalize_gitlab_diffs([]) == []


class TestFetchMrDiffsRaw:
    """Tests for _fetch_mr_diffs_raw with pagination."""

    @patch("cli.gitlab.httpx.Client")
    def test_single_page(self, mock_client_class):
        """Test fetching diffs that fit in a single page (less than per-page cap)."""
        # Less than GITLAB_DIFFS_PER_PAGE (20) so pagination stops after page 1
        diffs = [{"new_path": f"file{i}.py", "diff": "content"} for i in range(10)]

        mock_response = Mock()
        mock_response.json.return_value = diffs
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = _fetch_mr_diffs_raw("group/repo", 1, token="token")
        assert len(result) == 10
        assert mock_client.get.call_count == 1

    @patch("cli.gitlab.httpx.Client")
    def test_multi_page_pagination(self, mock_client_class):
        """Test that multiple pages are fetched until an incomplete page."""
        # First page is full (= per-page cap), second page is partial (< cap) so pagination stops
        page1 = [{"new_path": f"file{i}.py", "diff": "content"} for i in range(20)]
        page2 = [{"new_path": f"file{i}.py", "diff": "content"} for i in range(20, 30)]

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

        result = _fetch_mr_diffs_raw("group/repo", 1, token="token")
        assert len(result) == 30
        assert mock_client.get.call_count == 2

    @patch("cli.gitlab.httpx.Client")
    def test_stops_on_empty_page(self, mock_client_class):
        """Test that pagination stops when an empty page is returned."""
        # Full page so the loop continues; empty page on the next call triggers the break
        page1 = [{"new_path": f"file{i}.py", "diff": "content"} for i in range(20)]

        resp1 = Mock()
        resp1.json.return_value = page1
        resp1.raise_for_status = Mock()

        resp2 = Mock()
        resp2.json.return_value = []
        resp2.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get.side_effect = [resp1, resp2]
        mock_client_class.return_value = mock_client

        result = _fetch_mr_diffs_raw("group/repo", 1, token="token")
        assert len(result) == 20

    @patch("cli.gitlab.httpx.Client")
    def test_http_error_raises_gitlab_api_error(self, mock_client_class):
        """Test that HTTP errors are wrapped as GitLabAPIError."""
        import httpx

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

        with pytest.raises(GitLabAPIError) as exc_info:
            _fetch_mr_diffs_raw("group/repo", 1, token="token")
        assert exc_info.value.status_code == 404

    @patch("cli.gitlab.httpx.Client")
    def test_uses_provided_client(self, mock_client_class):
        """Test that a provided client is reused instead of creating a new one."""
        diffs = [{"new_path": "file.py", "diff": "content"}]

        mock_response = Mock()
        mock_response.json.return_value = diffs
        mock_response.raise_for_status = Mock()

        existing_client = Mock()
        existing_client.get.return_value = mock_response

        result = _fetch_mr_diffs_raw("group/repo", 1, token="token", client=existing_client)
        assert len(result) == 1
        # Should NOT have created a new client
        mock_client_class.assert_not_called()
        existing_client.get.assert_called_once()


class TestFetchMr:
    """Tests for fetch_mr (combined diff + metadata, single fetch)."""

    @patch("cli.gitlab.httpx.Client")
    def test_fetches_diffs_once(self, mock_client_class):
        """Test that diffs are fetched only once (not double-fetched)."""
        diffs_response = Mock()
        diffs_response.json.return_value = [
            {
                "old_path": "file.py",
                "new_path": "file.py",
                "diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            }
        ]
        diffs_response.raise_for_status = Mock()

        mr_response = Mock()
        mr_response.json.return_value = {
            "title": "Test MR",
            "web_url": "https://gitlab.com/group/repo/-/merge_requests/1",
            "iid": 1,
            "state": "merged",
            "author": {"username": "testuser"},
        }
        mr_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        # First call: diffs endpoint, second call: MR details endpoint
        mock_client.get.side_effect = [diffs_response, mr_response]
        mock_client_class.return_value = mock_client

        diff_text, metadata = fetch_mr("group/repo", 1, token="token", sleep_s=0)

        # Verify only 2 API calls (diffs + details), not 3 (diffs + details + diffs again)
        assert mock_client.get.call_count == 2

        # Verify diff content
        assert "diff --git" in diff_text
        assert "+new" in diff_text

        # Verify metadata
        assert metadata["title"] == "Test MR"
        assert metadata["state"] == "merged"
        assert metadata["user"]["login"] == "testuser"
        assert len(metadata["files"]) == 1
        assert metadata["files"][0]["filename"] == "file.py"

    def test_validates_project_path(self):
        """Test that fetch_mr validates project path."""
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_mr("", 1, token="token")

    def test_validates_mr_iid(self):
        """Test that fetch_mr validates MR IID."""
        with pytest.raises(ValueError, match="MR IID must be positive"):
            fetch_mr("group/repo", 0, token="token")

    @patch("cli.gitlab.httpx.Client")
    def test_404_raises_gitlab_api_error(self, mock_client_class):
        """Test that 404 responses raise GitLabAPIError."""
        import httpx

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

        with pytest.raises(GitLabAPIError) as exc_info:
            fetch_mr("group/repo", 1, token="token")
        assert exc_info.value.status_code == 404


class TestFetchMrWithRotation:
    """Tests for fetch_mr_with_rotation."""

    @patch("cli.gitlab.fetch_mr")
    def test_success_on_first_attempt(self, mock_fetch_mr):
        """Test successful fetch on first attempt."""
        from cli.github import TokenRotator

        mock_fetch_mr.return_value = ("diff", {"title": "MR"})
        rotator = TokenRotator(["token1", "token2"])

        diff, meta = fetch_mr_with_rotation("group/repo", 1, rotator)
        assert diff == "diff"
        assert meta["title"] == "MR"
        assert mock_fetch_mr.call_count == 1

    @patch("cli.gitlab.fetch_mr")
    def test_rotates_on_rate_limit(self, mock_fetch_mr):
        """Test that tokens are rotated on 429 responses."""
        from cli.github import TokenRotator

        rate_limit_error = GitLabAPIError(429, "Rate limited", "url")
        mock_fetch_mr.side_effect = [
            rate_limit_error,
            ("diff", {"title": "MR"}),
        ]

        rotator = TokenRotator(["token1", "token2"])
        # Mock wait_for_any_available to avoid actual waiting
        rotator.wait_for_any_available = Mock()

        diff, meta = fetch_mr_with_rotation("group/repo", 1, rotator)
        assert diff == "diff"
        assert mock_fetch_mr.call_count == 2

    @patch("cli.gitlab.fetch_mr")
    def test_non_rate_limit_error_not_retried(self, mock_fetch_mr):
        """Test that non-429 errors are raised immediately."""
        from cli.github import TokenRotator

        mock_fetch_mr.side_effect = GitLabAPIError(500, "Server error", "url")
        rotator = TokenRotator(["token1", "token2"])

        with pytest.raises(GitLabAPIError) as exc_info:
            fetch_mr_with_rotation("group/repo", 1, rotator)
        assert exc_info.value.status_code == 500
        assert mock_fetch_mr.call_count == 1

    @patch("cli.gitlab.fetch_mr")
    def test_max_retries_exhausted(self, mock_fetch_mr):
        """Test that max retries are respected."""
        from cli.github import TokenRotator

        mock_fetch_mr.side_effect = GitLabAPIError(429, "Rate limited", "url")
        rotator = TokenRotator(["token1"])
        rotator.wait_for_any_available = Mock()

        with pytest.raises(GitLabAPIError):
            fetch_mr_with_rotation("group/repo", 1, rotator, max_retries=3)
        assert mock_fetch_mr.call_count == 3
