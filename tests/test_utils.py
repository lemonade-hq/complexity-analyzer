"""Tests for utility functions."""

import pytest

from cli.utils import (
    build_github_diff_headers,
    build_github_headers,
    parse_pr_url,
    redact_token,
    setup_github_tokens,
)
from cli.constants import GITHUB_API_VERSION


class TestParsePrUrl:
    """Tests for parse_pr_url function."""

    def test_valid_url(self):
        """Test parsing valid PR URL."""
        owner, repo, pr = parse_pr_url("https://github.com/owner/repo/pull/123")
        assert owner == "owner"
        assert repo == "repo"
        assert pr == 123

    def test_valid_url_with_trailing_whitespace(self):
        """Test parsing URL with trailing whitespace."""
        owner, repo, pr = parse_pr_url("https://github.com/owner/repo/pull/456  ")
        assert owner == "owner"
        assert repo == "repo"
        assert pr == 456

    def test_valid_url_http(self):
        """Test parsing HTTP URL (not HTTPS)."""
        owner, repo, pr = parse_pr_url("http://github.com/owner/repo/pull/789")
        assert owner == "owner"
        assert repo == "repo"
        assert pr == 789

    def test_valid_url_with_dashes_and_dots(self):
        """Test parsing URL with dashes and dots in owner/repo names."""
        owner, repo, pr = parse_pr_url("https://github.com/my-org/my.repo/pull/100")
        assert owner == "my-org"
        assert repo == "my.repo"
        assert pr == 100

    def test_invalid_url_missing_pull(self):
        """Test that missing 'pull' segment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PR URL"):
            parse_pr_url("https://github.com/owner/repo/issues/123")

    def test_invalid_url_completely_wrong(self):
        """Test that completely wrong URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PR URL"):
            parse_pr_url("https://gitlab.com/owner/repo/pull/123")

    def test_invalid_url_empty(self):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PR URL"):
            parse_pr_url("")

    def test_invalid_url_no_pr_number(self):
        """Test that URL without PR number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PR URL"):
            parse_pr_url("https://github.com/owner/repo/pull/")


class TestBuildGithubHeaders:
    """Tests for build_github_headers function."""

    def test_with_token(self):
        """Test building headers with token."""
        headers = build_github_headers("test-token-123")
        assert headers["Authorization"] == "Bearer test-token-123"
        assert headers["Accept"] == "application/vnd.github+json"
        assert headers["X-GitHub-Api-Version"] == GITHUB_API_VERSION

    def test_without_token(self):
        """Test building headers without token."""
        headers = build_github_headers(None)
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github+json"
        assert headers["X-GitHub-Api-Version"] == GITHUB_API_VERSION

    def test_empty_token(self):
        """Test building headers with empty token."""
        headers = build_github_headers("")
        assert "Authorization" not in headers


class TestBuildGithubDiffHeaders:
    """Tests for build_github_diff_headers function."""

    def test_with_token(self):
        """Test building diff headers with token."""
        headers = build_github_diff_headers("test-token")
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Accept"] == "application/vnd.github.v3.diff"

    def test_without_token(self):
        """Test building diff headers without token."""
        headers = build_github_diff_headers(None)
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3.diff"


class TestRedactToken:
    """Tests for redact_token function."""

    def test_normal_token(self):
        """Test redacting a normal token."""
        result = redact_token("ghp_abc123456789")
        assert result == "ghp_..."
        assert "abc123456789" not in result

    def test_short_token(self):
        """Test redacting a token shorter than visible_chars."""
        result = redact_token("abc")
        assert result == "***"

    def test_exact_visible_chars(self):
        """Test redacting a token equal to visible_chars."""
        result = redact_token("abcd")
        assert result == "****"

    def test_empty_token(self):
        """Test redacting empty token."""
        result = redact_token("")
        assert result == "***"

    def test_custom_visible_chars(self):
        """Test redacting with custom visible_chars."""
        result = redact_token("ghp_abc123456789", visible_chars=8)
        assert result == "ghp_abc1..."


class TestSetupGithubTokens:
    """Tests for setup_github_tokens function."""

    def test_with_cli_tokens(self):
        """Test setup with CLI-provided tokens."""
        token_list, single = setup_github_tokens("token1,token2,token3")
        assert token_list == ["token1", "token2", "token3"]
        assert single == "token1"

    def test_with_cli_tokens_whitespace(self):
        """Test setup with CLI tokens containing whitespace."""
        token_list, single = setup_github_tokens("token1, token2 , token3")
        assert token_list == ["token1", "token2", "token3"]

    def test_empty_cli_tokens_uses_env(self):
        """Test that empty CLI tokens fall back to env getter."""

        def mock_getter():
            return ["env_token1", "env_token2"]

        token_list, single = setup_github_tokens(None, env_tokens_getter=mock_getter)
        assert token_list == ["env_token1", "env_token2"]
        assert single == "env_token1"

    def test_no_tokens(self):
        """Test when no tokens available."""

        def mock_getter():
            return []

        token_list, single = setup_github_tokens(None, env_tokens_getter=mock_getter)
        assert token_list == []
        assert single is None
