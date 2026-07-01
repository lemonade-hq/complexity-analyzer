"""Tests for error handling module."""

from unittest.mock import patch

from cli.errors import ErrorHandler


class TestHandleGitlab404:
    """Tests for handle_gitlab_404."""

    @patch("cli.errors.typer")
    def test_without_token(self, mock_typer):
        """Test 404 error message without token hints to set GITLAB_TOKEN."""
        ErrorHandler.handle_gitlab_404("group/repo", 42, has_token=False)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "not found" in combined.lower() or "not accessible" in combined.lower()
        assert "GL_TOKEN" in combined or "GITLAB_TOKEN" in combined

    @patch("cli.errors.typer")
    def test_with_token(self, mock_typer):
        """Test 404 error message with token hints to check access."""
        ErrorHandler.handle_gitlab_404("group/repo", 42, has_token=True)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "access" in combined.lower()


class TestHandleGitlabError:
    """Tests for handle_gitlab_error."""

    @patch("cli.errors.typer")
    def test_403_error(self, mock_typer):
        """Test 403 forbidden error."""
        from cli.gitlab import GitLabAPIError

        error = GitLabAPIError(403, "Forbidden", "https://gitlab.com/api/v4/...")
        ErrorHandler.handle_gitlab_error(error)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "forbidden" in combined.lower()

    @patch("cli.errors.typer")
    def test_401_error(self, mock_typer):
        """Test 401 authentication error."""
        from cli.gitlab import GitLabAPIError

        error = GitLabAPIError(401, "Unauthorized", "https://gitlab.com/api/v4/...")
        ErrorHandler.handle_gitlab_error(error)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "authentication" in combined.lower()

    @patch("cli.errors.typer")
    def test_429_error(self, mock_typer):
        """Test 429 rate limit error."""
        from cli.gitlab import GitLabAPIError

        error = GitLabAPIError(429, "Too Many Requests", "https://gitlab.com/api/v4/...")
        ErrorHandler.handle_gitlab_error(error)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "rate limit" in combined.lower()

    @patch("cli.errors.typer")
    def test_generic_error(self, mock_typer):
        """Test generic GitLab error."""
        from cli.gitlab import GitLabAPIError

        error = GitLabAPIError(500, "Server Error", "https://gitlab.com/api/v4/...")
        ErrorHandler.handle_gitlab_error(error)
        calls = [str(c) for c in mock_typer.echo.call_args_list]
        combined = " ".join(calls)
        assert "gitlab api error" in combined.lower()
