"""Centralized error handling for CLI commands."""

from typing import Optional, TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from .github import GitHubAPIError
    from .llm import LLMError


class ErrorHandler:
    """Centralized error handling for CLI commands."""

    @staticmethod
    def handle_github_404(owner: str, repo: str, pr: int, has_token: bool) -> None:
        """
        Handle PR not found errors with helpful hints.

        Args:
            owner: Repository owner
            repo: Repository name
            pr: PR number
            has_token: Whether a GitHub token was provided
        """
        typer.echo("Error: PR not found or not accessible", err=True)
        typer.echo(f"  URL: https://github.com/{owner}/{repo}/pull/{pr}", err=True)
        if not has_token:
            typer.echo(
                "  Hint: If this is a private repository, set GH_TOKEN or GITHUB_TOKEN",
                err=True,
            )
            typer.echo("  Example: export GH_TOKEN='your-token'", err=True)
        else:
            typer.echo(
                "  Hint: Check that the PR exists and you have access to it",
                err=True,
            )

    @staticmethod
    def handle_github_error(error: "GitHubAPIError") -> None:
        """
        Handle general GitHub API errors.

        Args:
            error: The GitHubAPIError that occurred
        """
        if error.status_code == 403:
            typer.echo("Error: GitHub API access forbidden", err=True)
            typer.echo(f"  URL: {error.url}", err=True)
            if "rate limit" in str(error.message).lower():
                typer.echo("  Hint: Rate limit exceeded. Wait or use token rotation.", err=True)
            else:
                typer.echo("  Hint: Check your token has the required permissions", err=True)
        elif error.status_code == 401:
            typer.echo("Error: GitHub authentication failed", err=True)
            typer.echo("  Hint: Check that your token is valid and not expired", err=True)
        elif error.status_code == 422:
            typer.echo("Error: GitHub API validation failed", err=True)
            typer.echo(f"  Details: {error.message}", err=True)
        else:
            typer.echo(f"GitHub API error: {error}", err=True)

    @staticmethod
    def handle_llm_error(error: "LLMError") -> None:
        """
        Handle LLM provider errors.

        Args:
            error: The LLMError that occurred
        """
        error_str = str(error).lower()
        if "rate limit" in error_str or "429" in error_str:
            typer.echo("Error: LLM rate limit exceeded", err=True)
            typer.echo("  Hint: Wait before retrying or use a different API key", err=True)
        elif "authentication" in error_str or "401" in error_str:
            typer.echo("Error: LLM authentication failed", err=True)
            typer.echo("  Hint: Check your OPENAI_API_KEY is valid", err=True)
        elif "timeout" in error_str:
            typer.echo("Error: LLM request timed out", err=True)
            typer.echo("  Hint: Try increasing --timeout or simplify the request", err=True)
        else:
            typer.echo(f"LLM error: {error}", err=True)

    @staticmethod
    def handle_validation_error(message: str) -> None:
        """
        Handle validation errors.

        Args:
            message: The validation error message
        """
        typer.echo(f"Validation error: {message}", err=True)

    @staticmethod
    def handle_file_error(error: Exception, path: Optional[str] = None) -> None:
        """
        Handle file I/O errors.

        Args:
            error: The exception that occurred
            path: Optional path that caused the error
        """
        if path:
            typer.echo(f"File error for '{path}': {error}", err=True)
        else:
            typer.echo(f"File error: {error}", err=True)

    @staticmethod
    def handle_unexpected_error(error: Exception, debug: bool = False) -> None:
        """
        Handle unexpected errors with optional traceback.

        Args:
            error: The exception that occurred
            debug: If True, print full traceback
        """
        typer.echo(f"Unexpected error: {error}", err=True)
        if debug:
            import traceback

            typer.echo(traceback.format_exc(), err=True)


def exit_with_error(message: str, code: int = 1) -> None:
    """
    Print error message and exit with specified code.

    Args:
        message: Error message to display
        code: Exit code (default: 1)

    Raises:
        typer.Exit: Always raises to exit the CLI
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code)
