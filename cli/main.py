"""Main CLI entry point."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import typer

from .config import get_github_token, get_openai_api_key, validate_owner_repo, validate_pr_number
from .github import fetch_pr, GitHubAPIError
from .llm import OpenAIProvider, LLMError
from .preprocess import process_diff, make_prompt_input
from .io_safety import read_text_file, write_json_atomic, normalize_path
from .scoring import InvalidResponseError

app = typer.Typer(help="Analyze GitHub PR complexity using LLMs")

# Regex to parse PR URL
_OWNER_REPO_RE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")


def parse_pr_url(url: str) -> Tuple[str, str, int]:
    """Parse owner, repo, and PR number from GitHub PR URL."""
    m = _OWNER_REPO_RE.match(url.strip())
    if not m:
        raise ValueError(f"Invalid PR URL: {url}")
    owner, repo, pr_str = m.group(1), m.group(2), m.group(3)
    return owner, repo, int(pr_str)


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    """Load prompt from file or use default embedded prompt."""
    if prompt_file:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return read_text_file(prompt_file)

    # Load default embedded prompt
    default_prompt_path = Path(__file__).parent / "prompt" / "default.txt"
    if not default_prompt_path.exists():
        raise FileNotFoundError(f"Default prompt not found: {default_prompt_path}")
    return read_text_file(default_prompt_path)


def _analyze_pr_impl(
    pr_url: str,
    prompt_file: Optional[Path] = None,
    model: str = "gpt-5.1",
    format: str = "json",
    output_file: Optional[Path] = None,
    timeout: float = 120.0,
    max_tokens: int = 50000,
    hunks_per_file: int = 2,
    sleep_seconds: float = 0.7,
    dry_run: bool = False,
    post_comment: bool = False,
    openai_api_key: Optional[str] = None,
    github_token: Optional[str] = None,
):
    """
    Analyze a GitHub PR and compute complexity score.

    Environment variables:
    - GH_TOKEN or GITHUB_TOKEN: GitHub API token (optional for public repos)
    - OPENAI_API_KEY: OpenAI API key (required)
    """
    try:
        # Parse PR URL
        owner, repo, pr = parse_pr_url(pr_url)
        validate_owner_repo(owner, repo)
        validate_pr_number(pr)

        # Get credentials (arg takes precedence over env)
        final_github_token = github_token or get_github_token()
        final_openai_key = openai_api_key or get_openai_api_key()

        if not final_openai_key:
            typer.echo(
                "Error: OPENAI_API_KEY environment variable or argument is required", err=True
            )
            typer.echo(
                "Set it with: export OPENAI_API_KEY='your-key' or pass --openai-api-key", err=True
            )
            raise typer.Exit(1)

        # Load prompt
        try:
            prompt_text = load_prompt(prompt_file)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Fetch PR
        typer.echo(f"Fetching PR {owner}/{repo}#{pr}...", err=True)
        try:
            diff_text, meta = fetch_pr(owner, repo, pr, final_github_token, sleep_s=sleep_seconds)
        except GitHubAPIError as e:
            if e.status_code == 404:
                typer.echo("Error: PR not found or not accessible", err=True)
                typer.echo(f"  URL: https://github.com/{owner}/{repo}/pull/{pr}", err=True)
                if not final_github_token:
                    typer.echo(
                        "  Hint: If this is a private repository, set GH_TOKEN or GITHUB_TOKEN",
                        err=True,
                    )
                else:
                    typer.echo(
                        "  Hint: Check that the PR exists and you have access to it", err=True
                    )
            else:
                typer.echo(f"GitHub API error: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Failed to fetch PR: {e}", err=True)
            raise typer.Exit(1)

        title = (meta.get("title") or "").strip()
        typer.echo(f"PR: {title}", err=True)

        # Process diff
        typer.echo("Processing diff...", err=True)
        truncated_diff, stats, selected_files = process_diff(
            diff_text, meta, max_tokens=max_tokens, hunks_per_file=hunks_per_file
        )

        # Format prompt input
        diff_for_prompt = make_prompt_input(pr_url, title, stats, selected_files, truncated_diff)

        if dry_run:
            typer.echo("Dry run: Skipping LLM call", err=True)
            typer.echo(f"Diff excerpt length: {len(truncated_diff)} chars", err=True)
            typer.echo(f"Selected files: {len(selected_files)}", err=True)
            raise typer.Exit(0)

        # Call LLM
        typer.echo("Analyzing complexity with LLM...", err=True)
        try:
            provider = OpenAIProvider(final_openai_key, model=model, timeout=timeout)
            result = provider.analyze_complexity(
                prompt=prompt_text,
                diff_excerpt=diff_for_prompt,
                stats_json=json.dumps(stats),
                title=title,
            )
        except LLMError as e:
            typer.echo(f"LLM error: {e}", err=True)
            raise typer.Exit(1)
        except InvalidResponseError as e:
            typer.echo(f"Invalid LLM response: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Unexpected error: {e}", err=True)
            raise typer.Exit(1)

        # Prepare output
        output = {
            "score": result["complexity"],
            "explanation": result["explanation"],
            "provider": result.get("provider", "openai"),
            "model": result.get("model", model),
            "tokens": result.get("tokens"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "repo": f"{owner}/{repo}",
            "pr": pr,
            "url": pr_url,
            "title": title,
        }

        # Output
        if format == "markdown":
            md = f"""# PR Complexity Analysis

**Score:** {output['score']}/10

**Explanation:** {output['explanation']}

**Details:**
- Repository: {output['repo']}
- PR: #{output['pr']}
- Model: {output['model']}
- Tokens used: {output.get('tokens', 'N/A')}
"""
            typer.echo(md)
        else:
            # JSON output
            json_output = json.dumps(
                {
                    "score": output["score"],
                    "explanation": output["explanation"],
                    "provider": output["provider"],
                    "model": output["model"],
                    "tokens": output.get("tokens"),
                    "timestamp": output["timestamp"],
                },
                ensure_ascii=False,
                indent=2,
            )
            typer.echo(json_output)

        # Set GitHub Action outputs
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                # Score
                f.write(f"score={output['score']}\n")

                # Explanation (handle multiline)
                explanation = output["explanation"]
                if "\n" in explanation:
                    delimiter = "EOF"
                    # Generate random delimiter to avoid conflicts? Simple EOF is usually fine for this content.
                    f.write(f"explanation<<{delimiter}\n{explanation}\n{delimiter}\n")
                else:
                    f.write(f"explanation={explanation}\n")

                # Full JSON output
                full_output_json = json.dumps(output, ensure_ascii=False)
                f.write(f"output={full_output_json}\n")

        # Write to file if requested
        if output_file:
            try:
                # Normalize path for safety
                if output_file.is_absolute():
                    # Allow absolute paths but warn
                    output_path = output_file
                else:
                    # Relative to current directory
                    output_path = normalize_path(Path.cwd(), str(output_file))

                write_json_atomic(output_path, output)
                typer.echo(f"Output written to: {output_path}", err=True)
            except Exception as e:
                typer.echo(f"Warning: Failed to write output file: {e}", err=True)

    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user", err=True)
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback

        if os.getenv("DEBUG"):
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


def get_pr_url_from_context() -> Optional[str]:
    """Get PR URL from GitHub Actions context."""
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        return None

    try:
        with open(event_path, "r") as f:
            event_data = json.load(f)
        return event_data.get("pull_request", {}).get("html_url")
    except (json.JSONDecodeError, IOError):
        return None


@app.command(name="analyze-pr")
def analyze_pr(
    pr_url: Optional[str] = typer.Argument(
        None, help="GitHub PR URL. If not provided, will try to infer from GitHub Actions context."
    ),
    prompt_file: Optional[Path] = typer.Option(
        None, "--prompt-file", "-p", help="Path to custom prompt file (default: embedded prompt)"
    ),
    model: str = typer.Option("gpt-5.1", "--model", "-m", help="OpenAI model name"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or markdown"),
    output_file: Optional[Path] = typer.Option(
        None, "--output-file", "-o", help="Write output to file"
    ),
    timeout: float = typer.Option(120.0, "--timeout", "-t", help="Request timeout in seconds"),
    max_tokens: int = typer.Option(50000, "--max-tokens", help="Maximum tokens for diff excerpt"),
    hunks_per_file: int = typer.Option(2, "--hunks-per-file", help="Maximum hunks per file"),
    sleep_seconds: float = typer.Option(
        0.7, "--sleep-seconds", help="Sleep between GitHub API calls"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch PR but don't call LLM"),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", help="OpenAI API key"),
    github_token: Optional[str] = typer.Option(None, "--github-token", help="GitHub token"),
):
    """Analyze a GitHub PR and compute complexity score."""
    final_pr_url = pr_url
    if not final_pr_url:
        typer.echo(
            "PR URL not provided, attempting to infer from GitHub Actions context...", err=True
        )
        final_pr_url = get_pr_url_from_context()
        if not final_pr_url:
            typer.echo("Error: Could not infer PR URL from context.", err=True)
            typer.echo("Please provide the PR URL as an argument.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Inferred PR URL: {final_pr_url}", err=True)

    _analyze_pr_impl(
        pr_url=final_pr_url,
        prompt_file=prompt_file,
        model=model,
        format=format,
        output_file=output_file,
        timeout=timeout,
        max_tokens=max_tokens,
        hunks_per_file=hunks_per_file,
        sleep_seconds=sleep_seconds,
        dry_run=dry_run,
        openai_api_key=openai_api_key,
        github_token=github_token,
    )


@app.callback(invoke_without_command=True, no_args_is_help=False)
def main(ctx: typer.Context):
    """
    Analyze a GitHub PR and compute complexity score.

    Environment variables:
    - GH_TOKEN or GITHUB_TOKEN: GitHub API token (optional for public repos)
    - OPENAI_API_KEY: OpenAI API key (required)
    """
    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise, handle direct URL invocation
    # Get the first argument from ctx.args
    if not ctx.args or len(ctx.args) == 0:
        typer.echo("Error: PR URL is required", err=True)
        typer.echo("Usage: complexity-cli analyze-pr <PR_URL>", err=True)
        typer.echo("   or: complexity-cli <PR_URL>", err=True)
        raise typer.Exit(1)

    # Check if first arg looks like a URL
    first_arg = ctx.args[0]
    if not _OWNER_REPO_RE.match(first_arg):
        typer.echo(f"Error: Invalid PR URL: {first_arg}", err=True)
        typer.echo("Usage: complexity-cli analyze-pr <PR_URL>", err=True)
        typer.echo("   or: complexity-cli <PR_URL>", err=True)
        raise typer.Exit(1)

    # For direct invocation, use defaults (options require analyze-pr subcommand)
    typer.echo(
        "Note: For options like --model, --format, etc., use 'complexity-cli analyze-pr <URL> [OPTIONS]'",
        err=True,
    )
    _analyze_pr_impl(
        pr_url=first_arg,
        prompt_file=None,
        model="gpt-5.1",
        format="json",
        out=None,
        timeout=120.0,
        max_tokens=50000,
        hunks_per_file=2,
        sleep_seconds=0.7,
        dry_run=False,
    )


if __name__ == "__main__":
    app()
