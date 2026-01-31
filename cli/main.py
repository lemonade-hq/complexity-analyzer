"""Main CLI entry point."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable

from dotenv import load_dotenv
import typer

# Load environment variables from .env file
load_dotenv()

from .config import (
    get_github_token,
    get_github_tokens,
    get_openai_api_key,
    validate_owner_repo,
    validate_pr_number,
)  # noqa: E402
from .github import (
    fetch_pr,
    fetch_pr_with_rotation,
    GitHubAPIError,
    check_rate_limit,
    update_complexity_label,
    TokenRotator,
)  # noqa: E402
from .llm import OpenAIProvider, LLMError  # noqa: E402
from .preprocess import process_diff, make_prompt_input  # noqa: E402
from .io_safety import read_text_file, write_json_atomic, normalize_path  # noqa: E402
from .scoring import InvalidResponseError  # noqa: E402
from .batch import (  # noqa: E402
    load_pr_urls_from_file,
    generate_pr_list_from_date_range,
)

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


def analyze_pr_to_dict(
    pr_url: str,
    prompt_text: str,
    github_token: Optional[str],
    openai_key: str,
    model: str = "gpt-5.2",
    timeout: float = 120.0,
    max_tokens: int = 50000,
    hunks_per_file: int = 2,
    sleep_seconds: float = 0.7,
    progress_callback: Optional[Callable[[str], None]] = None,
    token_rotator: Optional[TokenRotator] = None,
) -> dict:
    """
    Analyze a GitHub PR and return result as dictionary.

    This is the core analysis function that can be reused by both single PR
    and batch analysis workflows.

    Args:
        pr_url: GitHub PR URL
        prompt_text: Prompt text for LLM
        github_token: GitHub API token (optional, ignored if token_rotator is provided)
        openai_key: OpenAI API key (required)
        model: OpenAI model name
        timeout: Request timeout in seconds
        max_tokens: Maximum tokens for diff excerpt
        hunks_per_file: Maximum hunks per file
        sleep_seconds: Sleep between GitHub API calls
        progress_callback: Optional callback for progress messages (e.g., rate limit warnings)
        token_rotator: Optional TokenRotator for automatic token rotation on rate limits

    Returns:
        Dict with keys: score, explanation, provider, model, tokens, timestamp,
        repo, pr, url, title

    Raises:
        ValueError: If PR URL is invalid
        GitHubAPIError: If GitHub API call fails
        LLMError: If LLM call fails
        InvalidResponseError: If LLM response is invalid
    """
    # Parse PR URL
    owner, repo, pr = parse_pr_url(pr_url)
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    # Fetch PR - use token rotator if available, otherwise use single token
    if token_rotator:
        diff_text, meta = fetch_pr_with_rotation(
            owner,
            repo,
            pr,
            token_rotator,
            sleep_s=sleep_seconds,
            progress_callback=progress_callback,
            timeout=timeout,
        )
    else:
        diff_text, meta = fetch_pr(
            owner,
            repo,
            pr,
            github_token,
            sleep_s=sleep_seconds,
            progress_callback=progress_callback,
        )

    title = (meta.get("title") or "").strip()

    # Process diff
    truncated_diff, stats, selected_files = process_diff(
        diff_text, meta, max_tokens=max_tokens, hunks_per_file=hunks_per_file
    )

    # Format prompt input
    diff_for_prompt = make_prompt_input(pr_url, title, stats, selected_files, truncated_diff)

    # Call LLM
    provider = OpenAIProvider(openai_key, model=model, timeout=timeout)
    result = provider.analyze_complexity(
        prompt=prompt_text,
        diff_excerpt=diff_for_prompt,
        stats_json=json.dumps(stats),
        title=title,
    )

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

    return output


def _analyze_pr_impl(
    pr_url: str,
    prompt_file: Optional[Path] = None,
    model: str = "gpt-5.2",
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
        # Parse PR URL for display
        owner, repo, pr = parse_pr_url(pr_url)

        # Handle dry run
        if dry_run:
            typer.echo(f"Fetching PR {owner}/{repo}#{pr}...", err=True)
            try:
                diff_text, meta = fetch_pr(
                    owner, repo, pr, final_github_token, sleep_s=sleep_seconds
                )
                title = (meta.get("title") or "").strip()
                typer.echo(f"PR: {title}", err=True)
                typer.echo("Processing diff...", err=True)
                truncated_diff, stats, selected_files = process_diff(
                    diff_text, meta, max_tokens=max_tokens, hunks_per_file=hunks_per_file
                )
                typer.echo("Dry run: Skipping LLM call", err=True)
                typer.echo(f"Diff excerpt length: {len(truncated_diff)} chars", err=True)
                typer.echo(f"Selected files: {len(selected_files)}", err=True)
                raise typer.Exit(0)
            except GitHubAPIError as e:
                if e.status_code == 404:
                    typer.echo("Error: PR not found or not accessible", err=True)
                    typer.echo(f"  URL: https://github.com/{owner}/{repo}/pull/{pr}", err=True)
                    if not final_github_token:
                        typer.echo(
                            "  Hint: If this is a private repository, set GH_TOKEN or GITHUB_TOKEN environment variable",
                            err=True,
                        )
                        typer.echo("  Example: export GH_TOKEN='your-token'", err=True)
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

        # Analyze PR
        typer.echo(f"Fetching PR {owner}/{repo}#{pr}...", err=True)
        try:
            output = analyze_pr_to_dict(
                pr_url=pr_url,
                prompt_text=prompt_text,
                github_token=final_github_token,
                openai_key=final_openai_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                hunks_per_file=hunks_per_file,
                sleep_seconds=sleep_seconds,
            )
            typer.echo(f"PR: {output['title']}", err=True)
            typer.echo("Processing diff...", err=True)
            typer.echo("Analyzing complexity with LLM...", err=True)
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
        except LLMError as e:
            typer.echo(f"LLM error: {e}", err=True)
            raise typer.Exit(1)
        except InvalidResponseError as e:
            typer.echo(f"Invalid LLM response: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Unexpected error: {e}", err=True)
            raise typer.Exit(1)

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

                # Model used
                f.write(f"model={output.get('model', 'unknown')}\n")

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
    model: str = typer.Option("gpt-5.2", "--model", "-m", help="OpenAI model name"),
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


@app.command(name="batch-analyze")
def batch_analyze(
    input_file: Optional[Path] = typer.Option(
        None, "--input-file", "-i", help="File containing PR URLs (one per line)"
    ),
    org: Optional[str] = typer.Option(
        None, "--org", help="Organization name (for date range search)"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Start date (YYYY-MM-DD) for date range search"
    ),
    until: Optional[str] = typer.Option(
        None, "--until", help="End date (YYYY-MM-DD) for date range search"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output CSV file path (required unless --label is used)"
    ),
    cache_file: Optional[Path] = typer.Option(
        None, "--cache", help="Cache file for PR list (used with date range)"
    ),
    prompt_file: Optional[Path] = typer.Option(
        None, "--prompt-file", "-p", help="Path to custom prompt file (default: embedded prompt)"
    ),
    model: str = typer.Option("gpt-5.2", "--model", "-m", help="OpenAI model name"),
    timeout: float = typer.Option(120.0, "--timeout", "-t", help="Request timeout in seconds"),
    max_tokens: int = typer.Option(50000, "--max-tokens", help="Maximum tokens for diff excerpt"),
    hunks_per_file: int = typer.Option(2, "--hunks-per-file", help="Maximum hunks per file"),
    sleep_seconds: float = typer.Option(
        0.7, "--sleep-seconds", help="Sleep between GitHub API calls"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from existing output file"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of parallel workers (default: 1 = sequential)"
    ),
    label: bool = typer.Option(
        False, "--label", "-l", help="Label PRs with complexity instead of CSV output"
    ),
    label_prefix: str = typer.Option(
        "complexity:", "--label-prefix", help="Prefix for complexity labels (used with --label)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-analyze PRs even if they already have a complexity label"
    ),
    github_tokens: Optional[str] = typer.Option(
        None,
        "--github-tokens",
        help="Comma-separated list of GitHub tokens for rotation on rate limits. "
        "Can also be set via GH_TOKENS or GITHUB_TOKENS environment variables.",
    ),
):
    """
    Batch analyze multiple PRs from a file or date range.

    Either --input-file OR (--org, --since, --until) must be provided.

    By default, output is written to CSV with columns: pr_url, complexity, explanation.
    Use --label to apply complexity labels to PRs instead of CSV output.

    When using --label:
    - PRs that already have a complexity label are skipped
    - Labels are applied in the format "complexity:N" (customizable with --label-prefix)
    - No CSV output is generated unless --output is also specified

    If interrupted, run the same command again to resume from where it stopped.

    Use --workers to enable parallel processing (e.g., --workers 5 for 5 parallel workers).
    Note: Parallel processing may hit rate limits faster; adjust --sleep-seconds if needed.

    Token Rotation Mode:
    When multiple GitHub tokens are provided (via --github-tokens or GH_TOKENS env var),
    the tool automatically rotates between tokens when rate limits are hit. This allows
    for higher throughput when processing large batches of PRs.
    """
    try:
        # Validate inputs
        if input_file and (org or since or until):
            typer.echo("Error: Cannot specify both --input-file and date range options", err=True)
            raise typer.Exit(1)

        if not input_file and not (org and since and until):
            typer.echo(
                "Error: Must specify either --input-file OR (--org, --since, --until)", err=True
            )
            raise typer.Exit(1)

        # Require output_file unless --label is used
        if not label and not output_file:
            typer.echo("Error: --output is required unless --label is used", err=True)
            raise typer.Exit(1)

        # Get credentials
        openai_key = get_openai_api_key()

        if not openai_key:
            typer.echo("Error: OPENAI_API_KEY environment variable is required", err=True)
            typer.echo("Set it with: export OPENAI_API_KEY='your-key'", err=True)
            raise typer.Exit(1)

        # Get GitHub tokens - CLI option takes precedence over environment
        token_list: list[str] = []
        if github_tokens:
            # Parse comma-separated tokens from CLI
            token_list = [t.strip() for t in github_tokens.split(",") if t.strip()]
        else:
            # Fall back to environment variables
            token_list = get_github_tokens()

        # Create TokenRotator if we have multiple tokens
        token_rotator: Optional[TokenRotator] = None
        github_token: Optional[str] = None

        if len(token_list) > 1:
            token_rotator = TokenRotator(token_list)
            github_token = token_list[0]  # Use first token for non-rotatable operations
            typer.echo(f"Token rotation enabled with {len(token_list)} tokens", err=True)
        elif len(token_list) == 1:
            github_token = token_list[0]
        else:
            github_token = None

        # GitHub token is required for labeling
        if label and not github_token:
            typer.echo("Error: GitHub token is required for labeling PRs", err=True)
            typer.echo(
                "Set it with: export GH_TOKEN='your-token' or export GITHUB_TOKEN='your-token'",
                err=True,
            )
            raise typer.Exit(1)

        # Warn if GitHub token is missing (needed for private repos)
        if not github_token:
            typer.echo(
                "Warning: GH_TOKEN or GITHUB_TOKEN not set. Private repos may fail with 404 errors.",
                err=True,
            )
            typer.echo(
                "Set it with: export GH_TOKEN='your-token' or export GITHUB_TOKEN='your-token'",
                err=True,
            )

        # Load prompt
        try:
            prompt_text = load_prompt(prompt_file)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Get PR URLs
        if input_file:
            typer.echo(f"Loading PR URLs from file: {input_file}", err=True)
            pr_urls = load_pr_urls_from_file(input_file)
        else:
            # Parse dates
            try:
                since_dt = datetime.strptime(since, "%Y-%m-%d")
                until_dt = datetime.strptime(until, "%Y-%m-%d")
            except ValueError as e:
                typer.echo(f"Error: Invalid date format. Use YYYY-MM-DD: {e}", err=True)
                raise typer.Exit(1)

            if since_dt > until_dt:
                typer.echo("Error: --since date must be before --until date", err=True)
                raise typer.Exit(1)

            pr_urls = generate_pr_list_from_date_range(
                org=org,
                since=since_dt,
                until=until_dt,
                cache_file=cache_file,
                github_token=github_token,
                sleep_seconds=sleep_seconds,
            )

        # Create analyzer function with progress callback
        def progress_msg(msg: str) -> None:
            """Display progress messages."""
            typer.echo(msg, err=True)

        def analyze_fn(pr_url: str) -> dict:
            """Wrapper for analyze_pr_to_dict that handles errors."""
            return analyze_pr_to_dict(
                pr_url=pr_url,
                prompt_text=prompt_text,
                github_token=github_token,
                openai_key=openai_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                hunks_per_file=hunks_per_file,
                sleep_seconds=sleep_seconds,
                progress_callback=progress_msg,
                token_rotator=token_rotator,
            )

        # Validate workers
        if workers < 1:
            typer.echo("Error: --workers must be >= 1", err=True)
            raise typer.Exit(1)

        # Run batch analysis with labeling support
        from .batch import run_batch_analysis_with_labels

        run_batch_analysis_with_labels(
            pr_urls=pr_urls,
            output_file=output_file,
            analyze_fn=analyze_fn,
            resume=resume,
            workers=workers,
            label_prs=label,
            label_prefix=label_prefix,
            github_token=github_token,
            timeout=timeout,
            force=force,
        )

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


@app.command(name="rate-limit")
def rate_limit(
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or human"),
):
    """
    Check GitHub API rate limit status.

    Shows the current rate limit status for both core API and search API endpoints.
    """
    try:
        github_token = get_github_token()

        rate_limit_info = check_rate_limit(token=github_token)

        if format == "human":
            core = rate_limit_info["core"]
            search = rate_limit_info["search"]

            # Format reset time
            from datetime import datetime

            core_reset = datetime.fromtimestamp(core["reset"]) if core["reset"] else None
            search_reset = datetime.fromtimestamp(search["reset"]) if search["reset"] else None

            typer.echo("GitHub API Rate Limits:", err=False)
            typer.echo("", err=False)
            typer.echo("Core API:", err=False)
            typer.echo(f"  Limit: {core['limit']}", err=False)
            typer.echo(f"  Remaining: {core['remaining']}", err=False)
            typer.echo(f"  Used: {core['used']}", err=False)
            if core_reset:
                typer.echo(
                    f"  Resets at: {core_reset.strftime('%Y-%m-%d %H:%M:%S UTC')}", err=False
                )

            typer.echo("", err=False)
            typer.echo("Search API:", err=False)
            typer.echo(f"  Limit: {search['limit']}", err=False)
            typer.echo(f"  Remaining: {search['remaining']}", err=False)
            typer.echo(f"  Used: {search['used']}", err=False)
            if search_reset:
                typer.echo(
                    f"  Resets at: {search_reset.strftime('%Y-%m-%d %H:%M:%S UTC')}", err=False
                )
        else:
            # JSON output
            json_output = json.dumps(rate_limit_info, indent=2)
            typer.echo(json_output)

    except GitHubAPIError as e:
        typer.echo(f"GitHub API error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback

        if os.getenv("DEBUG"):
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command(name="label-pr")
def label_pr(
    pr_url: Optional[str] = typer.Argument(
        None, help="GitHub PR URL. If not provided, will try to infer from GitHub Actions context."
    ),
    prompt_file: Optional[Path] = typer.Option(
        None, "--prompt-file", "-p", help="Path to custom prompt file (default: embedded prompt)"
    ),
    model: str = typer.Option("gpt-5.2", "--model", "-m", help="OpenAI model name"),
    timeout: float = typer.Option(120.0, "--timeout", "-t", help="Request timeout in seconds"),
    max_tokens: int = typer.Option(50000, "--max-tokens", help="Maximum tokens for diff excerpt"),
    hunks_per_file: int = typer.Option(2, "--hunks-per-file", help="Maximum hunks per file"),
    sleep_seconds: float = typer.Option(
        0.7, "--sleep-seconds", help="Sleep between GitHub API calls"
    ),
    label_prefix: str = typer.Option(
        "complexity:", "--label-prefix", help="Prefix for the complexity label"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze but don't update label"),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", help="OpenAI API key"),
    github_token: Optional[str] = typer.Option(None, "--github-token", help="GitHub token"),
):
    """
    Analyze a GitHub PR and update its complexity label.

    Computes the complexity score and sets a label like "complexity:7" on the PR.
    Removes any existing complexity labels before adding the new one.

    Environment variables:
    - GH_TOKEN or GITHUB_TOKEN: GitHub API token (required for label updates)
    - OPENAI_API_KEY: OpenAI API key (required)
    """
    try:
        # Get PR URL
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

        # Parse PR URL
        owner, repo, pr = parse_pr_url(final_pr_url)
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

        if not final_github_token:
            typer.echo("Error: GitHub token is required to update labels", err=True)
            typer.echo("Set it with: export GH_TOKEN='your-token' or pass --github-token", err=True)
            raise typer.Exit(1)

        # Load prompt
        try:
            prompt_text = load_prompt(prompt_file)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Analyze PR
        typer.echo(f"Analyzing PR {owner}/{repo}#{pr}...", err=True)
        try:
            output = analyze_pr_to_dict(
                pr_url=final_pr_url,
                prompt_text=prompt_text,
                github_token=final_github_token,
                openai_key=final_openai_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                hunks_per_file=hunks_per_file,
                sleep_seconds=sleep_seconds,
            )
        except GitHubAPIError as e:
            if e.status_code == 404:
                typer.echo("Error: PR not found or not accessible", err=True)
                typer.echo(f"  URL: https://github.com/{owner}/{repo}/pull/{pr}", err=True)
            else:
                typer.echo(f"GitHub API error: {e}", err=True)
            raise typer.Exit(1)
        except LLMError as e:
            typer.echo(f"LLM error: {e}", err=True)
            raise typer.Exit(1)
        except InvalidResponseError as e:
            typer.echo(f"Invalid LLM response: {e}", err=True)
            raise typer.Exit(1)

        complexity = output["score"]
        typer.echo(f"Complexity score: {complexity}/10", err=True)
        typer.echo(f"Explanation: {output['explanation']}", err=True)

        # Update label
        if dry_run:
            label_name = f"{label_prefix}{complexity}"
            typer.echo(f"Dry run: Would set label '{label_name}'", err=True)
        else:
            typer.echo("Updating PR label...", err=True)
            try:
                label_name = update_complexity_label(
                    owner=owner,
                    repo=repo,
                    pr=pr,
                    complexity=complexity,
                    token=final_github_token,
                    label_prefix=label_prefix,
                    timeout=timeout,
                )
                typer.echo(f"Label set: {label_name}", err=True)
            except GitHubAPIError as e:
                typer.echo(f"Failed to update label: {e}", err=True)
                raise typer.Exit(1)

        # Output result as JSON
        result = {
            "score": complexity,
            "explanation": output["explanation"],
            "label": f"{label_prefix}{complexity}",
            "pr": pr,
            "repo": f"{owner}/{repo}",
            "url": final_pr_url,
            "dry_run": dry_run,
        }
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

        # Set GitHub Action outputs if available
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"score={complexity}\n")
                f.write(f"label={label_prefix}{complexity}\n")

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
        model="gpt-5.2",
        format="json",
        output_file=None,
        timeout=120.0,
        max_tokens=50000,
        hunks_per_file=2,
        sleep_seconds=0.7,
        dry_run=False,
    )


if __name__ == "__main__":
    app()
