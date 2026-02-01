"""Core PR analysis logic extracted from main.py."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import validate_owner_repo, validate_pr_number
from .config_types import AnalysisConfig
from .github import fetch_pr, fetch_pr_with_rotation
from .io_safety import read_text_file
from .llm import OpenAIProvider
from .preprocess import make_prompt_input, process_diff
from .utils import parse_pr_url


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    """
    Load prompt from file or use default embedded prompt.

    Args:
        prompt_file: Optional path to custom prompt file

    Returns:
        Prompt text

    Raises:
        FileNotFoundError: If prompt file not found
    """
    if prompt_file:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return read_text_file(prompt_file)

    # Load default embedded prompt
    default_prompt_path = Path(__file__).parent / "prompt" / "default.txt"
    if not default_prompt_path.exists():
        raise FileNotFoundError(f"Default prompt not found: {default_prompt_path}")
    return read_text_file(default_prompt_path)


def analyze_single_pr(
    pr_url: str,
    config: AnalysisConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Analyze a single PR and return the result.

    This is the core analysis function that can be reused by both single PR
    and batch analysis workflows.

    Args:
        pr_url: GitHub PR URL
        config: Analysis configuration
        progress_callback: Optional callback for progress messages

    Returns:
        Dict with keys: score, explanation, provider, model, tokens, timestamp,
        repo, pr, url, title

    Raises:
        ValueError: If PR URL is invalid or config is missing required values
        GitHubAPIError: If GitHub API call fails
        LLMError: If LLM call fails
        InvalidResponseError: If LLM response is invalid
    """
    if not config.openai_key:
        raise ValueError("OpenAI API key is required")

    # Parse PR URL
    owner, repo, pr = parse_pr_url(pr_url)
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    # Load prompt if not provided
    prompt_text = config.prompt_text
    if not prompt_text:
        prompt_text = load_prompt()

    # Fetch PR - use token rotator if available, otherwise use single token
    if config.token_rotator:
        diff_text, meta = fetch_pr_with_rotation(
            owner,
            repo,
            pr,
            config.token_rotator,
            sleep_s=config.sleep_seconds,
            progress_callback=progress_callback,
            timeout=config.timeout,
        )
    else:
        diff_text, meta = fetch_pr(
            owner,
            repo,
            pr,
            config.github_token,
            sleep_s=config.sleep_seconds,
            progress_callback=progress_callback,
        )

    title = (meta.get("title") or "").strip()

    # Process diff
    truncated_diff, stats, selected_files = process_diff(
        diff_text, meta, max_tokens=config.max_tokens, hunks_per_file=config.hunks_per_file
    )

    # Format prompt input
    diff_for_prompt = make_prompt_input(pr_url, title, stats, selected_files, truncated_diff)

    # Call LLM
    provider = OpenAIProvider(config.openai_key, model=config.model, timeout=config.timeout)
    result = provider.analyze_complexity(
        prompt=prompt_text,
        diff_excerpt=diff_for_prompt,
        stats_json=json.dumps(stats),
        title=title,
    )

    # Prepare output
    return {
        "score": result["complexity"],
        "explanation": result["explanation"],
        "provider": result.get("provider", "openai"),
        "model": result.get("model", config.model),
        "tokens": result.get("tokens"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo": f"{owner}/{repo}",
        "pr": pr,
        "url": pr_url,
        "title": title,
    }


def handle_dry_run(
    pr_url: str,
    config: AnalysisConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Handle dry-run mode - fetch and process PR but don't call LLM.

    Args:
        pr_url: GitHub PR URL
        config: Analysis configuration
        progress_callback: Optional callback for progress messages

    Returns:
        Dict with dry run information including diff stats

    Raises:
        ValueError: If PR URL is invalid
        GitHubAPIError: If GitHub API call fails
    """
    # Parse PR URL
    owner, repo, pr = parse_pr_url(pr_url)
    validate_owner_repo(owner, repo)
    validate_pr_number(pr)

    # Fetch PR
    if config.token_rotator:
        diff_text, meta = fetch_pr_with_rotation(
            owner,
            repo,
            pr,
            config.token_rotator,
            sleep_s=config.sleep_seconds,
            progress_callback=progress_callback,
            timeout=config.timeout,
        )
    else:
        diff_text, meta = fetch_pr(
            owner,
            repo,
            pr,
            config.github_token,
            sleep_s=config.sleep_seconds,
            progress_callback=progress_callback,
        )

    title = (meta.get("title") or "").strip()

    # Process diff
    truncated_diff, stats, selected_files = process_diff(
        diff_text, meta, max_tokens=config.max_tokens, hunks_per_file=config.hunks_per_file
    )

    return {
        "dry_run": True,
        "title": title,
        "diff_excerpt_length": len(truncated_diff),
        "selected_files_count": len(selected_files),
        "selected_files": selected_files,
        "stats": stats,
        "repo": f"{owner}/{repo}",
        "pr": pr,
        "url": pr_url,
    }


def format_output(result: Dict[str, Any], output_format: str) -> str:
    """
    Format analysis result for output.

    Args:
        result: Analysis result dict
        output_format: Output format ("json" or "markdown")

    Returns:
        Formatted output string
    """
    if output_format == "markdown":
        return f"""# PR Complexity Analysis

**Score:** {result['score']}/10

**Explanation:** {result['explanation']}

**Details:**
- Repository: {result['repo']}
- PR: #{result['pr']}
- Model: {result['model']}
- Tokens used: {result.get('tokens', 'N/A')}
"""
    else:
        # JSON output
        return json.dumps(
            {
                "score": result["score"],
                "explanation": result["explanation"],
                "provider": result["provider"],
                "model": result["model"],
                "tokens": result.get("tokens"),
                "timestamp": result["timestamp"],
            },
            ensure_ascii=False,
            indent=2,
        )
