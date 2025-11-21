"""Batch analysis orchestration with resume capability."""
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Callable
import typer

from .github import search_closed_prs, GitHubAPIError
from .io_safety import read_text_file, normalize_path


def load_pr_urls_from_file(file_path: Path) -> List[str]:
    """
    Load PR URLs from a text file (one URL per line).
    
    Args:
        file_path: Path to file containing PR URLs
        
    Returns:
        List of PR URLs
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or contains invalid URLs
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    content = read_text_file(file_path)
    urls = [line.strip() for line in content.splitlines() if line.strip()]
    
    if not urls:
        raise ValueError(f"Input file is empty: {file_path}")
    
    return urls


def generate_pr_list_from_date_range(
    org: str,
    since: datetime,
    until: datetime,
    cache_file: Optional[Path],
    github_token: Optional[str],
    sleep_seconds: float = 0.7,
) -> List[str]:
    """
    Generate PR list from date range, optionally using cache.
    
    If cache_file exists and is valid, loads from cache.
    Otherwise, fetches from GitHub and saves to cache.
    
    Args:
        org: Organization name
        since: Start date
        until: End date
        cache_file: Optional path to cache file
        github_token: GitHub token
        sleep_seconds: Sleep between API calls
        
    Returns:
        List of PR URLs
    """
    # Check cache first
    if cache_file and cache_file.exists():
        try:
            typer.echo(f"Loading PR list from cache: {cache_file}", err=True)
            urls = load_pr_urls_from_file(cache_file)
            typer.echo(f"Loaded {len(urls)} PRs from cache", err=True)
            return urls
        except Exception as e:
            typer.echo(f"Warning: Failed to load cache, will fetch from GitHub: {e}", err=True)
    
    # Fetch from GitHub
    typer.echo(f"Fetching closed PRs for org '{org}' from {since.date()} to {until.date()}...", err=True)
    try:
        urls = search_closed_prs(
            org=org,
            since=since,
            until=until,
            token=github_token,
            sleep_s=sleep_seconds,
        )
        typer.echo(f"Found {len(urls)} PRs", err=True)
        
        # Save to cache if specified
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with cache_file.open("w", encoding="utf-8") as f:
                    for url in urls:
                        f.write(f"{url}\n")
                typer.echo(f"Saved PR list to cache: {cache_file}", err=True)
            except Exception as e:
                typer.echo(f"Warning: Failed to save cache: {e}", err=True)
        
        return urls
    except GitHubAPIError as e:
        typer.echo(f"Error fetching PRs: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error fetching PRs: {e}", err=True)
        raise typer.Exit(1)


def load_completed_prs(output_file: Path) -> Set[str]:
    """
    Load already-completed PR URLs from existing CSV output file.
    
    Args:
        output_file: Path to CSV output file
        
    Returns:
        Set of PR URLs that have already been analyzed
    """
    completed: Set[str] = set()
    
    if not output_file.exists():
        return completed
    
    try:
        with output_file.open("r", encoding="utf-8") as f:
            # Try reading as DictReader first (with header)
            reader = csv.DictReader(f)
            
            # Check if CSV has proper header
            if reader.fieldnames and "pr_url" in reader.fieldnames:
                # Has proper header, read normally
                for row in reader:
                    pr_url = row.get("pr_url", "").strip()
                    if pr_url:
                        completed.add(pr_url)
            else:
                # No header or wrong header, read as simple CSV (pr_url is first column)
                f.seek(0)  # Reset to beginning
                reader = csv.reader(f)
                for row in reader:
                    if row and len(row) > 0:
                        pr_url = row[0].strip()
                        if pr_url and pr_url.startswith("http"):
                            completed.add(pr_url)
    except Exception as e:
        typer.echo(f"Warning: Failed to read existing output file: {e}", err=True)
        typer.echo("Will start from beginning", err=True)
    
    return completed


def write_csv_row(output_file: Path, pr_url: str, complexity: int, explanation: str) -> None:
    """
    Write a single row to CSV output file atomically.
    
    Creates file with header if it doesn't exist.
    
    Args:
        output_file: Path to CSV output file
        pr_url: PR URL
        complexity: Complexity score
        explanation: Explanation text
    """
    file_exists = output_file.exists()
    
    # Normalize path for safety
    if output_file.is_absolute():
        output_path = output_file
    else:
        output_path = normalize_path(Path.cwd(), str(output_file))
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write atomically using temp file
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    
    # Read existing content if file exists
    existing_rows = []
    if file_exists:
        try:
            with output_path.open("r", encoding="utf-8") as f:
                # Try reading as DictReader first (with header)
                reader = csv.DictReader(f)
                
                # Check if CSV has proper header
                if reader.fieldnames and "pr_url" in reader.fieldnames:
                    # Has proper header, read normally
                    for row in reader:
                        filtered_row = {
                            "pr_url": row.get("pr_url", "").strip(),
                            "complexity": row.get("complexity", "").strip(),
                            "explanation": row.get("explanation", "").strip(),
                        }
                        if filtered_row["pr_url"]:
                            existing_rows.append(filtered_row)
                else:
                    # No header or wrong header, read as simple CSV
                    f.seek(0)  # Reset to beginning
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) >= 3:
                            filtered_row = {
                                "pr_url": row[0].strip(),
                                "complexity": row[1].strip(),
                                "explanation": row[2].strip(),
                            }
                            if filtered_row["pr_url"] and filtered_row["pr_url"].startswith("http"):
                                existing_rows.append(filtered_row)
        except Exception as e:
            # If we can't read existing file, start fresh
            typer.echo(f"Warning: Failed to read existing CSV, will start fresh: {e}", err=True)
            file_exists = False
    
    # Write all rows (existing + new) to temp file
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["pr_url", "complexity", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Always write header since we are creating a new temp file
        writer.writeheader()
        
        # Write existing rows
        for row in existing_rows:
            writer.writerow(row)
        
        # Write new row
        writer.writerow({
            "pr_url": pr_url,
            "complexity": complexity,
            "explanation": explanation,
        })
    
    # Atomic replace
    tmp_path.replace(output_path)


def run_batch_analysis(
    pr_urls: List[str],
    output_file: Path,
    analyze_fn: Callable[[str], Dict[str, Any]],
    resume: bool = True,
    workers: int = 1,
) -> None:
    """
    Run batch analysis with resume capability and optional multi-threading.
    
    Args:
        pr_urls: List of PR URLs to analyze
        output_file: Path to CSV output file
        analyze_fn: Function that takes pr_url and returns dict with 'score' and 'explanation'
        resume: If True, skip already-completed PRs
        workers: Number of parallel workers (default: 1 for sequential execution)
    """
    # Load completed PRs if resuming
    completed = set()
    if resume:
        completed = load_completed_prs(output_file)
        if completed:
            typer.echo(f"Found {len(completed)} already-completed PRs, will skip them", err=True)
    
    # Filter out completed PRs
    remaining = [url for url in pr_urls if url not in completed]
    total = len(pr_urls)
    remaining_count = len(remaining)
    
    if remaining_count == 0:
        typer.echo("All PRs have already been analyzed!", err=True)
        return
    
    typer.echo(f"Analyzing {remaining_count} PRs (out of {total} total) with {workers} worker(s)", err=True)
    
    # Track progress
    completed_count = 0
    
    # Use ThreadPoolExecutor for parallel execution if workers > 1
    if workers == 1:
        # Sequential execution (original behavior)
        for idx, pr_url in enumerate(remaining, 1):
            try:
                typer.echo(f"\n[{idx}/{remaining_count}] Analyzing {pr_url}...", err=True)
                
                result = analyze_fn(pr_url)
                
                # Extract complexity and explanation
                complexity = result.get("score", result.get("complexity", 0))
                explanation = result.get("explanation", "")
                
                # Write to CSV
                write_csv_row(output_file, pr_url, complexity, explanation)
                typer.echo(f"✓ Completed: complexity={complexity}", err=True)
                completed_count += 1
                
            except KeyboardInterrupt:
                typer.echo(f"\n\nInterrupted. Progress saved. Resume by running the same command again.", err=True)
                raise typer.Exit(130)
            except Exception as e:
                typer.echo(f"✗ Error analyzing {pr_url}: {e}", err=True)
                typer.echo("Continuing with next PR...", err=True)
                # Continue to next PR instead of failing
    else:
        # Parallel execution with ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_url = {
                    executor.submit(analyze_fn, pr_url): pr_url
                    for pr_url in remaining
                }
                
                # Process results as they complete
                for future in as_completed(future_to_url):
                    pr_url = future_to_url[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        
                        # Extract complexity and explanation
                        complexity = result.get("score", result.get("complexity", 0))
                        explanation = result.get("explanation", "")
                        
                        # Write to CSV (on main thread to avoid race conditions)
                        write_csv_row(output_file, pr_url, complexity, explanation)
                        typer.echo(f"\n[{completed_count}/{remaining_count}] ✓ Completed {pr_url}: complexity={complexity}", err=True)
                        
                    except KeyboardInterrupt:
                        typer.echo(f"\n\nInterrupted. Cancelling pending tasks...", err=True)
                        # Cancel pending futures
                        for f in future_to_url:
                            f.cancel()
                        typer.echo("Progress saved. Resume by running the same command again.", err=True)
                        raise typer.Exit(130)
                    except Exception as e:
                        typer.echo(f"\n[{completed_count}/{remaining_count}] ✗ Error analyzing {pr_url}: {e}", err=True)
                        typer.echo("Continuing with next PR...", err=True)
                        # Continue to next PR instead of failing
        except KeyboardInterrupt:
            typer.echo(f"\n\nInterrupted. Progress saved. Resume by running the same command again.", err=True)
            raise typer.Exit(130)
    
    typer.echo(f"\n✓ Batch analysis complete! Results written to: {output_file}", err=True)

