"""Batch analysis orchestration with resume capability."""
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Callable, Tuple
import threading
import typer
import httpx

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
    Automatically splits date range if it exceeds GitHub's 1000 result limit.
    
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
    cache_file_handle = None
    try:
        # Open cache file for writing if specified
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file_handle = cache_file.open("w", encoding="utf-8")
                typer.echo(f"Writing PRs to cache file: {cache_file}", err=True)
            except Exception as e:
                typer.echo(f"Warning: Failed to open cache file for writing: {e}", err=True)
                cache_file = None
        
        # Define callback to write PRs incrementally
        def write_pr_to_cache(pr_url: str) -> None:
            """Write PR URL to cache file immediately."""
            if cache_file_handle:
                try:
                    cache_file_handle.write(f"{pr_url}\n")
                    cache_file_handle.flush()  # Ensure it's written to disk immediately
                except Exception as e:
                    typer.echo(f"Warning: Failed to write PR to cache: {e}", err=True)
        
        # Progress callback for rate limit messages
        def progress_msg(msg: str) -> None:
            """Display progress messages."""
            typer.echo(msg, err=True)
        
        # Try fetching with the full date range first
        with httpx.Client(timeout=60.0) as client:
            try:
                urls = search_closed_prs(
                    org=org,
                    since=since,
                    until=until,
                    token=github_token,
                    sleep_s=sleep_seconds,
                    on_pr_found=write_pr_to_cache if cache_file else None,
                    progress_callback=progress_msg,
                    client=client,
                )
                typer.echo(f"Found {len(urls)} PRs", err=True)
                return urls
            except GitHubAPIError as e:
                # Check if it's the 1000 result limit error
                if e.status_code == 422 and "1000 search results" in str(e.message).lower():
                    typer.echo(f"Date range exceeds GitHub's 1000 result limit. Splitting into smaller ranges...", err=True)
                    
                    # Calculate date range in days
                    date_range_days = (until - since).days + 1
                    
                    # Start with splitting into halves, but ensure minimum chunk size
                    # Try splitting into progressively smaller chunks
                    chunk_days = max(1, date_range_days // 2)
                    
                    all_urls = []
                    current_since = since
                    chunk_num = 1
                    initial_chunk_days = chunk_days
                    
                    while current_since <= until:
                        # Calculate chunk end date
                        chunk_until = min(current_since + timedelta(days=chunk_days - 1), until)
                        
                        typer.echo(f"Fetching chunk {chunk_num}: {current_since.date()} to {chunk_until.date()}...", err=True)
                        
                        try:
                            chunk_urls = search_closed_prs(
                                org=org,
                                since=current_since,
                                until=chunk_until,
                                token=github_token,
                                sleep_s=sleep_seconds,
                                on_pr_found=write_pr_to_cache if cache_file else None,
                                progress_callback=progress_msg,
                                client=client,
                            )
                            all_urls.extend(chunk_urls)
                            typer.echo(f"Found {len(chunk_urls)} PRs in chunk {chunk_num}", err=True)
                            
                            # Success - move to next chunk
                            current_since = chunk_until + timedelta(days=1)
                            chunk_num += 1
                            # Reset chunk_days to initial value for next chunks
                            chunk_days = initial_chunk_days
                            
                        except GitHubAPIError as chunk_error:
                            if chunk_error.status_code == 422 and "1000 search results" in str(chunk_error.message).lower():
                                # Still hitting limit, need smaller chunks
                                if chunk_days <= 1:
                                    # Can't split further - this is a very active org
                                    typer.echo(f"Error: Even single-day chunks exceed 1000 PRs. Consider using repository-specific queries.", err=True)
                                    raise typer.Exit(1)
                                
                                # Reduce chunk size and retry this chunk (don't advance current_since)
                                chunk_days = max(1, chunk_days // 2)
                                typer.echo(f"Chunk still too large. Reducing to {chunk_days} days and retrying...", err=True)
                                # Update initial_chunk_days so future chunks also use smaller size
                                initial_chunk_days = chunk_days
                                continue
                            else:
                                # Other error, re-raise
                                raise
                        
                        # Small delay between chunks
                        if current_since <= until:
                            time.sleep(sleep_seconds)
                    
                    typer.echo(f"Found {len(all_urls)} total PRs across {chunk_num - 1} chunks", err=True)
                    return all_urls
                else:
                    # Other GitHub API error, re-raise
                    raise
        
    except GitHubAPIError as e:
        typer.echo(f"Error fetching PRs: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error fetching PRs: {e}", err=True)
        raise typer.Exit(1)
    finally:
        # Close cache file if opened (ensure it's closed even if there's an error)
        if cache_file_handle:
            try:
                cache_file_handle.close()
                if cache_file:
                    typer.echo(f"Saved PR list to cache: {cache_file}", err=True)
            except Exception as e:
                typer.echo(f"Warning: Failed to close cache file: {e}", err=True)


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
            reader = csv.DictReader(f)
            # Check if CSV has the expected columns
            if "pr_url" in reader.fieldnames or reader.fieldnames:
                for row in reader:
                    # Handle both possible column names
                    pr_url = row.get("pr_url") or row.get("PR link") or row.get(list(row.keys())[0] if row else "")
                    if pr_url:
                        completed.add(pr_url.strip())
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
    has_header = False
    if file_exists:
        try:
            with output_path.open("r", encoding="utf-8") as f:
                # Try to detect if file has headers by reading first line
                first_line = f.readline()
                f.seek(0)  # Reset to beginning
                
                # Check if first line looks like headers (contains expected field names)
                first_line_lower = first_line.lower().strip()
                has_header = (
                    "pr_url" in first_line_lower or 
                    "pr link" in first_line_lower or
                    "complexity" in first_line_lower
                )
                
                if has_header:
                    # File has headers, use DictReader
                    reader = csv.DictReader(f)
                    # Map various possible column names to our standard fieldnames
                    for row in reader:
                        mapped_row = {}
                        # Try to find pr_url column (handle various names)
                        pr_url_val = (
                            row.get("pr_url") or 
                            row.get("PR link") or 
                            row.get("pr link") or
                            row.get(list(row.keys())[0] if row else "")
                        )
                        mapped_row["pr_url"] = pr_url_val or ""
                        
                        # Try to find complexity column
                        complexity_val = (
                            row.get("complexity") or 
                            row.get("Complexity") or
                            row.get(list(row.keys())[1] if len(row.keys()) > 1 else "")
                        )
                        mapped_row["complexity"] = complexity_val or ""
                        
                        # Try to find explanation column
                        explanation_val = (
                            row.get("explanation") or 
                            row.get("Explanation") or
                            row.get(list(row.keys())[2] if len(row.keys()) > 2 else "")
                        )
                        mapped_row["explanation"] = explanation_val or ""
                        
                        existing_rows.append(mapped_row)
                else:
                    # File doesn't have headers, use regular reader and map columns
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            existing_rows.append({
                                "pr_url": row[0].strip(),
                                "complexity": row[1].strip(),
                                "explanation": row[2].strip(),
                            })
                        elif len(row) >= 2:
                            # Handle case with just URL and complexity
                            existing_rows.append({
                                "pr_url": row[0].strip(),
                                "complexity": row[1].strip(),
                                "explanation": "",
                            })
                        elif len(row) >= 1:
                            # Handle case with just URL
                            existing_rows.append({
                                "pr_url": row[0].strip(),
                                "complexity": "",
                                "explanation": "",
                            })
        except Exception:
            # If we can't read existing file, start fresh
            file_exists = False
    
    # Write all rows (existing + new) to temp file
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["pr_url", "complexity", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Always write header (even if file existed, we're standardizing the format)
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
    Run batch analysis with resume capability and optional parallel processing.
    
    Args:
        pr_urls: List of PR URLs to analyze
        output_file: Path to CSV output file
        analyze_fn: Function that takes pr_url and returns dict with 'score' and 'explanation'
        resume: If True, skip already-completed PRs
        workers: Number of parallel workers (1 = sequential, >1 = parallel)
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
    
    if workers > 1:
        typer.echo(f"Analyzing {remaining_count} PRs (out of {total} total) with {workers} parallel workers", err=True)
    else:
        typer.echo(f"Analyzing {remaining_count} PRs (out of {total} total)", err=True)
    
    # Thread-safe counter and lock for CSV writing
    completed_lock = threading.Lock()
    completed_count = [0]  # Use list to allow modification in nested function
    
    def process_single_pr(pr_url: str, idx: int) -> Tuple[str, Optional[int], Optional[str], Optional[Exception]]:
        """Process a single PR and return result or error."""
        try:
            if workers == 1:
                typer.echo(f"\n[{idx}/{remaining_count}] Analyzing {pr_url}...", err=True)
            else:
                typer.echo(f"[{idx}/{remaining_count}] Analyzing {pr_url}...", err=True)
            
            result = analyze_fn(pr_url)
            
            # Extract complexity and explanation
            complexity = result.get("score", result.get("complexity", 0))
            explanation = result.get("explanation", "")
            
            return pr_url, complexity, explanation, None
        except Exception as e:
            return pr_url, None, None, e
    
    # Process PRs sequentially or in parallel
    if workers == 1:
        # Sequential processing (original behavior)
        for idx, pr_url in enumerate(remaining, 1):
            try:
                pr_url_result, complexity, explanation, error = process_single_pr(pr_url, idx)
                
                if error:
                    # Handle 404 errors (PR not found) with a clearer message
                    if isinstance(error, GitHubAPIError) and error.status_code == 404:
                        typer.echo(f"⚠ Skipping {pr_url_result}: PR not found or inaccessible (404)", err=True)
                        typer.echo("  This may be a private repo - ensure GH_TOKEN or GITHUB_TOKEN is set with proper access", err=True)
                    else:
                        typer.echo(f"✗ Error analyzing {pr_url_result}: {error}", err=True)
                    typer.echo("Continuing with next PR...", err=True)
                    continue
                
                # Write to CSV
                write_csv_row(output_file, pr_url_result, complexity, explanation)
                typer.echo(f"✓ Completed: complexity={complexity}", err=True)
                
            except KeyboardInterrupt:
                typer.echo(f"\n\nInterrupted. Progress saved. Resume by running the same command again.", err=True)
                raise typer.Exit(130)
    else:
        # Parallel processing
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_pr = {
                    executor.submit(process_single_pr, pr_url, idx): (pr_url, idx)
                    for idx, pr_url in enumerate(remaining, 1)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_pr):
                    pr_url, idx = future_to_pr[future]
                    try:
                        pr_url_result, complexity, explanation, error = future.result()
                        
                        if error:
                            # Handle 404 errors (PR not found) with a clearer message
                            if isinstance(error, GitHubAPIError) and error.status_code == 404:
                                typer.echo(f"⚠ Skipping {pr_url_result}: PR not found or inaccessible (404)", err=True)
                                typer.echo("  This may be a private repo - ensure GH_TOKEN or GITHUB_TOKEN is set with proper access", err=True)
                            else:
                                typer.echo(f"✗ Error analyzing {pr_url_result}: {error}", err=True)
                            typer.echo("Continuing with next PR...", err=True)
                            continue
                        
                        # Write to CSV (atomic writes are thread-safe)
                        write_csv_row(output_file, pr_url_result, complexity, explanation)
                        
                        with completed_lock:
                            completed_count[0] += 1
                            typer.echo(f"✓ [{completed_count[0]}/{remaining_count}] Completed {pr_url_result}: complexity={complexity}", err=True)
                            
                    except Exception as e:
                        typer.echo(f"✗ Unexpected error processing {pr_url}: {e}", err=True)
                        typer.echo("Continuing with next PR...", err=True)
                        
        except KeyboardInterrupt:
            typer.echo(f"\n\nInterrupted. Progress saved. Resume by running the same command again.", err=True)
            raise typer.Exit(130)
    
    typer.echo(f"\n✓ Batch analysis complete! Results written to: {output_file}", err=True)

