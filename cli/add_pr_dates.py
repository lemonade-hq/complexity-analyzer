"""Add PR/MR creation date as the 2nd column to a complexity results CSV."""

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx

from .config import get_github_token, get_gitlab_token
from .constants import DEFAULT_TIMEOUT, GITHUB_API_BASE_URL
from .gitlab import _encode_project_path, build_gitlab_headers
from .utils import build_github_headers, parse_mr_url, ssl_verify_enabled

MAX_RATE_LIMIT_RETRIES = 5


def _fetch_github_created_at(
    owner: str, repo: str, pr_number: int, client: httpx.Client, headers: dict
) -> str:
    url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}"
    for _ in range(MAX_RATE_LIMIT_RETRIES):
        resp = client.get(url, headers=headers)
        if resp.status_code == 403:
            reset = resp.headers.get("X-RateLimit-Reset")
            if not reset:
                raise SystemExit(f"GitHub API 403: {resp.text}")
            wait = max(0, int(reset) - int(time.time())) + 1
            print(f"Rate limited. Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            print(f"Warning: failed to fetch {owner}/{repo}#{pr_number} (HTTP {resp.status_code})", file=sys.stderr)
            return ""
        created_at = resp.json().get("created_at", "")
        return created_at[:10] if created_at else ""
    print(f"Warning: gave up on {owner}/{repo}#{pr_number} after {MAX_RATE_LIMIT_RETRIES} rate-limit retries", file=sys.stderr)
    return ""


def _fetch_gitlab_created_at(
    project_path: str, mr_iid: int, base_url: str, client: httpx.Client, headers: dict
) -> str:
    encoded = _encode_project_path(project_path)
    url = f"{base_url}/api/v4/projects/{encoded}/merge_requests/{mr_iid}"
    for _ in range(MAX_RATE_LIMIT_RETRIES):
        resp = client.get(url, headers=headers)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "60"))
            print(f"Rate limited. Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            print(f"Warning: failed to fetch {project_path}!{mr_iid} (HTTP {resp.status_code})", file=sys.stderr)
            return ""
        created_at = resp.json().get("created_at", "")
        return created_at[:10] if created_at else ""
    print(f"Warning: gave up on {project_path}!{mr_iid} after {MAX_RATE_LIMIT_RETRIES} rate-limit retries", file=sys.stderr)
    return ""


def _fetch_date_for_row(
    index: int,
    row: list,
    client: httpx.Client,
    gh_headers: dict,
    gl_headers: dict,
) -> tuple:
    pr_url = row[0]
    try:
        owner_or_project, repo, number, provider, base_url = parse_mr_url(pr_url)
        if provider == "gitlab":
            date = _fetch_gitlab_created_at(owner_or_project, number, base_url, client, gl_headers)
        else:
            date = _fetch_github_created_at(owner_or_project, repo, number, client, gh_headers)
    except ValueError:
        print(f"Warning: skipping invalid URL on row {index}: {pr_url}", file=sys.stderr)
        date = ""
    return index, date


def add_dates_to_csv(
    input_path: str,
    output_path: Optional[str] = None,
    token: Optional[str] = None,
    workers: int = 1,
):
    """Read a CSV, add pr_date as the 2nd column, write to output."""
    gh_token = token or get_github_token()
    gl_token = get_gitlab_token()

    if not gh_token and not gl_token:
        print("Error: No token found. Set GH_TOKEN/GITHUB_TOKEN or GITLAB_TOKEN.", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = input_path

    gh_headers = build_github_headers(gh_token) if gh_token else {}
    gl_headers = build_gitlab_headers(gl_token) if gl_token else {}

    with open(input_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Empty CSV.", file=sys.stderr)
        return

    header = rows[0]
    data_rows = rows[1:]
    total = len(data_rows)
    dates = [""] * total

    completed = 0
    lock = threading.Lock()

    print(f"Fetching dates for {total} PRs/MRs with {workers} worker(s)...", file=sys.stderr)

    with httpx.Client(timeout=DEFAULT_TIMEOUT, verify=ssl_verify_enabled()) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_date_for_row, i, row, client, gh_headers, gl_headers): i
                for i, row in enumerate(data_rows)
            }

            for future in as_completed(futures):
                index, date = future.result()
                dates[index] = date
                with lock:
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"Progress: {completed}/{total} ({100 * completed // total}%)", file=sys.stderr)

    new_header = [header[0], "pr_date"] + header[1:]
    new_rows = [new_header]
    for row, date in zip(data_rows, dates):
        new_rows.append([row[0], date] + row[1:])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Done. Wrote {total} rows to {output_path}", file=sys.stderr)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add PR creation date column to a complexity CSV")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: overwrite input)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL certificate verification")
    args = parser.parse_args()

    if args.no_verify_ssl:
        os.environ["SSL_NO_VERIFY"] = "1"

    add_dates_to_csv(args.input, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
