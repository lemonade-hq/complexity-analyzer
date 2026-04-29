"""List PRs/MRs opened by a user on GitHub or GitLab in the last N months."""

import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from .config import get_github_token, get_gitlab_token
from .constants import DEFAULT_TIMEOUT, GITHUB_API_BASE_URL, GITHUB_PER_PAGE, GITLAB_PER_PAGE
from .gitlab import build_gitlab_headers
from .utils import build_github_headers, ssl_verify_enabled


def _compute_since_date(months: int) -> datetime:
    """Compute the start date N months back from the 1st of current month."""
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month - months
    while month < 1:
        month += 12
        year -= 1
    return now.replace(year=year, month=month, day=1)


def list_user_prs(
    username: str,
    months: int = 1,
    token: Optional[str] = None,
    state: str = "all",
    timeout: float = DEFAULT_TIMEOUT,
) -> List[dict]:
    """
    List PRs opened by a GitHub user in the last N months.

    Args:
        username: GitHub username
        months: Number of months to look back (default: 1)
        token: GitHub token (falls back to GH_TOKEN / GITHUB_TOKEN env vars)
        state: PR state filter — "open", "closed", or "all" (default: "all")
        timeout: Request timeout in seconds

    Returns:
        List of dicts with PR info (url, title, repo, state, created_at, updated_at)
    """
    if not token:
        token = get_github_token()
    if not token:
        print("Error: No GitHub token found. Set GH_TOKEN or GITHUB_TOKEN.", file=sys.stderr)
        sys.exit(1)

    headers = build_github_headers(token)
    since = _compute_since_date(months)
    since_str = since.strftime("%Y-%m-%d")

    query = f"author:{username} is:pr created:>={since_str}"
    if state in ("open", "closed"):
        query += f" is:{state}"

    url = f"{GITHUB_API_BASE_URL}/search/issues"
    page = 1
    all_prs: List[dict] = []

    with httpx.Client(timeout=timeout) as client:
        while True:
            params = {"q": query, "per_page": GITHUB_PER_PAGE, "page": page, "sort": "created", "order": "desc"}
            resp = client.get(url, headers=headers, params=params)

            if resp.status_code == 403:
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(0, int(reset) - int(time.time())) + 1
                    print(f"Rate limited. Waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise SystemExit(f"GitHub API 403: {resp.text}")

            if resp.status_code != 200:
                raise SystemExit(f"GitHub API error {resp.status_code}: {resp.text}")

            data = resp.json()
            items = data.get("items", [])

            for item in items:
                pr_url = item.get("html_url", "")
                repo_url = item.get("repository_url", "")
                repo_name = "/".join(repo_url.split("/")[-2:]) if repo_url else ""

                all_prs.append({
                    "url": pr_url,
                    "title": item.get("title", ""),
                    "repo": repo_name,
                    "state": item.get("state", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                })

            if len(items) < GITHUB_PER_PAGE:
                break

            page += 1
            time.sleep(0.5)

    return all_prs


def list_user_mrs(
    username: str,
    months: int = 1,
    token: Optional[str] = None,
    state: str = "all",
    gitlab_domain: str = "gitlab.com",
    timeout: float = DEFAULT_TIMEOUT,
) -> List[dict]:
    """
    List MRs opened by a GitLab user in the last N months.

    Args:
        username: GitLab username
        months: Number of months to look back (default: 1)
        token: GitLab private token (falls back to GITLAB_TOKEN env var)
        state: MR state filter — "opened", "closed", "merged", or "all" (default: "all")
        gitlab_domain: GitLab instance domain (default: "gitlab.com")
        timeout: Request timeout in seconds

    Returns:
        List of dicts with MR info (url, title, repo, state, created_at, updated_at)
    """
    if not token:
        token = get_gitlab_token()
    if not token:
        print("Error: No GitLab token found. Set GITLAB_TOKEN.", file=sys.stderr)
        sys.exit(1)

    headers = build_gitlab_headers(token)
    since = _compute_since_date(months)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    scheme = "https"
    base_url = f"{scheme}://{gitlab_domain}"
    url = f"{base_url}/api/v4/merge_requests"
    page = 1
    all_mrs: List[dict] = []

    gl_state: Optional[str] = None
    if state == "open":
        gl_state = "opened"
    elif state in ("closed", "merged"):
        gl_state = state
    # "all" → omit state param (GitLab API treats absence as all states)

    with httpx.Client(timeout=timeout, verify=ssl_verify_enabled()) as client:
        while True:
            params: dict = {
                "author_username": username,
                "created_after": since_str,
                "per_page": GITLAB_PER_PAGE,
                "page": page,
                "sort": "desc",
                "order_by": "created_at",
                "scope": "all",
            }
            if gl_state:
                params["state"] = gl_state

            resp = client.get(url, headers=headers, params=params)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "60")
                wait = int(retry_after)
                print(f"Rate limited. Waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise SystemExit(f"GitLab API error {resp.status_code}: {resp.text[:500]}")

            items = resp.json()

            for item in items:
                web_url = item.get("web_url", "")
                project_path = ""
                full_ref = item.get("references", {}).get("full", "")
                if full_ref:
                    # references.full has shape "group/project!123"
                    project_path = full_ref.rsplit("!", 1)[0]

                all_mrs.append({
                    "url": web_url,
                    "title": item.get("title", ""),
                    "repo": project_path,
                    "state": item.get("state", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                })

            if len(items) < GITLAB_PER_PAGE:
                break

            page += 1
            time.sleep(0.5)

    return all_mrs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="List PRs/MRs opened by a user")
    parser.add_argument("username", help="GitHub or GitLab username")
    parser.add_argument("--months", type=int, default=1, help="Months to look back (default: 1)")
    parser.add_argument("--state", choices=["open", "closed", "merged", "all"], default="all", help="PR/MR state filter (default: all)")
    parser.add_argument("--gitlab", metavar="DOMAIN", help="Use GitLab instead of GitHub (e.g. gitlab.com or gitlab.mycompany.com)")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL certificate verification")
    args = parser.parse_args()

    if args.no_verify_ssl:
        os.environ["SSL_NO_VERIFY"] = "1"

    if args.gitlab:
        mrs = list_user_mrs(username=args.username, months=args.months, state=args.state, gitlab_domain=args.gitlab)
        for mr in mrs:
            print(mr["url"])
    else:
        prs = list_user_prs(username=args.username, months=args.months, state=args.state)
        for pr in prs:
            print(pr["url"])


if __name__ == "__main__":
    main()
