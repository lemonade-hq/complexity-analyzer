"""Diff preprocessing: redaction, filtering, chunking, and stats."""

import os
import re
from typing import Dict, Any, List, Tuple


# File filtering patterns
IGNORE_EXT_RE = re.compile(
    r"(?:\.(?:png|jpg|jpeg|gif|webp|ico|pdf|zip|gz|bz2|xz|mp4|mov|mp3|wav|ogg|wasm|min\.js|map|lock)|package-lock\.json|pnpm-lock\.yaml)$",
    re.IGNORECASE,
)
IGNORE_PATH_RE = re.compile(
    r"(?:^|/)(?:vendor|node_modules|dist|build|coverage)(?:/|$)",
    re.IGNORECASE,
)

# Redaction patterns
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{8,}['\"]"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Default limits
DEFAULT_MAX_TOKENS = 50000
DEFAULT_HUNKS_PER_FILE = 2


def redact(text: str) -> str:
    """Redact secrets and emails from diff text."""
    text = SECRET_RE.sub("[REDACTED_SECRET]", text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text


def parse_diff_sections(
    diff_text: str, hunks_per_file: int = DEFAULT_HUNKS_PER_FILE
) -> Dict[str, List[str]]:
    """
    Parse diff into sections per file, limiting hunks per file.

    Args:
        diff_text: Raw unified diff text
        hunks_per_file: Maximum hunks to include per file

    Returns:
        Dict mapping filename -> list of lines
    """
    files: Dict[str, List[str]] = {}
    current_file: str | None = None
    current_lines: List[str] = []
    hunk_count = 0

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # Save previous file
            if current_file is not None and current_lines:
                files[current_file] = current_lines

            # Parse new file
            parts = line.strip().split()
            if len(parts) >= 4:
                b_path = parts[-1]
                filename = b_path[2:] if b_path.startswith("b/") else b_path
                current_file = filename
                current_lines = [line]
                hunk_count = 0
            else:
                current_file = None
                current_lines = []
                hunk_count = 0
        elif line.startswith("@@ "):
            if current_file is not None:
                if hunk_count < hunks_per_file:
                    current_lines.append(line)
                    hunk_count += 1
                # Skip hunks beyond limit
        else:
            if current_file is not None:
                if hunk_count > 0:
                    # We're inside a hunk
                    current_lines.append(line)
                else:
                    # Header lines before first hunk
                    if line.startswith(("index ", "--- ", "+++ ")):
                        current_lines.append(line)

    # Save last file
    if current_file is not None and current_lines:
        files[current_file] = current_lines

    return files


def filter_file(path: str) -> bool:
    """Check if file should be included (not ignored)."""
    if IGNORE_PATH_RE.search(path):
        return False
    if IGNORE_EXT_RE.search(path):
        return False
    return True


def ext_from_filename(filename: str) -> str:
    """Extract file extension."""
    base = os.path.basename(filename)
    _, ext = os.path.splitext(base)
    return ext.lstrip(".").lower()


def build_stats(meta: Dict[str, Any], filenames: List[str]) -> Dict[str, Any]:
    """
    Build statistics from PR metadata and selected filenames.

    Args:
        meta: PR metadata dict from GitHub API
        filenames: List of selected filenames

    Returns:
        Stats dict with additions, deletions, changedFiles, byExt, byLang, fileCount
    """
    additions = meta.get("additions", 0)
    deletions = meta.get("deletions", 0)
    changed_files = meta.get("changed_files") or meta.get("changedFiles")
    if not changed_files:
        files_list = meta.get("files") or []
        changed_files = len(files_list)

    # Count by extension
    by_ext: Dict[str, int] = {}
    for fn in filenames:
        ext = ext_from_filename(fn)
        by_ext[ext] = by_ext.get(ext, 0) + 1

    # Map extensions to languages
    lang_map = {
        "ts": "TypeScript",
        "tsx": "TypeScript",
        "js": "JavaScript",
        "jsx": "JavaScript",
        "py": "Python",
        "rb": "Ruby",
        "go": "Go",
        "java": "Java",
        "kt": "Kotlin",
        "cs": "CSharp",
        "php": "PHP",
        "rs": "Rust",
        "swift": "Swift",
        "m": "Objective-C",
        "scala": "Scala",
        "sh": "Shell",
        "yml": "YAML",
        "yaml": "YAML",
        "json": "JSON",
        "sql": "SQL",
    }

    by_lang: Dict[str, int] = {}
    for ext, count in by_ext.items():
        lang = lang_map.get(ext or "", "Other")
        by_lang[lang] = by_lang.get(lang, 0) + count

    return {
        "additions": additions,
        "deletions": deletions,
        "changedFiles": changed_files,
        "byExt": by_ext,
        "byLang": by_lang,
        "fileCount": len(filenames),
    }


def truncate_to_token_limit(text: str, max_tokens: int) -> Tuple[str, int]:
    """
    Truncate text to at most max_tokens using tiktoken.

    Args:
        text: Text to truncate
        max_tokens: Maximum token count

    Returns:
        Tuple of (truncated_text, token_count)
    """
    if max_tokens <= 0:
        return "", 0

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text, len(tokens)
        # Decode only the first max_tokens
        truncated = enc.decode(tokens[:max_tokens])
        return truncated, max_tokens
    except ImportError:
        # Fallback: approximate by characters (rough estimate: 4 chars per token)
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text, len(text) // 4
        return text[:char_limit], max_tokens


def make_prompt_input(
    url: str, title: str, stats: Dict[str, Any], files: List[str], diff_excerpt: str
) -> str:
    """
    Format prompt input with PR context and diff.

    Args:
        url: PR URL
        title: PR title
        stats: Stats dict
        files: List of changed files
        diff_excerpt: Truncated diff text

    Returns:
        Formatted prompt input string
    """
    header = (
        f"PR: {url}\n"
        f"Title: {title}\n"
        f"Stats: additions={stats.get('additions')} deletions={stats.get('deletions')} "
        f"changedFiles={stats.get('changedFiles')} filesTop={len(files)}\n"
        f"Files: {', '.join(files[:10])}\n"
        f"--- DIFF START ---\n"
    )
    return header + (diff_excerpt or "") + "\n--- DIFF END ---"


def process_diff(
    diff_text: str,
    meta: Dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    hunks_per_file: int = DEFAULT_HUNKS_PER_FILE,
) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Process diff: redact, filter, chunk, truncate, and build stats.

    Args:
        diff_text: Raw diff text
        meta: PR metadata
        max_tokens: Maximum tokens for diff excerpt
        hunks_per_file: Maximum hunks per file

    Returns:
        Tuple of (formatted_diff_excerpt, stats_dict, selected_files_list)
    """
    # Redact secrets
    redacted_diff = redact(diff_text)

    # Parse into sections
    sections = parse_diff_sections(redacted_diff, hunks_per_file)

    # Filter files
    selected_files: List[str] = []
    excerpt_lines: List[str] = []
    for fn, lines in sections.items():
        if not filter_file(fn):
            continue
        selected_files.append(fn)
        excerpt_lines.extend(lines)

    # Combine and truncate
    excerpt = "\n".join(excerpt_lines)
    truncated, _tok = truncate_to_token_limit(excerpt, max_tokens)

    # Build stats
    stats = build_stats(meta, selected_files)

    return truncated, stats, selected_files
