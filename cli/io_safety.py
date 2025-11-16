"""Safe file I/O operations with path normalization."""
import json
from pathlib import Path
from typing import Any


def normalize_path(base: Path, subpath: str) -> Path:
    """
    Safely join a base directory and a user-supplied subpath without allowing
    escapes ('..') or absolute paths outside the base.
    """
    if not subpath or subpath.strip() == "":
        raise ValueError("Empty subpath is not allowed.")
    
    # Resolve the base to absolute
    base_resolved = base.resolve()
    
    # Join and resolve
    candidate = (base_resolved / subpath).resolve()
    
    # Check that candidate is within base
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Unsafe path detected: {subpath} would escape base directory {base_resolved}"
        )
    
    return candidate


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_text_file(path: Path) -> str:
    """Read text file safely."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()

