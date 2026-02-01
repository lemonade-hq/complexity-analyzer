"""Batched CSV writer for performance optimization."""

import csv
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .constants import CSV_BATCH_SIZE
from .io_safety import normalize_path


class CSVBatchWriter:
    """
    Buffered CSV writer that batches writes for performance.

    Thread-safe implementation that accumulates rows in a buffer and
    flushes them to disk when the batch size is reached or when explicitly closed.
    """

    def __init__(
        self,
        output_file: Path,
        batch_size: int = CSV_BATCH_SIZE,
        fieldnames: Optional[List[str]] = None,
    ):
        """
        Initialize the CSV batch writer.

        Args:
            output_file: Path to the CSV output file
            batch_size: Number of rows to accumulate before flushing (default: 10)
            fieldnames: CSV column names (default: ["pr_url", "complexity", "explanation"])
        """
        self._output_file = output_file
        self._batch_size = batch_size
        self._fieldnames = fieldnames or ["pr_url", "complexity", "explanation"]
        self._buffer: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._initialized = False

        # Normalize path for safety
        if output_file.is_absolute():
            self._output_path = output_file
        else:
            self._output_path = normalize_path(Path.cwd(), str(output_file))

        # Ensure parent directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_initialized(self) -> None:
        """Create file with header if it doesn't exist."""
        if self._initialized:
            return

        if not self._output_path.exists():
            # Create file with header
            with self._output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

        self._initialized = True

    def add_row(self, pr_url: str, complexity: int, explanation: str) -> None:
        """
        Add a row to the buffer, flush if batch size reached.

        Args:
            pr_url: PR URL
            complexity: Complexity score
            explanation: Explanation text
        """
        with self._lock:
            self._ensure_initialized()

            self._buffer.append(
                {
                    "pr_url": pr_url,
                    "complexity": str(complexity),
                    "explanation": explanation,
                }
            )

            if len(self._buffer) >= self._batch_size:
                self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Write buffered rows to CSV (caller must hold lock)."""
        if not self._buffer:
            return

        # Append rows to file
        with self._output_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            for row in self._buffer:
                writer.writerow(row)

        self._buffer.clear()

    def flush(self) -> None:
        """Manually flush buffered rows to CSV."""
        with self._lock:
            self._flush_unlocked()

    def close(self) -> None:
        """Flush remaining rows and close."""
        self.flush()

    @property
    def output_path(self) -> Path:
        """Get the output file path."""
        return self._output_path

    @property
    def pending_count(self) -> int:
        """Get the number of rows pending in the buffer."""
        with self._lock:
            return len(self._buffer)

    def __enter__(self) -> "CSVBatchWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - flush remaining rows."""
        self.close()


def load_completed_prs_from_csv(csv_file: Path) -> set:
    """
    Load already-completed PR URLs from existing CSV output file.

    Args:
        csv_file: Path to CSV output file

    Returns:
        Set of PR URLs that have already been analyzed
    """
    completed: set = set()

    if not csv_file.exists():
        return completed

    try:
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    # Handle various possible column names
                    pr_url = (
                        row.get("pr_url")
                        or row.get("PR link")
                        or row.get("pr link")
                        or row.get(list(row.keys())[0] if row else "")
                    )
                    if pr_url:
                        completed.add(pr_url.strip())
    except Exception:
        # If we can't read the file, return empty set
        pass

    return completed
