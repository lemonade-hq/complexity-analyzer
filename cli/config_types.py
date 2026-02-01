"""Typed configuration classes for PR analysis."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .constants import (
    DEFAULT_HUNKS_PER_FILE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TIMEOUT,
)

if TYPE_CHECKING:
    from .github import TokenRotator


@dataclass
class AnalysisConfig:
    """Configuration for PR analysis."""

    # LLM settings
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS

    # Diff processing
    hunks_per_file: int = DEFAULT_HUNKS_PER_FILE

    # API settings
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS

    # Credentials (optional - can be provided at runtime)
    github_token: Optional[str] = None
    openai_key: Optional[str] = None

    # Token rotation (optional)
    token_rotator: Optional["TokenRotator"] = None

    # Custom prompt (optional)
    prompt_text: Optional[str] = None


@dataclass
class BatchConfig:
    """Configuration for batch PR analysis."""

    # Base analysis config
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Batch-specific settings
    workers: int = 1
    resume: bool = True

    # Labeling options
    label_prs: bool = False
    label_prefix: str = "complexity:"
    force: bool = False

    # Output
    output_file: Optional[str] = None


@dataclass
class OutputConfig:
    """Configuration for output formatting."""

    format: str = "json"  # "json" or "markdown"
    output_file: Optional[str] = None
    write_github_output: bool = True
