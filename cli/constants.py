"""Centralized constants for the CLI."""

# Timeouts
DEFAULT_TIMEOUT = 120.0

# API sleep/delay
DEFAULT_SLEEP_SECONDS = 0.7

# Token display
TOKEN_VISIBLE_CHARS = 4

# LLM defaults
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_MAX_TOKENS = 50000
DEFAULT_HUNKS_PER_FILE = 2

# Rate limits
MAX_RATE_LIMIT_WAIT = 3600  # 1 hour
RATE_LIMIT_CACHE_SECONDS = 30

# CSV
CSV_BATCH_SIZE = 10  # Write every N rows

# Retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# GitHub API
GITHUB_API_VERSION = "2022-11-28"
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_PER_PAGE = 100  # Max items per page for GitHub API
