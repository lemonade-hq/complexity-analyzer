# Complexity CLI

A command-line tool to analyze GitHub pull request complexity using LLMs.

## Installation

```bash
pip install complexity-cli
```

Or install from source:

```bash
git clone <repo-url>
cd complexity-cli
pip install -e .
```

## Usage

### Basic Usage

```bash
export OPENAI_API_KEY="your-key"
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123"
```

### Options

- `--prompt-file`, `-p`: Path to custom prompt file (default: embedded prompt)
- `--model`, `-m`: OpenAI model name (default: `gpt-5.1`)
- `--format`, `-f`: Output format: `json` or `markdown` (default: `json`)
- `--out`, `-o`: Write output to file
- `--timeout`, `-t`: Request timeout in seconds (default: 120)
- `--max-tokens`: Maximum tokens for diff excerpt (default: 50000)
- `--hunks-per-file`: Maximum hunks per file (default: 2)
- `--sleep-seconds`: Sleep between GitHub API calls (default: 0.7)
- `--dry-run`: Fetch PR but don't call LLM

### Environment Variables

- `OPENAI_API_KEY` (required): OpenAI API key
- `GH_TOKEN` or `GITHUB_TOKEN` (optional): GitHub API token for private repos or higher rate limits

### Examples

```bash
# Analyze a PR with default settings
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123"

# Use a different model
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123" --model gpt-4

# Output as markdown
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123" --format markdown

# Save output to file
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123" --out result.json

# Dry run (fetch PR but skip LLM)
complexity-cli analyze-pr "https://github.com/owner/repo/pull/123" --dry-run
```

## Output Format

### JSON Output (default)

```json
{
  "score": 5,
  "explanation": "Multiple modules/services with non-trivial control flow changes",
  "provider": "openai",
  "model": "gpt-5.1",
  "tokens": 1234,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Markdown Output

```markdown
# PR Complexity Analysis

**Score:** 5/10

**Explanation:** Multiple modules/services with non-trivial control flow changes

**Details:**
- Repository: owner/repo
- PR: #123
- Model: gpt-5.1
- Tokens used: 1234
```

## How It Works

1. **Fetch PR**: Downloads the PR diff and metadata from GitHub API
2. **Process Diff**: 
   - Redacts secrets and emails
   - Filters out binary files, lockfiles, and vendor directories
   - Truncates to token limit while preserving structure
   - Builds statistics (additions, deletions, file counts, languages)
3. **Analyze**: Sends formatted prompt to LLM with diff excerpt, stats, and title
4. **Score**: Parses LLM response and returns complexity score (1-10) with explanation

## Security

- Secrets are never logged or persisted
- API keys are read from environment variables only
- File paths are normalized to prevent directory traversal
- Diffs are redacted to remove secrets and emails

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black cli tests

# Lint
ruff check cli tests
```