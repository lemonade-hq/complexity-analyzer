# Complexity CLI

A command-line tool to analyze GitHub pull request complexity using LLMs.

## Installation

Install from source:

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
- `--model`, `-m`: OpenAI model name (default: `gpt-5.2`)
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

### Batch Analysis

Analyze multiple PRs in batch mode with resume capability.

#### From Input File

```bash
# Create a file with PR URLs (one per line)
cat > prs.txt << EOF
https://github.com/owner/repo/pull/123
https://github.com/owner/repo/pull/124
https://github.com/owner/repo/pull/125
EOF

# Analyze all PRs (sequential, default)
complexity-cli batch-analyze --input-file prs.txt --output results.csv

# Analyze with 8 parallel workers for faster processing
complexity-cli batch-analyze --input-file prs.txt --output results.csv --workers 8
```

#### From Date Range

```bash
# Analyze all PRs closed in an organization within a date range
complexity-cli batch-analyze \
  --org myorg \
  --since 2024-01-01 \
  --until 2024-01-31 \
  --output results.csv \
  --cache pr-list.txt

# On subsequent runs, the cache file will be used to skip fetching the PR list
complexity-cli batch-analyze \
  --org myorg \
  --since 2024-01-01 \
  --until 2024-01-31 \
  --output results.csv \
  --cache pr-list.txt
```

#### Resume Capability

If the batch analysis is interrupted (Ctrl+C), you can resume by running the same command again. The tool will automatically skip PRs that have already been analyzed by reading the existing output file.

```bash
# First run (interrupted after 10 PRs)
complexity-cli batch-analyze --input-file prs.txt --output results.csv

# Resume (will skip the 10 already-analyzed PRs)
complexity-cli batch-analyze --input-file prs.txt --output results.csv
```

#### Batch Analysis Options

- `--input-file`, `-i`: File containing PR URLs (one per line)
- `--org`: Organization name (for date range search)
- `--since`: Start date in YYYY-MM-DD format (for date range search)
- `--until`: End date in YYYY-MM-DD format (for date range search)
- `--output`, `-o`: Output CSV file path (required)
- `--cache`: Cache file for PR list (used with date range to avoid re-fetching)
- `--prompt-file`, `-p`: Path to custom prompt file
- `--model`, `-m`: OpenAI model name (default: `gpt-5.2`)
- `--timeout`, `-t`: Request timeout in seconds (default: 120)
- `--max-tokens`: Maximum tokens for diff excerpt (default: 50000)
- `--hunks-per-file`: Maximum hunks per file (default: 2)
- `--sleep-seconds`: Sleep between GitHub API calls (default: 0.7)
- `--resume/--no-resume`: Enable/disable resume from existing output (default: enabled)
- `--workers`, `-w`: Number of parallel workers for concurrent analysis (default: 4, minimum: 1)

**Note:** When using `--workers` > 1, results are written to the CSV file as soon as each analyzer finishes, so the output order may differ from the input order. This does not affect resume capability - the tool still correctly skips already-analyzed PRs.

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

### Batch CSV Output

Batch analysis outputs a CSV file with the following columns:

- `pr_url`: The GitHub PR URL
- `complexity`: The complexity score (1-10)
- `explanation`: The explanation text

Example:

```csv
pr_url,complexity,explanation
https://github.com/owner/repo/pull/123,5,"Multiple modules/services with non-trivial control flow changes"
https://github.com/owner/repo/pull/124,3,"Simple refactoring with minimal changes"
https://github.com/owner/repo/pull/125,8,"Complex architectural changes across multiple services"
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