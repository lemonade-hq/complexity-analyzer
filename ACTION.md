# PR Complexity Analyzer GitHub Action

A reusable GitHub Action that analyzes pull request complexity using OpenAI's LLM. This action uses the complexity-analyzer tool to calculate a complexity score (1-10) and provide an explanation for the score.

## Features

- 🔍 Analyzes PR diffs to calculate complexity scores (1-10 scale)
- 🤖 Powered by OpenAI's GPT models
- 📊 Provides detailed explanations for complexity ratings
- 🔄 Reusable across all repositories
- ⚙️ Highly configurable with sensible defaults
- 📝 Supports both JSON and Markdown output formats
- 💬 Optional automatic PR comments

## Usage

### Basic Usage

```yaml
name: Analyze PR Complexity

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  complexity-analysis:
    runs-on: ubuntu-latest
    steps:
      - name: Analyze PR Complexity
        uses: lemonade-hq/complexity-analyzer@main
        with:
          pr-url: ${{ github.event.pull_request.html_url }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Advanced Usage with All Options

```yaml
name: Analyze PR Complexity

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  complexity-analysis:
    runs-on: ubuntu-latest
    steps:
      - name: Analyze PR Complexity
        id: complexity
        uses: lemonade-hq/complexity-analyzer@main
        with:
          pr-url: ${{ github.event.pull_request.html_url }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
          model: 'gpt-4'
          format: 'json'
          output-file: 'complexity-report.json'
          timeout: '120'
          max-tokens: '50000'
          hunks-per-file: '2'
          sleep-seconds: '0.7'
          python-version: '3.11'
      
      - name: Display complexity score
        run: |
          echo "Complexity Score: ${{ steps.complexity.outputs.score }}"
          echo "Explanation: ${{ steps.complexity.outputs.explanation }}"
      
      - name: Upload complexity report
        uses: actions/upload-artifact@v4
        with:
          name: complexity-report
          path: complexity-report.json
```

### Post Complexity as PR Comment

```yaml
name: Analyze PR Complexity

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  complexity-analysis:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - name: Analyze PR Complexity
        id: complexity
        uses: lemonade-hq/complexity-analyzer@main
        with:
          pr-url: ${{ github.event.pull_request.html_url }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
          format: 'json'
      
      - name: Comment PR
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const score = ${{ steps.complexity.outputs.score }};
            const explanation = JSON.parse(${{ steps.complexity.outputs.explanation }});
            
            const comment = `## 📊 PR Complexity Analysis
            
            **Complexity Score:** ${score}/10
            
            **Explanation:**
            ${explanation}
            
            ---
            *Analyzed by [Complexity Analyzer](https://github.com/lemonade-hq/complexity-analyzer)*`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `pr-url` | GitHub PR URL (e.g., `https://github.com/owner/repo/pull/123`) | Yes | - |
| `openai-api-key` | OpenAI API key for LLM analysis | Yes | - |
| `github-token` | GitHub token for API access | No | `${{ github.token }}` |
| `model` | OpenAI model name (e.g., `gpt-4`, `gpt-5.1`) | No | `gpt-5.1` |
| `format` | Output format: `json` or `markdown` | No | `json` |
| `output-file` | Path to write output file | No | - |
| `timeout` | Request timeout in seconds | No | `120` |
| `max-tokens` | Maximum tokens for diff excerpt | No | `50000` |
| `hunks-per-file` | Maximum hunks per file to analyze | No | `2` |
| `sleep-seconds` | Sleep duration between GitHub API calls | No | `0.7` |
| `python-version` | Python version to use | No | `3.11` |

## Outputs

| Output | Description |
|--------|-------------|
| `score` | Complexity score from 1-10 |
| `explanation` | Detailed explanation of the complexity rating |
| `output` | Full JSON output from the analyzer |

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

When using `format: 'markdown'`, the action outputs:

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

## Required Secrets

You need to configure the following secrets in your repository:

1. **`OPENAI_API_KEY`** (Required)
   - Your OpenAI API key
   - Get one from: https://platform.openai.com/api-keys
   
2. **`GITHUB_TOKEN`** (Optional but recommended)
   - Automatically provided by GitHub Actions
   - Used for accessing GitHub API (higher rate limits for authenticated requests)
   - Required for private repositories

### Setting up Secrets

1. Go to your repository settings
2. Navigate to **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add `OPENAI_API_KEY` with your OpenAI API key

## Complexity Score Scale

The analyzer returns a score from 1-10:

- **1-3**: Simple changes (docs, small bug fixes, minor refactors)
- **4-6**: Moderate complexity (new features, multiple files, some logic changes)
- **7-8**: Significant complexity (architectural changes, complex logic)
- **9-10**: Very high complexity (major refactors, system-wide changes)

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
- Diffs are redacted to remove secrets and emails
- File paths are normalized to prevent directory traversal

## Limitations

- Requires OpenAI API access (costs may apply based on usage)
- Large PRs may be truncated to fit within token limits
- Analysis quality depends on the selected model
- Rate limited by GitHub API (0.7s delay between calls by default)

## Examples

### Save Report as Artifact

```yaml
- name: Analyze PR
  uses: lemonade-hq/complexity-analyzer@main
  with:
    pr-url: ${{ github.event.pull_request.html_url }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    output-file: 'complexity.json'

- name: Upload Report
  uses: actions/upload-artifact@v4
  with:
    name: complexity-analysis
    path: complexity.json
```

### Multi-Model Analysis

```yaml
- name: Analyze with GPT-4
  id: gpt4
  uses: lemonade-hq/complexity-analyzer@main
  with:
    pr-url: ${{ github.event.pull_request.html_url }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    model: 'gpt-4'

- name: Analyze with GPT-5.1
  id: gpt5
  uses: lemonade-hq/complexity-analyzer@main
  with:
    pr-url: ${{ github.event.pull_request.html_url }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    model: 'gpt-5.1'

- name: Compare Results
  run: |
    echo "GPT-4 Score: ${{ steps.gpt4.outputs.score }}"
    echo "GPT-5.1 Score: ${{ steps.gpt5.outputs.score }}"
```

## Troubleshooting

### Error: OPENAI_API_KEY environment variable is required

Make sure you've set up the `OPENAI_API_KEY` secret in your repository settings.

### Error: PR not found or not accessible

- Ensure the PR URL is correct
- For private repositories, make sure `GITHUB_TOKEN` is provided
- Check that the token has sufficient permissions

### Error: Rate limit exceeded

Increase the `sleep-seconds` parameter to add more delay between GitHub API calls.

### Token limit exceeded

Reduce `max-tokens` or `hunks-per-file` to limit the amount of diff data sent to the LLM.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/lemonade-hq/complexity-analyzer) for more information.

## Support

For issues, questions, or feature requests, please open an issue in the [GitHub repository](https://github.com/lemonade-hq/complexity-analyzer/issues).
