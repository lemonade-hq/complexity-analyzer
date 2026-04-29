#!/usr/bin/env bash
set -euo pipefail

GITLAB_DOMAIN="git.datik.io"

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <username> [months] [workers]"
  exit 1
fi

USERNAME="$1"
MONTHS="${2:-12}"
WORKERS="${3:-8}"

# Pass through SSL_NO_VERIFY if set
SSL_FLAG=""
if [ "${SSL_NO_VERIFY:-}" = "1" ] || [ "${SSL_NO_VERIFY:-}" = "true" ]; then
  SSL_FLAG="--no-verify-ssl"
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

GH_LIST="$TMPDIR/github_prs.txt"
GL_LIST="$TMPDIR/gitlab_mrs.txt"
PR_LIST="$TMPDIR/prs.txt"
RAW_CSV="$TMPDIR/raw.csv"
OUTPUT="${USERNAME}.csv"

echo "Fetching GitHub PRs and GitLab MRs for $USERNAME (last $MONTHS months) in parallel..."
python -m cli.list_user_prs "$USERNAME" --months "$MONTHS" $SSL_FLAG > "$GH_LIST" 2>"$TMPDIR/gh.err" &
GH_PID=$!
python -m cli.list_user_prs "$USERNAME" --months "$MONTHS" --gitlab "$GITLAB_DOMAIN" $SSL_FLAG > "$GL_LIST" 2>"$TMPDIR/gl.err" &
GL_PID=$!

wait "$GH_PID" || echo "  Warning: GitHub list failed: $(tail -1 "$TMPDIR/gh.err")"
wait "$GL_PID" || echo "  Warning: GitLab list failed: $(tail -1 "$TMPDIR/gl.err")"

GH_COUNT=$(wc -l < "$GH_LIST" | tr -d ' ')
GL_COUNT=$(wc -l < "$GL_LIST" | tr -d ' ')
echo "  Found $GH_COUNT GitHub PRs, $GL_COUNT GitLab MRs"

grep -hv '^$' "$GH_LIST" "$GL_LIST" > "$PR_LIST" || true
TOTAL=$(wc -l < "$PR_LIST" | tr -d ' ')

if [ "$TOTAL" -eq 0 ]; then
  echo "No PRs/MRs found."
  exit 0
fi

echo "Analyzing $TOTAL PRs/MRs..."
complexity-cli batch-analyze --input-file "$PR_LIST" --output "$RAW_CSV" --workers "$WORKERS" $SSL_FLAG

echo "Adding PR dates..."
python -m cli.add_pr_dates "$RAW_CSV" -o "$OUTPUT" -w "$WORKERS" $SSL_FLAG

echo "Done: $OUTPUT"
