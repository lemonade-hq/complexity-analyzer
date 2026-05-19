"""Tests for analyze module."""

from cli.analyze import is_automated_sync_pr

SYNC_TITLE = "chore(cursor): [skip-ci] synced file(s) with lemonade-hq/cursor-rules"
SYNC_BODY = (
    "synced local file(s) with [lemonade-hq/cursor-rules](https://github.com/...).\n"
    "This PR was created automatically by the [repo-file-sync-action] workflow."
)


def test_sync_pr_detected_by_title_and_bot_author():
    assert is_automated_sync_pr(SYNC_TITLE, "github-actions[bot]") is True
    assert is_automated_sync_pr(SYNC_TITLE, "repo-file-sync-action") is True


def test_sync_pr_detected_by_title_and_body_signature():
    # Sync workflows often commit under a human PAT, so author_login is a
    # real username. The body signature is the fallback signal.
    assert is_automated_sync_pr(SYNC_TITLE, "dor-tzur-lmnd", body=SYNC_BODY) is True


def test_sync_title_alone_is_not_enough():
    # Human-authored PR — no bot login, no sync-action body marker.
    assert is_automated_sync_pr(SYNC_TITLE, "erez.dickman", body="rebased branch") is False
    assert is_automated_sync_pr(SYNC_TITLE, "erez.dickman", body=None) is False


def test_bot_author_alone_is_not_enough():
    # A bot PR with a substantive title should NOT short-circuit.
    title = "feat: bump dependency from 1.0 to 2.0"
    assert is_automated_sync_pr(title, "dependabot[bot]") is False


def test_skip_ci_alone_does_not_trigger():
    # "[skip-ci]" is widely used by humans for CI bypass on docs/typo fixes.
    # Without "synced file(s)", we must not short-circuit even with a bot author.
    assert is_automated_sync_pr("[skip-ci] fix typo in README", "github-actions[bot]") is False


def test_missing_signals():
    assert is_automated_sync_pr("", "github-actions[bot]") is False
    assert is_automated_sync_pr("synced file(s)", None) is False
    assert is_automated_sync_pr("synced file(s)", "") is False


def test_synced_local_file_phrasing():
    title = "chore: synced local file(s) with org/upstream"
    assert is_automated_sync_pr(title, "github-actions[bot]") is True
