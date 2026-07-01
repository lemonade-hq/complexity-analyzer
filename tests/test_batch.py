"""Tests for batch module."""

import csv
import pytest
from datetime import datetime
from unittest.mock import patch
from cli.batch import (
    load_pr_urls_from_file,
    generate_pr_list_from_date_range,
    load_completed_prs,
    write_csv_row,
    run_batch_analysis,
    run_batch_analysis_with_labels,
)


def test_load_pr_urls_from_file(tmp_path):
    """Test loading PR URLs from file."""
    pr_file = tmp_path / "prs.txt"
    pr_file.write_text(
        "https://github.com/owner/repo/pull/123\n"
        "https://github.com/owner/repo/pull/124\n"
        "https://github.com/owner/repo/pull/125\n"
    )

    urls = load_pr_urls_from_file(pr_file)
    assert len(urls) == 3
    assert urls[0] == "https://github.com/owner/repo/pull/123"
    assert urls[1] == "https://github.com/owner/repo/pull/124"
    assert urls[2] == "https://github.com/owner/repo/pull/125"


def test_load_pr_urls_from_file_not_found(tmp_path):
    """Test loading PR URLs from non-existent file."""
    pr_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        load_pr_urls_from_file(pr_file)


def test_load_pr_urls_from_file_empty(tmp_path):
    """Test loading PR URLs from empty file."""
    pr_file = tmp_path / "empty.txt"
    pr_file.write_text("")

    with pytest.raises(ValueError):
        load_pr_urls_from_file(pr_file)


@patch("cli.batch.search_prs")
def test_generate_pr_list_from_cache(mock_search, tmp_path):
    """Test generating PR list from cache file."""
    cache_file = tmp_path / "cache.txt"
    cache_file.write_text(
        "https://github.com/owner/repo/pull/123\n" "https://github.com/owner/repo/pull/124\n"
    )

    urls = generate_pr_list_from_date_range(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        cache_file=cache_file,
        github_token="token",
    )

    assert len(urls) == 2
    mock_search.assert_not_called()


@patch("cli.batch.search_prs")
def test_generate_pr_list_from_github(mock_search, tmp_path):
    """Test generating PR list from GitHub API."""
    pr_urls = [
        "https://github.com/owner/repo/pull/123",
        "https://github.com/owner/repo/pull/124",
    ]

    def mock_search_with_callback(
        org, since, until, token, sleep_s, on_pr_found, progress_callback, client, state="closed"
    ):
        """Mock that calls the on_pr_found callback for each URL."""
        for url in pr_urls:
            if on_pr_found:
                on_pr_found(url)
        return pr_urls

    mock_search.side_effect = mock_search_with_callback

    cache_file = tmp_path / "cache.txt"

    urls = generate_pr_list_from_date_range(
        org="testorg",
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        cache_file=cache_file,
        github_token="token",
    )

    assert len(urls) == 2
    mock_search.assert_called_once()
    assert cache_file.exists()
    assert "https://github.com/owner/repo/pull/123" in cache_file.read_text()


def test_load_completed_prs(tmp_path):
    """Test loading completed PRs from CSV."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        "pr_url,complexity,explanation\n"
        "https://github.com/owner/repo/pull/123,5,Test explanation\n"
        "https://github.com/owner/repo/pull/124,3,Another explanation\n"
    )

    completed = load_completed_prs(csv_file)
    assert len(completed) == 2
    assert "https://github.com/owner/repo/pull/123" in completed
    assert "https://github.com/owner/repo/pull/124" in completed


def test_load_completed_prs_not_exists(tmp_path):
    """Test loading completed PRs from non-existent file."""
    csv_file = tmp_path / "nonexistent.csv"

    completed = load_completed_prs(csv_file)
    assert len(completed) == 0


def test_write_csv_row_new_file(tmp_path):
    """Test writing CSV row to new file."""
    csv_file = tmp_path / "results.csv"

    write_csv_row(csv_file, "https://github.com/owner/repo/pull/123", 5, "Test explanation")

    assert csv_file.exists()
    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["pr_url"] == "https://github.com/owner/repo/pull/123"
        assert rows[0]["complexity"] == "5"
        assert rows[0]["explanation"] == "Test explanation"


def test_write_csv_row_existing_file(tmp_path):
    """Test writing CSV row to existing file."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        "pr_url,complexity,explanation\n"
        "https://github.com/owner/repo/pull/123,5,Test explanation\n"
    )

    write_csv_row(csv_file, "https://github.com/owner/repo/pull/124", 3, "Another explanation")

    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["pr_url"] == "https://github.com/owner/repo/pull/123"
        assert rows[1]["pr_url"] == "https://github.com/owner/repo/pull/124"


@patch("cli.batch.typer")
def test_run_batch_analysis(mock_typer, tmp_path):
    """Test running batch analysis."""
    output_file = tmp_path / "results.csv"

    pr_urls = [
        "https://github.com/owner/repo/pull/123",
        "https://github.com/owner/repo/pull/124",
    ]

    def analyze_fn(url):
        return {
            "score": 5,
            "explanation": f"Analysis for {url}",
        }

    run_batch_analysis(pr_urls, output_file, analyze_fn, resume=True)

    assert output_file.exists()
    with output_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2


@patch("cli.batch.typer")
def test_run_batch_analysis_resume(mock_typer, tmp_path):
    """Test batch analysis resume capability."""
    output_file = tmp_path / "results.csv"
    output_file.write_text(
        "pr_url,complexity,explanation\n" "https://github.com/owner/repo/pull/123,5,Already done\n"
    )

    pr_urls = [
        "https://github.com/owner/repo/pull/123",
        "https://github.com/owner/repo/pull/124",
    ]

    analyze_count = 0

    def analyze_fn(url):
        nonlocal analyze_count
        analyze_count += 1
        return {
            "score": 5,
            "explanation": f"Analysis for {url}",
        }

    run_batch_analysis(pr_urls, output_file, analyze_fn, resume=True)

    # Should only analyze the second PR (first is already done)
    assert analyze_count == 1
    assert output_file.exists()
    with output_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2


@patch("cli.batch.typer")
def test_batch_analysis_with_gitlab_urls(mock_typer, tmp_path):
    """Test that batch analysis works with GitLab MR URLs."""
    output_file = tmp_path / "results.csv"

    pr_urls = [
        "https://gitlab.com/group/repo/-/merge_requests/1",
        "https://github.com/owner/repo/pull/123",
    ]

    def analyze_fn(url):
        return {"score": 7, "explanation": f"Analysis for {url}"}

    run_batch_analysis(pr_urls, output_file, analyze_fn, resume=False)

    assert output_file.exists()
    with output_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2


@patch("cli.batch.has_complexity_label")
@patch("cli.batch.typer")
def test_batch_labels_skips_gitlab_for_label_check(mock_typer, mock_has_label, tmp_path):
    """Test that label checking skips GitLab URLs gracefully."""
    output_file = tmp_path / "results.csv"

    pr_urls = [
        "https://gitlab.com/group/repo/-/merge_requests/1",
        "https://github.com/owner/repo/pull/123",
    ]

    # Only the GitHub URL should be checked for existing labels
    mock_has_label.return_value = None

    def analyze_fn(url):
        return {"score": 5, "explanation": "test"}

    run_batch_analysis_with_labels(
        pr_urls=pr_urls,
        output_file=output_file,
        analyze_fn=analyze_fn,
        label_prs=True,
        github_token="token",
        resume=False,
    )

    # has_complexity_label should only be called for the GitHub URL
    assert mock_has_label.call_count == 1
    call_args = mock_has_label.call_args
    assert call_args[0][0] == "owner"  # owner
    assert call_args[0][1] == "repo"  # repo


@patch("cli.batch.update_complexity_label")
@patch("cli.batch.typer")
def test_batch_labels_skips_gitlab_for_labeling(mock_typer, mock_update_label, tmp_path):
    """Test that labeling is skipped for GitLab MR URLs."""
    output_file = tmp_path / "results.csv"

    pr_urls = [
        "https://gitlab.com/group/repo/-/merge_requests/1",
    ]

    mock_update_label.return_value = "complexity:5"

    def analyze_fn(url):
        return {"score": 5, "explanation": "test"}

    run_batch_analysis_with_labels(
        pr_urls=pr_urls,
        output_file=output_file,
        analyze_fn=analyze_fn,
        label_prs=True,
        github_token="token",
        resume=False,
        force=True,
    )

    # update_complexity_label should NOT be called for GitLab URLs
    mock_update_label.assert_not_called()
