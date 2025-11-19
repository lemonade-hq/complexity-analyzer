"""Tests for preprocess module."""

from cli.preprocess import redact, filter_file, parse_diff_sections, build_stats


def test_redact_secrets():
    """Test secret redaction."""
    text = 'api_key = "secret123456"'
    result = redact(text)
    assert "[REDACTED_SECRET]" in result
    assert "secret123456" not in result


def test_redact_emails():
    """Test email redaction."""
    text = "Contact: user@example.com"
    result = redact(text)
    assert "[REDACTED_EMAIL]" in result
    assert "user@example.com" not in result


def test_filter_file():
    """Test file filtering."""
    assert filter_file("src/main.py") is True
    assert filter_file("node_modules/package.js") is False
    assert filter_file("dist/bundle.js") is False
    assert filter_file("package-lock.json") is False
    assert filter_file("test.png") is False


def test_parse_diff_sections():
    """Test diff parsing."""
    diff = """diff --git a/file.py b/file.py
index 123..456
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
    sections = parse_diff_sections(diff)
    assert "file.py" in sections
    assert len(sections["file.py"]) > 0


def test_build_stats():
    """Test stats building."""
    meta = {
        "additions": 10,
        "deletions": 5,
        "changed_files": 2,
    }
    files = ["file1.py", "file2.ts"]
    stats = build_stats(meta, files)
    assert stats["additions"] == 10
    assert stats["deletions"] == 5
    assert stats["changedFiles"] == 2
    assert stats["fileCount"] == 2
    assert "byExt" in stats
    assert "byLang" in stats
