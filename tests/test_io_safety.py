"""Tests for IO safety module."""

import json
import pytest

from cli.io_safety import normalize_path, read_text_file, write_json_atomic


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_safe_relative_path(self, tmp_path):
        """Test safe relative path normalization."""
        base = tmp_path
        result = normalize_path(base, "subdir/file.txt")
        assert result == base / "subdir" / "file.txt"

    def test_safe_nested_path(self, tmp_path):
        """Test safe nested path normalization."""
        base = tmp_path
        result = normalize_path(base, "a/b/c/d.txt")
        expected = base / "a" / "b" / "c" / "d.txt"
        assert result == expected

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        base = tmp_path / "subdir"
        base.mkdir()

        with pytest.raises(ValueError, match="Unsafe path detected"):
            normalize_path(base, "../../../etc/passwd")

    def test_path_traversal_blocked_complex(self, tmp_path):
        """Test that complex path traversal is blocked."""
        base = tmp_path / "subdir"
        base.mkdir()

        with pytest.raises(ValueError, match="Unsafe path detected"):
            normalize_path(base, "a/b/../../..")

    def test_absolute_path_within_base(self, tmp_path):
        """Test absolute path within base directory."""
        base = tmp_path
        subdir = base / "subdir"
        subdir.mkdir()

        # This should work since the absolute path is within base
        result = normalize_path(base, str(subdir / "file.txt"))
        assert result == subdir / "file.txt"

    def test_empty_subpath_raises(self, tmp_path):
        """Test that empty subpath raises ValueError."""
        with pytest.raises(ValueError, match="Empty subpath"):
            normalize_path(tmp_path, "")

    def test_whitespace_only_subpath_raises(self, tmp_path):
        """Test that whitespace-only subpath raises ValueError."""
        with pytest.raises(ValueError, match="Empty subpath"):
            normalize_path(tmp_path, "   ")

    def test_dot_path(self, tmp_path):
        """Test that single dot path works."""
        result = normalize_path(tmp_path, "./file.txt")
        assert result == tmp_path / "file.txt"


class TestWriteJsonAtomic:
    """Tests for write_json_atomic function."""

    def test_atomic_write_success(self, tmp_path):
        """Test successful atomic write."""
        output_file = tmp_path / "output.json"
        data = {"key": "value", "number": 42}

        write_json_atomic(output_file, data)

        assert output_file.exists()
        with output_file.open() as f:
            result = json.load(f)
        assert result == data

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        """Test parent directory creation."""
        output_file = tmp_path / "nested" / "deep" / "output.json"
        data = {"nested": True}

        write_json_atomic(output_file, data)

        assert output_file.exists()
        with output_file.open() as f:
            result = json.load(f)
        assert result == data

    def test_atomic_write_overwrites_existing(self, tmp_path):
        """Test that atomic write overwrites existing file."""
        output_file = tmp_path / "output.json"
        output_file.write_text('{"old": "data"}')

        new_data = {"new": "data"}
        write_json_atomic(output_file, new_data)

        with output_file.open() as f:
            result = json.load(f)
        assert result == new_data

    def test_atomic_write_no_temp_file_left(self, tmp_path):
        """Test that no temp file is left after write."""
        output_file = tmp_path / "output.json"
        write_json_atomic(output_file, {"test": True})

        # Check no .tmp file exists
        tmp_file = output_file.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_atomic_write_complex_data(self, tmp_path):
        """Test atomic write with complex data structures."""
        output_file = tmp_path / "output.json"
        data = {
            "string": "hello",
            "number": 123,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"a": {"b": {"c": "deep"}}},
            "unicode": "Hello, World!",
        }

        write_json_atomic(output_file, data)

        with output_file.open() as f:
            result = json.load(f)
        assert result == data


class TestReadTextFile:
    """Tests for read_text_file function."""

    def test_read_simple_file(self, tmp_path):
        """Test reading a simple text file."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!\nLine 2"
        test_file.write_text(content)

        result = read_text_file(test_file)
        assert result == content

    def test_read_unicode_file(self, tmp_path):
        """Test reading a file with unicode content."""
        test_file = tmp_path / "unicode.txt"
        content = "Hello, World!"
        test_file.write_text(content, encoding="utf-8")

        result = read_text_file(test_file)
        assert result == content

    def test_read_empty_file(self, tmp_path):
        """Test reading an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = read_text_file(test_file)
        assert result == ""

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a non-existent file raises error."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            read_text_file(test_file)
