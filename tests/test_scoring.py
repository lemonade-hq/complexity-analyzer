"""Tests for scoring module."""

import pytest
from cli.scoring import parse_complexity_response, InvalidResponseError


def test_parse_valid_response():
    """Test parsing valid JSON response."""
    response = '{"complexity": 5, "explanation": "Test explanation"}'
    result = parse_complexity_response(response)
    assert result["complexity"] == 5
    assert result["explanation"] == "Test explanation"


def test_parse_response_with_extra_text():
    """Test parsing response with extra text."""
    response = 'Some text {"complexity": 7, "explanation": "Complex"} more text'
    result = parse_complexity_response(response)
    assert result["complexity"] == 7
    assert result["explanation"] == "Complex"


def test_parse_response_clamps_range():
    """Test that complexity is clamped to 1-10."""
    response = '{"complexity": 15, "explanation": "Test"}'
    result = parse_complexity_response(response)
    assert result["complexity"] == 10

    response = '{"complexity": -5, "explanation": "Test"}'
    result = parse_complexity_response(response)
    assert result["complexity"] == 1


def test_parse_response_sanitizes_newlines():
    """Test that newlines in explanation are replaced."""
    response = '{"complexity": 5, "explanation": "Line1\\nLine2"}'
    result = parse_complexity_response(response)
    assert "\n" not in result["explanation"]


def test_parse_invalid_json():
    """Test parsing invalid JSON."""
    with pytest.raises(InvalidResponseError):
        parse_complexity_response("not json")


def test_parse_missing_keys():
    """Test parsing response with missing keys."""
    with pytest.raises(InvalidResponseError):
        parse_complexity_response('{"complexity": 5}')
    with pytest.raises(InvalidResponseError):
        parse_complexity_response('{"explanation": "test"}')
