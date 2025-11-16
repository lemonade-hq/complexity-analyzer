"""Parse and validate LLM responses for complexity scoring."""
import json
import re
from typing import Dict, Any


class InvalidResponseError(Exception):
    """Raised when LLM response cannot be parsed."""


def parse_complexity_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response and extract complexity score and explanation.
    
    Expected format: {"complexity": <int 1..10>, "explanation": "<string>"}
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Dict with 'complexity' (int) and 'explanation' (str) keys
        
    Raises:
        InvalidResponseError: If response cannot be parsed or validated
    """
    # Try to extract JSON from response (may have extra text)
    response_text = response_text.strip()
    
    # Look for JSON object
    json_match = re.search(r"\{[^{}]*\"complexity\"[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise InvalidResponseError(f"Failed to parse JSON: {e}")
    
    # Validate structure
    if not isinstance(data, dict):
        raise InvalidResponseError("Response is not a JSON object")
    
    if "complexity" not in data:
        raise InvalidResponseError("Missing 'complexity' key in response")
    
    if "explanation" not in data:
        raise InvalidResponseError("Missing 'explanation' key in response")
    
    # Extract and validate complexity
    try:
        complexity = int(data["complexity"])
    except (ValueError, TypeError):
        raise InvalidResponseError(f"Invalid complexity value: {data['complexity']}")
    
    # Clamp to valid range
    complexity = max(1, min(10, complexity))
    
    # Extract and sanitize explanation
    explanation = str(data.get("explanation", "")).strip()
    # Remove newlines (replace with spaces)
    explanation = re.sub(r"\s+", " ", explanation)
    
    return {
        "complexity": complexity,
        "explanation": explanation,
    }

