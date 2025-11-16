"""LLM provider adapter with JSON schema validation and retries."""
import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from .scoring import parse_complexity_response, InvalidResponseError


class LLMError(Exception):
    """LLM provider error."""


class OpenAIProvider:
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-5.1", timeout: float = 120.0):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-5.1", "gpt-4")
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout
    
    def analyze_complexity(
        self,
        prompt: str,
        diff_excerpt: str,
        stats_json: str,
        title: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Analyze PR complexity using OpenAI.
        
        Args:
            prompt: System prompt/instructions
            diff_excerpt: Formatted diff excerpt
            stats_json: JSON string with stats
            title: PR title
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            
        Returns:
            Dict with 'complexity' (int) and 'explanation' (str)
            
        Raises:
            LLMError: If analysis fails after retries
        """
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"diff_excerpt:\n{diff_excerpt}\n\nstats_json:\n{stats_json}\n\ntitle:\n{title}",
            },
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from OpenAI")
                
                # Parse and validate response
                result = parse_complexity_response(content)
                
                # Add metadata
                result["provider"] = "openai"
                result["model"] = self.model
                result["tokens"] = response.usage.total_tokens if response.usage else None
                
                return result
                
            except InvalidResponseError as e:
                # If JSON parsing fails, try repair prompt
                if attempt < max_retries - 1:
                    repair_prompt = (
                        "The previous response was invalid. Please respond with ONLY a valid JSON object "
                        f"of the form: {{'complexity': <int 1..10>, 'explanation': '<string>'}}. "
                        f"Error: {str(e)}"
                    )
                    messages.append({
                        "role": "assistant",
                        "content": content if 'content' in locals() else "",
                    })
                    messages.append({
                        "role": "user",
                        "content": repair_prompt,
                    })
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise LLMError(f"Failed to parse response after {max_retries} attempts: {e}")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    # Add jitter
                    delay += (time.time() % 1) * 0.1
                    time.sleep(delay)
                    continue
                raise LLMError(f"OpenAI API error after {max_retries} attempts: {e}")
        
        raise LLMError(f"Failed after {max_retries} attempts")

