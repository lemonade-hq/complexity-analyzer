"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This defines the interface that all LLM providers must implement,
    allowing for easy swapping between different providers (OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    def analyze_complexity(
        self,
        prompt: str,
        diff_excerpt: str,
        stats_json: str,
        title: str,
    ) -> Dict[str, Any]:
        """
        Analyze PR complexity and return score with explanation.

        Args:
            prompt: System prompt/instructions for the LLM
            diff_excerpt: Formatted diff excerpt to analyze
            stats_json: JSON string with PR statistics
            title: PR title

        Returns:
            Dict with at least:
                - 'complexity' (int): Complexity score 1-10
                - 'explanation' (str): Explanation of the score
                - 'provider' (str): Provider name
                - 'model' (str): Model used
                - 'tokens' (int, optional): Total tokens used

        Raises:
            LLMError: If analysis fails
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass
