"""Abstract base class for all LLM backends."""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
    Common interface for LLM backends.

    RAG pipeline and Agent code only depend on this interface —
    never on a concrete implementation (Ollama, LMDeploy, etc.).
    """

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        """Send prompt, return the model's response as a plain string."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict) -> "BaseBackend":
        """Construct an instance from a YAML config dict."""
        ...
