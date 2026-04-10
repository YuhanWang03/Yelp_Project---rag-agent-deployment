"""Ollama backend — wraps the local Ollama /api/chat endpoint."""

import requests

from yelp_rag_agent.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    """
    Calls a locally running Ollama server.

    Start Ollama before using:
        ollama serve
        ollama pull qwen2.5:7b
    """

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model"   : self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream"  : False,
                "options" : {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    @classmethod
    def from_config(cls, cfg: dict) -> "OllamaBackend":
        return cls(
            base_url = cfg["base_url"],
            model    = cfg["model"],
            timeout  = cfg.get("timeout", 120),
        )
