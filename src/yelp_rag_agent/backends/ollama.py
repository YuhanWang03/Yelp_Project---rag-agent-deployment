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
        self._usage   = self._zero_usage()

    @staticmethod
    def _zero_usage() -> dict:
        return {"calls": 0, "input_tokens": 0, "input_cached_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0}

    def reset_usage(self) -> None:
        self._usage = self._zero_usage()

    def get_usage(self) -> dict:
        return dict(self._usage)

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
        data = resp.json()
        self._usage["calls"] += 1
        self._usage["input_tokens"] += int(data.get("prompt_eval_count", 0) or 0)
        self._usage["output_tokens"] += int(data.get("eval_count", 0) or 0)
        return data["message"]["content"]

    @classmethod
    def from_config(cls, cfg: dict) -> "OllamaBackend":
        return cls(
            base_url = cfg["base_url"],
            model    = cfg["model"],
            timeout  = cfg.get("timeout", 120),
        )
