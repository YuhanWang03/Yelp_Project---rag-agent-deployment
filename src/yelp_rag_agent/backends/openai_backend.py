"""
OpenAI backend.

Uses OpenAI's /v1/chat/completions endpoint (same OpenAI-compatible shape as
the Groq backend, different base_url + key). Used as an INDEPENDENT judge for
cross-judge bias validation — a different model family from DeepSeek-V4 so the
self-preference concern (V4 judging V4) can be tested empirically.

Set OPENAI_API_KEY in the environment, or pass api_key= explicitly.

Recommended judge models: gpt-4o-mini (cheap, adequate) or gpt-4o (stronger).
NOTE: reasoning models (o1/o3-series) reject `temperature` and use
`max_completion_tokens` — stick to gpt-4o / gpt-4o-mini here.
"""

import os

from yelp_rag_agent.backends.base import BaseBackend


class OpenAIBackend(BaseBackend):
    """Calls the OpenAI API (chat completions)."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 120):
        self.model    = model
        self.api_key  = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided. Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model"      : self.model,
                "messages"   : [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens" : max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenAIBackend":
        return cls(
            model    = cfg["model"],
            api_key  = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            base_url = cfg.get("base_url", "https://api.openai.com/v1"),
            timeout  = cfg.get("timeout", 120),
        )
