"""
Groq backend.

Uses Groq's OpenAI-compatible /v1/chat/completions endpoint. Groq's LPU
inference is extremely fast (500+ tok/s) and supports native OpenAI
function calling on Llama 3.1/3.3 and Mixtral models — which makes the
Full Agent pipeline actually work, unlike Qwen2.5 + LMDeploy where
LangChain can't parse Qwen's <tool_call> tags.

Free tier requires only a GROQ_API_KEY (no credit card).

Recommended models for tool calling:
    llama-3.3-70b-versatile     — best quality, native tool calling
    llama-3.1-8b-instant        — faster, still supports tools
    mixtral-8x7b-32768          — long context, tools supported
"""

import os

from yelp_rag_agent.backends.base import BaseBackend


class GroqBackend(BaseBackend):
    """Calls the Groq API (OpenAI-compatible)."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.groq.com/openai/v1",
                 timeout: int = 120):
        self.model    = model
        self.api_key  = api_key or os.environ.get("GROQ_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Pass api_key= or set the "
                "GROQ_API_KEY environment variable."
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
    def from_config(cls, cfg: dict) -> "GroqBackend":
        return cls(
            model    = cfg["model"],
            api_key  = cfg.get("api_key") or os.environ.get("GROQ_API_KEY"),
            base_url = cfg.get("base_url", "https://api.groq.com/openai/v1"),
            timeout  = cfg.get("timeout", 120),
        )
