"""
HuggingFace Inference API backend.

Uses HF's serverless inference endpoint (OpenAI-compatible).
Suitable for Hugging Face Spaces deployments where no local GPU is available.

Requires a HF_TOKEN with Inference API access.
"""

import os

from yelp_rag_agent.backends.base import BaseBackend


class HFInferenceBackend(BaseBackend):
    """
    Calls the HuggingFace Inference API (serverless).

    The endpoint is OpenAI-compatible:
        https://api-inference.huggingface.co/models/{model}/v1
    """

    def __init__(self, model: str, token: str | None = None, timeout: int = 120):
        self.model   = model
        self.token   = token or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._client = None

    @property
    def _base_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}/v1"

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        import requests

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json={
                "model"      : self.model,
                "messages"   : [{"role": "user", "content": prompt}],
                "temperature": max(temperature, 0.01),  # HF API rejects 0
                "max_tokens" : max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @classmethod
    def from_config(cls, cfg: dict) -> "HFInferenceBackend":
        return cls(
            model   = cfg["model"],
            token   = cfg.get("token") or os.environ.get("HF_TOKEN"),
            timeout = cfg.get("timeout", 120),
        )
