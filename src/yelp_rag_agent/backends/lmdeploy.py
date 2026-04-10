"""
LMDeploy backend — HTTP mode (OpenAI-compatible API).

Start LMDeploy before using (run on Colab A100):
    # fp16
    lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct \\
        --server-port 23333 --backend pytorch

    # AWQ
    lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --server-port 23333 --backend turbomind --model-format awq

NOTE (V1): HTTP mode only. Direct Python API (lower latency) can be
added as V2 once the HTTP path is stable.
"""

import requests

from yelp_rag_agent.backends.base import BaseBackend


class LMDeployBackend(BaseBackend):
    """
    Calls an LMDeploy server that exposes an OpenAI-compatible
    /v1/chat/completions endpoint.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
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
    def from_config(cls, cfg: dict) -> "LMDeployBackend":
        return cls(
            base_url = cfg["base_url"],
            model    = cfg["model"],
            timeout  = cfg.get("timeout", 180),
        )
