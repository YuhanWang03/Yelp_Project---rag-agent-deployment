"""
DeepSeek backend (V4 series).

Uses DeepSeek's OpenAI-compatible /v1/chat/completions endpoint. V4 was
released 2026-04-24 with two production models:

    deepseek-v4-flash   — 284B / 13B active MoE, $0.14/$0.28 per 1M tok
    deepseek-v4-pro     — 1.6T / 49B active MoE, $1.74/$3.48 per 1M tok
                          (75% discount through 2026-05-31 → $0.44/$0.87)

Both support a 1M-token context, native OpenAI-style `tool_calls` JSON
(so no Qwen-style <tool_call> XML parsing bug), and a dual Thinking /
Non-Thinking mode toggle.

Set DEEPSEEK_API_KEY in the environment, or pass api_key= explicitly.

Thinking mode is toggled via the nested field `thinking: {type: enabled|
disabled}` (per DeepSeek V4 docs). IMPORTANT: V4 defaults thinking to
ENABLED, so non-thinking runs MUST send `disabled` explicitly — omitting
the field silently leaves thinking on, which would corrupt the
thinking-off arm of the paradigm experiment. When thinking is on,
sampling params (temperature/top_p) are ignored by the API, and the CoT
is returned in a separate `reasoning_content` field.
"""

import os

from yelp_rag_agent.backends.base import BaseBackend

# Extra token budget added to the caller's max_tokens when thinking is on,
# to cover reasoning_content (which shares the max_tokens budget and is
# emitted before the answer). Empirically reasoning runs ~0.8-3K tokens for
# this task; 8192 leaves comfortable headroom so content is never truncated.
_THINKING_REASONING_HEADROOM = 8192


class DeepSeekBackend(BaseBackend):
    """Calls the DeepSeek API (OpenAI-compatible)."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 thinking: bool = False, reasoning_effort: str | None = None,
                 timeout: int = 180):
        self.model            = model
        self.api_key          = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url         = base_url.rstrip("/")
        self.thinking         = thinking
        self.reasoning_effort = reasoning_effort
        self.timeout          = timeout
        self._usage           = self._zero_usage()
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not provided. Pass api_key= or set the "
                "DEEPSEEK_API_KEY environment variable."
            )

    # --- Token-usage accounting (normalized shape shared with the eval) ---
    @staticmethod
    def _zero_usage() -> dict:
        return {"calls": 0, "input_tokens": 0, "input_cached_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0}

    def reset_usage(self) -> None:
        """Zero the usage accumulator (call before a pipeline run to scope it)."""
        self._usage = self._zero_usage()

    def get_usage(self) -> dict:
        """Return a copy of accumulated usage since the last reset_usage()."""
        return dict(self._usage)

    def _accumulate_usage(self, usage: dict) -> None:
        if not usage:
            return
        self._usage["calls"]               += 1
        self._usage["input_tokens"]        += usage.get("prompt_tokens", 0)
        self._usage["input_cached_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
        self._usage["output_tokens"]       += usage.get("completion_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self._usage["reasoning_tokens"]    += details.get("reasoning_tokens", 0)

    def _build_payload(self, prompt: str, temperature: float,
                       max_tokens: int) -> dict:
        payload = {
            "model"   : self.model,
            "messages": [{"role": "user", "content": prompt}],
            # V4 defaults thinking to ON — always send explicit state so the
            # thinking-off experiment arm is truly off.
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }
        if self.thinking:
            # In thinking mode max_tokens is a SHARED budget for
            # reasoning_content + content, and reasoning is emitted first.
            # Without headroom, reasoning consumes the whole budget and
            # content comes back empty (finish_reason="length"). Add a
            # generous reasoning allowance ON TOP of the caller's content
            # budget so the answer always has room.
            payload["max_tokens"] = max_tokens + _THINKING_REASONING_HEADROOM
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["max_tokens"] = max_tokens
            # Sampling params are ignored in thinking mode; only meaningful off.
            payload["temperature"] = temperature
        return payload

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._build_payload(prompt, temperature, max_tokens),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._accumulate_usage(data.get("usage", {}))
        return data["choices"][0]["message"]["content"]

    @classmethod
    def from_config(cls, cfg: dict) -> "DeepSeekBackend":
        return cls(
            model    = cfg["model"],
            api_key  = cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url = cfg.get("base_url", "https://api.deepseek.com/v1"),
            thinking = cfg.get("thinking", False),
            timeout  = cfg.get("timeout", 180),
        )
