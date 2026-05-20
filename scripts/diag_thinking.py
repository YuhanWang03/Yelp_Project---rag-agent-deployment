"""
Diagnose DeepSeek V4 thinking-mode response shape (Stage E2.4 debugging).

The thinking-on smoke run produced empty plans (0 steps) for the
plan/rewoo/reflection paradigms. This sends the EXACT planner prompt in
thinking mode and dumps the full response so we can see whether the JSON
plan lands in `content`, in `reasoning_content`, or gets truncated
(finish_reason == "length"), and how tokens are budgeted.

Usage:
    set DEEPSEEK_API_KEY=...
    python scripts/diag_thinking.py
"""

import os
import requests

from yelp_rag_agent.pipelines._paradigm_common import (
    _PLAN_INSTRUCTION, TOOL_DESCRIPTIONS,
)

QUESTION    = "What do customers complain about most at this business?"
BUSINESS_ID = "gebiRewfieSdtt17PTW6Zg"


def build_plan_prompt() -> str:
    business_ctx = f"\n[Target business_id: {BUSINESS_ID}]"
    return _PLAN_INSTRUCTION.format(
        tools=TOOL_DESCRIPTIONS, question=QUESTION, business_ctx=business_ctx
    )


def call(prompt: str, thinking: bool, max_tokens: int) -> dict:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    payload = {
        "model"     : "deepseek-v4-flash",
        "messages"  : [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "thinking"  : {"type": "enabled" if thinking else "disabled"},
    }
    if not thinking:
        payload["temperature"] = 0.1
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload, timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def dump(label: str, data: dict):
    choice  = data["choices"][0]
    msg     = choice["message"]
    content = msg.get("content") or ""
    reason  = msg.get("reasoning_content") or ""
    usage   = data.get("usage", {})
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"finish_reason     : {choice.get('finish_reason')}")
    print(f"usage             : {usage}")
    print(f"len(content)          = {len(content)}")
    print(f"len(reasoning_content)= {len(reason)}")
    print(f"\n--- content[:800] ---\n{content[:800]}")
    print(f"\n--- reasoning_content[:800] ---\n{reason[:800]}")


def main():
    assert os.environ.get("DEEPSEEK_API_KEY"), "Set DEEPSEEK_API_KEY first."
    prompt = build_plan_prompt()
    print(f"PLAN PROMPT ({len(prompt)} chars):\n{prompt}\n")

    dump("thinking=OFF, max_tokens=512", call(prompt, thinking=False, max_tokens=512))
    dump("thinking=ON,  max_tokens=512", call(prompt, thinking=True,  max_tokens=512))
    dump("thinking=ON,  max_tokens=2048", call(prompt, thinking=True, max_tokens=2048))


if __name__ == "__main__":
    main()
