"""
Token usage & API cost for the paradigm study.

Answers "how many tokens / how much $" per pipeline run, for the
paradigm × thinking comparison. Token counts are collected by
DeepSeekBackend.get_usage() (generate-based paradigms) and by the ReAct
LangChain extractor; this module turns them into USD.
"""

# USD per 1M tokens (DeepSeek V4 list prices, 2026). reasoning_content is
# billed as output. NOTE: V4-Pro is 75% off through 2026-05-31 — these are
# the LIST prices; the paradigm comparison is about RELATIVE cost, which is
# invariant to the flat discount.
DEEPSEEK_PRICING = {
    "deepseek-v4-flash": {"input_miss": 0.14, "input_hit": 0.003, "output": 0.28},
    "deepseek-v4-pro"  : {"input_miss": 1.74, "input_hit": 0.015, "output": 3.48},
}

# Normalized usage shape produced by DeepSeekBackend.get_usage() and by the
# ReAct LangChain extractor:
#   {calls, input_tokens, input_cached_tokens, output_tokens, reasoning_tokens}


def compute_cost(usage: dict, model: str) -> dict:
    """USD cost for a normalized usage dict at the given model's list price.

    Returns input/output/total in USD, or Nones if model/usage unknown.
    """
    pricing = DEEPSEEK_PRICING.get(model)
    if not pricing or not usage:
        return {"input_cost_usd": None, "output_cost_usd": None,
                "total_cost_usd": None}
    inp    = usage.get("input_tokens", 0)
    cached = usage.get("input_cached_tokens", 0)
    out    = usage.get("output_tokens", 0)
    miss   = max(inp - cached, 0)
    input_cost  = (miss * pricing["input_miss"] + cached * pricing["input_hit"]) / 1e6
    output_cost = out * pricing["output"] / 1e6
    return {
        "input_cost_usd" : round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd" : round(input_cost + output_cost, 6),
    }
