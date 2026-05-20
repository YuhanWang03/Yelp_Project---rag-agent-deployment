"""
Plan-and-Solve Pipeline (Stage E paradigm comparison).

Plan-and-Solve (Wang et al., 2023): the LLM first devises a COMPLETE plan
of tool calls, then executes them sequentially WITHOUT replanning, then
synthesizes the final answer from all observations.

Contrast (same tools, same backend, only the reasoning structure differs):
  - ReAct (agent_runner): interleaved — re-decides each step from observations
  - Plan-and-Solve (this) : plan upfront -> SEQUENTIAL exec -> solve
  - ReWOO (rewoo)         : plan upfront -> PARALLEL  exec -> solve
  - Reflection (reflection): base answer -> self-critique -> revise

LLM calls: 2 (plan + solve), independent of the number of tool steps.

Return schema (shared across paradigm pipelines):
    {
        "question", "business_id", "paradigm", "thinking",
        "final_answer", "plan", "tool_calls", "steps",
        "llm_calls", "elapsed_seconds"
    }
"""

import time
from typing import Optional

from yelp_rag_agent.pipelines._paradigm_common import (
    get_backend, apply_thinking, plan, execute_step,
    stringify_observation, solve,
)


def run_plan_and_solve(
    question: str,
    business_id: Optional[str] = None,
    thinking: bool = False,
) -> dict:
    backend = get_backend()
    apply_thinking(backend, thinking)

    print(f"\n{'='*60}")
    print(f"Plan-and-Solve  |  model={getattr(backend, 'model', '?')}  |  thinking={thinking}")
    print(f"Question: {question}")
    if business_id:
        print(f"Business ID: {business_id}")
    print(f"{'='*60}")

    t0 = time.time()

    # --- Plan phase (LLM call 1) ---
    steps_plan = plan(question, business_id, backend)
    print(f"  [Plan] {len(steps_plan)} step(s): {[s['tool'] for s in steps_plan]}")

    # --- Execute phase (no LLM) ---
    tool_calls: list[dict] = []
    observations: list[str] = []
    for i, step in enumerate(steps_plan, 1):
        out = execute_step(step["tool"], step["args"])
        out_str = stringify_observation(out)
        print(f"  [Exec {i}] {step['tool']}({str(step['args'])[:50]}…)")
        tool_calls.append({
            "tool"  : step["tool"],
            "input" : str(step["args"]),
            "output": out_str,
        })
        observations.append(f"[{step['tool']}] {out_str}")

    # --- Solve phase (LLM call 2) ---
    final_answer = solve(question, observations, backend)
    elapsed = round(time.time() - t0, 2)
    print(f"  Elapsed: {elapsed}s  |  llm_calls=2")

    return {
        "question"       : question,
        "business_id"    : business_id,
        "paradigm"       : "plan_and_solve",
        "thinking"       : thinking,
        "final_answer"   : final_answer,
        "plan"           : steps_plan,
        "tool_calls"     : tool_calls,
        "steps"          : len(tool_calls),
        "llm_calls"      : 2,
        "elapsed_seconds": elapsed,
    }
