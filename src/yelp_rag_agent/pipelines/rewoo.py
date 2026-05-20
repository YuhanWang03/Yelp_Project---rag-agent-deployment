"""
ReWOO Pipeline (Stage E paradigm comparison).

ReWOO — Reasoning WithOut Observation (Xu et al., 2023). The Planner emits
the COMPLETE blueprint of tool calls in one shot, the Workers execute them
in PARALLEL (the model never sees intermediate observations), and the
Solver synthesizes the final answer from all results.

Contrast (same tools, same backend, only the reasoning structure differs):
  - ReAct (agent_runner) : interleaved — re-decides each step from observations
  - Plan-and-Solve       : plan upfront -> SEQUENTIAL exec -> solve
  - ReWOO (this)         : plan upfront -> PARALLEL   exec -> solve
  - Reflection           : base answer -> self-critique -> revise

Same planner and solver as Plan-and-Solve (shared via _paradigm_common), so
the ONLY difference vs Plan-and-Solve is parallel vs sequential execution —
this is what the latency comparison isolates. ReWOO should win wall-clock
time whenever the plan has multiple independent tool calls.

LLM calls: 2 (plan + solve), independent of step count.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from yelp_rag_agent.pipelines._paradigm_common import (
    get_backend, apply_thinking, plan, execute_step,
    stringify_observation, solve,
)

_MAX_WORKERS = 4


def run_rewoo(
    question: str,
    business_id: Optional[str] = None,
    thinking: bool = False,
) -> dict:
    backend = get_backend()
    apply_thinking(backend, thinking)

    print(f"\n{'='*60}")
    print(f"ReWOO  |  model={getattr(backend, 'model', '?')}  |  thinking={thinking}")
    print(f"Question: {question}")
    if business_id:
        print(f"Business ID: {business_id}")
    print(f"{'='*60}")

    t0 = time.time()

    # --- Plan phase (LLM call 1) ---
    steps_plan = plan(question, business_id, backend)
    print(f"  [Plan] {len(steps_plan)} step(s): {[s['tool'] for s in steps_plan]}")

    # --- Worker phase: PARALLEL execution (no LLM) ---
    # This is the defining difference from Plan-and-Solve. Errors are caught
    # inside execute_step, so a failing tool degrades to an error string
    # rather than crashing the batch.
    outputs: list = []
    if steps_plan:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(steps_plan))) as ex:
            outputs = list(ex.map(
                lambda s: execute_step(s["tool"], s["args"]), steps_plan
            ))

    tool_calls: list[dict] = []
    observations: list[str] = []
    for step, out in zip(steps_plan, outputs):
        out_str = stringify_observation(out)
        tool_calls.append({
            "tool"  : step["tool"],
            "input" : str(step["args"]),
            "output": out_str,
        })
        observations.append(f"[{step['tool']}] {out_str}")

    # --- Solve phase (LLM call 2) ---
    final_answer = solve(question, observations, backend)
    elapsed = round(time.time() - t0, 2)
    print(f"  Elapsed: {elapsed}s  |  llm_calls=2  |  parallel workers")

    return {
        "question"       : question,
        "business_id"    : business_id,
        "paradigm"       : "rewoo",
        "thinking"       : thinking,
        "final_answer"   : final_answer,
        "plan"           : steps_plan,
        "tool_calls"     : tool_calls,
        "steps"          : len(tool_calls),
        "llm_calls"      : 2,
        "elapsed_seconds": elapsed,
    }
