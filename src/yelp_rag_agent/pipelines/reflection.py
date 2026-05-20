"""
Reflection Pipeline (Stage E paradigm comparison).

Reflection / Self-Refine (Madaan et al., 2023; Shinn et al., 2023): produce
an initial answer, then have the model CRITIQUE its own answer against the
evidence, then REVISE it. Repeat up to `max_revisions` rounds, stopping early
if the critique finds no issues.

To isolate the value of the reflection loop, the base (plan -> execute ->
initial solve) is exactly the Plan-and-Solve flow. The reflection loop is the
ONLY thing added, so "Reflection vs Plan-and-Solve" measures the marginal
value of self-refinement. The core Stage E research question: does explicit
Reflection still help once the model has built-in thinking (DeepSeek-V4
thinking mode), or is it absorbed by internal CoT?

LLM calls: 2 (plan + initial solve) + up to 2 per revision round
(critique + revise). With max_revisions=1 that is 3-4 calls — the most
expensive paradigm, which the cost table is expected to show.
"""

import time
from typing import Optional

from yelp_rag_agent.pipelines._paradigm_common import (
    get_backend, apply_thinking, plan, execute_step,
    stringify_observation, solve, extract_json,
)

_CRITIQUE_INSTRUCTION = """You are a strict reviewer checking a draft answer to a \
Yelp review question. Judge ONLY against the evidence provided — flag any claim \
not supported by the evidence, any missing aspect the evidence covers, and any \
factual error.

QUESTION:
{question}

EVIDENCE:
{evidence}

DRAFT ANSWER:
{answer}

Respond with ONLY a JSON object:
{{"needs_revision": <true|false>, "critique": "<specific, actionable issues, or 'none'>"}}"""

_REVISE_INSTRUCTION = """Improve the draft answer using the critique. Use ONLY the \
evidence; keep it to 3-5 concise bullet points citing star ratings or quotes. \
Output only the improved answer, no preamble.

QUESTION:
{question}

EVIDENCE:
{evidence}

DRAFT ANSWER:
{answer}

CRITIQUE:
{critique}

Improved answer:"""

_MAX_EVIDENCE_CHARS = 6000


def _critique(question: str, evidence: str, answer: str, backend) -> dict:
    prompt = _CRITIQUE_INSTRUCTION.format(
        question=question, evidence=evidence[:_MAX_EVIDENCE_CHARS], answer=answer
    )
    raw = backend.generate(prompt, temperature=0.1, max_tokens=512)
    parsed = extract_json(raw)
    if isinstance(parsed, dict) and "needs_revision" in parsed:
        return {
            "needs_revision": bool(parsed.get("needs_revision")),
            "critique"      : str(parsed.get("critique", "")),
        }
    # Unparseable critique → assume no revision needed (fail safe, avoids loops).
    return {"needs_revision": False, "critique": "(critique unparseable)"}


def _revise(question: str, evidence: str, answer: str, critique: str, backend) -> str:
    prompt = _REVISE_INSTRUCTION.format(
        question=question, evidence=evidence[:_MAX_EVIDENCE_CHARS],
        answer=answer, critique=critique,
    )
    return backend.generate(prompt, temperature=0.1, max_tokens=1024).strip()


def run_reflection(
    question: str,
    business_id: Optional[str] = None,
    thinking: bool = False,
    max_revisions: int = 1,
) -> dict:
    backend = get_backend()
    apply_thinking(backend, thinking)

    print(f"\n{'='*60}")
    print(f"Reflection  |  model={getattr(backend, 'model', '?')}  |  thinking={thinking}")
    print(f"Question: {question}")
    if business_id:
        print(f"Business ID: {business_id}")
    print(f"{'='*60}")

    t0 = time.time()
    llm_calls = 0

    # --- Base: plan -> execute -> initial solve (= Plan-and-Solve) ---
    steps_plan = plan(question, business_id, backend)
    llm_calls += 1
    print(f"  [Plan] {len(steps_plan)} step(s): {[s['tool'] for s in steps_plan]}")

    tool_calls: list[dict] = []
    observations: list[str] = []
    for step in steps_plan:
        out_str = stringify_observation(execute_step(step["tool"], step["args"]))
        tool_calls.append({
            "tool": step["tool"], "input": str(step["args"]), "output": out_str,
        })
        observations.append(f"[{step['tool']}] {out_str}")
    evidence = "\n\n".join(observations) if observations else "(no evidence collected)"

    answer = solve(question, observations, backend)
    llm_calls += 1
    print(f"  [Initial answer] {len(answer)} chars")

    # --- Reflection loop: critique -> revise ---
    reflection_trace: list[dict] = []
    revisions = 0
    for rnd in range(max_revisions):
        crit = _critique(question, evidence, answer, backend)
        llm_calls += 1
        print(f"  [Critique {rnd+1}] needs_revision={crit['needs_revision']}")
        entry = {"critique": crit["critique"],
                 "needs_revision": crit["needs_revision"],
                 "revised_answer": None}
        if not crit["needs_revision"]:
            reflection_trace.append(entry)
            break
        answer = _revise(question, evidence, answer, crit["critique"], backend)
        llm_calls += 1
        revisions += 1
        entry["revised_answer"] = answer
        reflection_trace.append(entry)
        print(f"  [Revise {rnd+1}] {len(answer)} chars")

    elapsed = round(time.time() - t0, 2)
    print(f"  Elapsed: {elapsed}s  |  llm_calls={llm_calls}  |  revisions={revisions}")

    return {
        "question"        : question,
        "business_id"     : business_id,
        "paradigm"        : "reflection",
        "thinking"        : thinking,
        "final_answer"    : answer,
        "plan"            : steps_plan,
        "tool_calls"      : tool_calls,
        "steps"           : len(tool_calls),
        "revisions"       : revisions,
        "reflection_trace": reflection_trace,
        "llm_calls"       : llm_calls,
        "elapsed_seconds" : elapsed,
    }
