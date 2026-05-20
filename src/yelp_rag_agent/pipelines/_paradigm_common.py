"""
Shared helpers for the Stage E paradigm pipelines
(plan_and_solve, rewoo, reflection).

Keeps tool dispatch, JSON extraction, backend access, and the final
synthesis ("solve") prompt IDENTICAL across paradigms, so the only thing
that varies between experiments is the reasoning structure itself.

All three paradigms drive the LLM through backend.generate() (the plain
string interface) rather than LangChain. That means DeepSeek-V4 thinking
mode is handled automatically by DeepSeekBackend._build_payload — no
LangChain extra_body plumbing needed. Only ReAct (agent_runner) uses
LangChain, because it needs native tool_calls to interleave observations.
"""

import json
import re

from yelp_rag_agent.tools.retrieval_tool import (
    search_review_chunks_global,
    search_review_chunks_by_business,
)
from yelp_rag_agent.tools.stats_tool import get_business_stats

# Tools the planners may schedule. summarize_evidence is intentionally
# excluded: final synthesis is the Solve/answer phase, not a tool — mirrors
# the ReAct agent, which also leaves summarization to the model's reply.
_TOOLS = {
    "search_review_chunks_global"     : search_review_chunks_global,
    "search_review_chunks_by_business": search_review_chunks_by_business,
    "get_business_stats"              : get_business_stats,
}

TOOL_DESCRIPTIONS = """Available tools:
- search_review_chunks_global(query: str, top_k: int): semantic search across ALL reviews. Use when no specific business is targeted.
- search_review_chunks_by_business(business_id: str, query: str, top_k: int): semantic search within ONE business's reviews.
- get_business_stats(business_id: str): star-rating distribution and review count for one business."""


def get_backend():
    """Return the active backend singleton (set via summarizer_tool.set_backend)."""
    from yelp_rag_agent.tools.summarizer_tool import _backend
    if _backend is None:
        raise RuntimeError(
            "No backend set. Call set_backend() before running a paradigm pipeline."
        )
    return _backend


def apply_thinking(backend, thinking: bool) -> None:
    """Set thinking mode if the backend supports it (DeepSeek only); no-op otherwise."""
    if hasattr(backend, "thinking"):
        backend.thinking = thinking


def extract_json(raw: str):
    """Best-effort JSON object/array extraction from an LLM response."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
    if match:
        candidate = (match.group()
                     .replace("‘", "'").replace("’", "'")
                     .replace("“", '"').replace("”", '"')
                     .replace("–", "-").replace("—", "-"))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


_PLAN_INSTRUCTION = """You are planning how to answer a Yelp review question. \
Devise a COMPLETE plan of tool calls to gather the evidence needed, then stop. \
Do NOT answer the question yet — only plan the tool calls.

{tools}

QUESTION:
{question}{business_ctx}

Respond with ONLY a JSON array of steps, each: {{"tool": "<name>", "args": {{...}}}}.
Use top_k=5 for searches. If a business_id is given, prefer the business-specific tools.
Example:
[{{"tool": "get_business_stats", "args": {{"business_id": "abc"}}}},
 {{"tool": "search_review_chunks_by_business", "args": {{"business_id": "abc", "query": "service quality", "top_k": 5}}}}]"""


def plan(question: str, business_id, backend, retries: int = 1) -> list[dict]:
    """Shared planner: one LLM call producing a complete list of {tool, args}
    steps. Used identically by Plan-and-Solve, ReWOO, and Reflection so the
    only variable across paradigms is how the plan is executed/refined.

    Planning is stochastic (especially in thinking mode) and occasionally
    emits unparseable JSON → an empty plan → a "no evidence" answer. Retry
    once on an empty parse so a rare format glitch doesn't contaminate a row.
    Retries only fire when the first parse yields nothing, so well-behaved
    rows are unaffected."""
    business_ctx = f"\n[Target business_id: {business_id}]" if business_id else ""
    prompt = _PLAN_INSTRUCTION.format(
        tools=TOOL_DESCRIPTIONS, question=question, business_ctx=business_ctx
    )
    steps: list[dict] = []
    for _ in range(retries + 1):
        raw = backend.generate(prompt, temperature=0.1, max_tokens=512)
        steps = normalize_plan(extract_json(raw))
        if steps:
            return steps
    return steps


def normalize_plan(parsed) -> list[dict]:
    """Coerce a parsed planner output into a clean list of {tool, args} steps."""
    if isinstance(parsed, dict) and "plan" in parsed:
        parsed = parsed["plan"]
    if not isinstance(parsed, list):
        return []
    steps = []
    for s in parsed:
        if (isinstance(s, dict)
                and isinstance(s.get("tool"), str)
                and isinstance(s.get("args"), dict)):
            steps.append({"tool": s["tool"], "args": s["args"]})
    return steps


def execute_step(tool_name: str, args: dict):
    """Invoke one tool; return its raw output, or an error string on failure."""
    tool = _TOOLS.get(tool_name)
    if tool is None:
        return f"(unknown tool: {tool_name})"
    try:
        return tool.invoke(args)
    except Exception as e:
        return f"(tool error: {type(e).__name__}: {e})"


def stringify_observation(obs) -> str:
    if isinstance(obs, str):
        return obs
    try:
        return json.dumps(obs, ensure_ascii=False)
    except TypeError:
        return str(obs)


_SOLVE_INSTRUCTION = """You are a Yelp Business Intelligence analyst. Using ONLY the \
evidence collected below, answer the question in 3-5 concise bullet points. Cite \
specific star ratings or quotes from the evidence. Do not use outside knowledge.

QUESTION:
{question}

EVIDENCE COLLECTED:
{evidence}

Final answer:"""

MAX_EVIDENCE_CHARS = 6000


def solve(question: str, observations: list[str], backend) -> str:
    """Final synthesis step shared by Plan-and-Solve and ReWOO."""
    evidence = "\n\n".join(observations) if observations else "(no evidence collected)"
    prompt = _SOLVE_INSTRUCTION.format(
        question=question, evidence=evidence[:MAX_EVIDENCE_CHARS]
    )
    return backend.generate(prompt, temperature=0.1, max_tokens=1024).strip()
