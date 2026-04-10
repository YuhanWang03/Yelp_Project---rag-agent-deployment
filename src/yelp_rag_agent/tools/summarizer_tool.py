"""
Tool 4 — Evidence Summarizer

Calls the active LLM backend to synthesise retrieved review chunks into
a structured analytical answer.

The backend is injected at startup via set_backend(). All pipeline and
Agent code calls summarize_evidence without knowing which LLM is used.

NOTE (V1 design): uses a module-level singleton (_backend).
This is appropriate for single-process demo and benchmark usage.
Future refactor: pass backend explicitly for multi-backend concurrency.

Output format:
{
    "main_findings": ["Finding 1 ...", "Finding 2 ..."],
    "supporting_evidence": [
        {"claim": "...", "evidence": ["quote 1", "quote 2"]}
    ],
    "uncertainties": ["Caveat ..."]
}
"""

import json
import re
from typing import Optional

from langchain.tools import tool

from yelp_rag_agent.backends.base import BaseBackend

MAX_EVIDENCE_IN_PROMPT = 10
MAX_CHUNK_CHARS        = 400

# ---------------------------------------------------------------------------
# Backend singleton — set at application startup
# ---------------------------------------------------------------------------

_backend: Optional[BaseBackend] = None


def set_backend(backend: BaseBackend) -> None:
    """Inject the LLM backend. Must be called before summarize_evidence."""
    global _backend
    _backend = backend


def get_backend() -> Optional[BaseBackend]:
    return _backend


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: list[dict]) -> str:
    evidence_lines = []
    for i, chunk in enumerate(chunks[:MAX_EVIDENCE_IN_PROMPT], 1):
        stars  = chunk.get("stars", "?")
        text   = chunk.get("chunk_text", chunk.get("text", ""))[:MAX_CHUNK_CHARS]
        biz_id = chunk.get("business_id", "unknown")[:12]
        evidence_lines.append(f"[{i}] ({stars}★, biz:{biz_id}…)\n{text}")
    evidence_block = "\n\n".join(evidence_lines)

    return f"""You are an expert Yelp review analyst.
Your task: answer the question below using ONLY the provided review excerpts as evidence.
Do NOT use prior knowledge. Every claim must be supported by at least one excerpt.

QUESTION:
{question}

REVIEW EXCERPTS:
{evidence_block}

Respond with a JSON object in exactly this format (no extra text outside the JSON):
{{
  "main_findings": [
    "<finding 1>",
    "<finding 2>"
  ],
  "supporting_evidence": [
    {{
      "claim": "<restate a main finding>",
      "evidence": ["<direct quote or paraphrase from excerpt>", "..."]
    }}
  ],
  "uncertainties": [
    "<aspect that is mentioned but not well-supported by the evidence>"
  ]
}}"""


# ---------------------------------------------------------------------------
# Three-layer JSON parser (handles Qwen formatting quirks)
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    """
    Strategy 1 — direct parse.
    Strategy 2 — regex extraction + Unicode normalisation.
    Strategy 3 — extract main_findings array only.
    Final fallback — return raw text as single finding.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")
        candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
        candidate = candidate.replace("\u2013", "-").replace("\u2014", "-")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    findings_match = re.search(
        r'"main_findings"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL
    )
    if findings_match:
        findings = re.findall(r'"([^"]+)"', findings_match.group(1))
        if findings:
            return {
                "main_findings"      : findings,
                "supporting_evidence": [],
                "uncertainties"      : ["Partial parse: only main_findings extracted."],
            }

    return {
        "main_findings"      : [raw.strip()],
        "supporting_evidence": [],
        "uncertainties"      : ["JSON parsing failed; see main_findings for raw output."],
    }


# ---------------------------------------------------------------------------
# Tool 4
# ---------------------------------------------------------------------------

@tool
def summarize_evidence(question: str, evidence_chunks: list) -> dict:
    """
    Synthesise a list of retrieved Yelp review chunks into a structured
    analytical answer using the active LLM backend.

    This tool does NOT search the database — it only analyses the evidence
    passed to it. Always call a retrieval tool first, then pass results here.

    Args:
        question:        The original user question.
        evidence_chunks: List of chunk dicts from search_review_chunks_* tools.

    Returns:
        Dict with keys: main_findings, supporting_evidence, uncertainties.
    """
    if _backend is None:
        raise RuntimeError(
            "LLM backend not initialised. "
            "Call set_backend() before using summarize_evidence."
        )
    if not evidence_chunks:
        return {
            "main_findings"      : ["No evidence provided."],
            "supporting_evidence": [],
            "uncertainties"      : [],
        }

    prompt = _build_prompt(question, evidence_chunks)
    raw    = _backend.generate(prompt, temperature=0.1)
    return _parse_response(raw)
