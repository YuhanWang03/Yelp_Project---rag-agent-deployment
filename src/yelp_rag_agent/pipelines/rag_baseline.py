"""
RAG Baseline Pipeline

Fixed-flow retrieval-augmented generation. No autonomous tool selection —
execution order is hardcoded per question type.

  Flow A — Business question (business_id provided)
    1. get_business_stats
    2. search_review_chunks_by_business
    3. summarize_evidence

  Flow B — Global question (no business_id)
    1. search_review_chunks_global
    2. summarize_evidence

Return schema:
    {
        "question"        : str,
        "mode"            : "business" | "global",
        "business_id"     : str | None,
        "business_stats"  : dict | None,
        "retrieved_chunks": list[dict],
        "synthesis"       : {
            "main_findings"      : list[str],
            "supporting_evidence": list[dict],
            "uncertainties"      : list[str]
        },
        "tools_called"    : list[str],
        "elapsed_seconds" : float
    }
"""

import time
from typing import Optional

from yelp_rag_agent.config import DEFAULT_TOP_K
from yelp_rag_agent.tools.retrieval_tool import (
    search_review_chunks_global,
    search_review_chunks_by_business,
)
from yelp_rag_agent.tools.stats_tool import get_business_stats
from yelp_rag_agent.tools.summarizer_tool import summarize_evidence


def _run_flow_a(question: str, business_id: str, top_k: int) -> dict:
    tools_called: list[str] = []

    print(f"  [Flow A / Step 1] get_business_stats({business_id[:16]}…)")
    stats = get_business_stats.invoke({"business_id": business_id})
    tools_called.append("get_business_stats")

    if stats["review_count"] == 0:
        return {
            "business_stats"  : stats,
            "retrieved_chunks": [],
            "synthesis"       : {
                "main_findings"      : [f"No reviews found for business_id={business_id}"],
                "supporting_evidence": [],
                "uncertainties"      : [],
            },
            "tools_called": tools_called,
        }

    print(f"  [Flow A / Step 2] search_review_chunks_by_business('{question[:50]}…')")
    chunks = search_review_chunks_by_business.invoke(
        {"business_id": business_id, "query": question, "top_k": top_k}
    )
    tools_called.append("search_review_chunks_by_business")

    print(f"  [Flow A / Step 3] summarize_evidence ({len(chunks)} chunks) …")
    synthesis = summarize_evidence.invoke(
        {"question": question, "evidence_chunks": chunks}
    )
    tools_called.append("summarize_evidence")

    return {
        "business_stats"  : stats,
        "retrieved_chunks": chunks,
        "synthesis"       : synthesis,
        "tools_called"    : tools_called,
    }


def _run_flow_b(question: str, top_k: int) -> dict:
    tools_called: list[str] = []

    print(f"  [Flow B / Step 1] search_review_chunks_global('{question[:50]}…')")
    chunks = search_review_chunks_global.invoke({"query": question, "top_k": top_k})
    tools_called.append("search_review_chunks_global")

    print(f"  [Flow B / Step 2] summarize_evidence ({len(chunks)} chunks) …")
    synthesis = summarize_evidence.invoke(
        {"question": question, "evidence_chunks": chunks}
    )
    tools_called.append("summarize_evidence")

    return {
        "business_stats"  : None,
        "retrieved_chunks": chunks,
        "synthesis"       : synthesis,
        "tools_called"    : tools_called,
    }


def run_rag_pipeline(
    question: str,
    business_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    mode = "business" if business_id else "global"
    print(f"\n{'='*60}")
    print(f"RAG Pipeline  |  mode={mode}  |  top_k={top_k}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    t0 = time.time()
    flow_result = _run_flow_a(question, business_id, top_k) if business_id \
                  else _run_flow_b(question, top_k)
    elapsed = round(time.time() - t0, 2)

    return {
        "question"        : question,
        "mode"            : mode,
        "business_id"     : business_id,
        "business_stats"  : flow_result["business_stats"],
        "retrieved_chunks": flow_result["retrieved_chunks"],
        "synthesis"       : flow_result["synthesis"],
        "tools_called"    : flow_result["tools_called"],
        "elapsed_seconds" : elapsed,
    }
