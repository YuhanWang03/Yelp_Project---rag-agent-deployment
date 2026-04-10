"""
Full Agent Pipeline — LangGraph ReAct

Gives the LLM (via the active backend) autonomous control over all 5 tools,
letting it decide the call order based on the question.

Registered tools:
    search_review_chunks_global
    search_review_chunks_by_business
    get_business_stats
    classify_review
    summarize_evidence

Return schema:
    {
        "question"        : str,
        "business_id"     : str | None,
        "final_answer"    : str,
        "tool_calls"      : [{"tool": str, "input": str, "output": str}],
        "steps"           : int,
        "elapsed_seconds" : float
    }
"""

import time
import warnings
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

from yelp_rag_agent.tools.retrieval_tool import (
    search_review_chunks_global,
    search_review_chunks_by_business,
)
from yelp_rag_agent.tools.stats_tool import get_business_stats
from yelp_rag_agent.tools.classifier_tool import classify_review
from yelp_rag_agent.tools.summarizer_tool import summarize_evidence

SYSTEM_PROMPT = """You are a Yelp Business Intelligence Agent with access to a \
database of 50,000 real Yelp reviews.

You have the following tools:
- search_review_chunks_global: search all reviews semantically
- search_review_chunks_by_business: search reviews for a specific business
- get_business_stats: get star distribution for a specific business
- classify_review: predict star rating for a piece of text using a fine-tuned model
- summarize_evidence: synthesise retrieved review chunks into a structured answer

Guidelines:
1. If a business_id is mentioned in the question, use the business-specific tools.
2. Always retrieve evidence BEFORE calling summarize_evidence.
3. Call summarize_evidence as your LAST step to produce the final structured answer.
4. Use classify_review when you want to verify the sentiment of a specific text snippet.
5. Be concise in tool inputs — avoid repeating the full question verbatim.
"""

_agent = None
_agent_backend = None  # track which backend instance was used to build the agent


def _make_chat_model():
    """Create a LangChain chat model appropriate for the active backend."""
    from yelp_rag_agent.tools.summarizer_tool import _backend
    from yelp_rag_agent.backends.ollama import OllamaBackend
    from yelp_rag_agent.backends.lmdeploy import LMDeployBackend

    if _backend is None:
        raise RuntimeError("No backend set. Call set_backend() before running the agent.")

    if isinstance(_backend, OllamaBackend):
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=_backend.base_url, model=_backend.model, temperature=0)

    if isinstance(_backend, LMDeployBackend):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=f"{_backend.base_url}/v1",
            api_key="none",
            model=_backend.model,
            temperature=0,
        )

    raise RuntimeError(f"Unsupported backend type for agent: {type(_backend).__name__}")


def _get_agent():
    global _agent, _agent_backend
    from yelp_rag_agent.tools.summarizer_tool import _backend
    if _agent is None or _backend is not _agent_backend:
        print("[agent_runner] Initialising LangGraph ReAct agent …")
        llm = _make_chat_model()
        tools = [
            search_review_chunks_global,
            search_review_chunks_by_business,
            get_business_stats,
            classify_review,
            summarize_evidence,
        ]
        _agent = create_react_agent(llm, tools)
        _agent_backend = _backend
        print("[agent_runner] Agent ready.")
    return _agent


def _extract_trace(messages: list) -> tuple[str, list[dict]]:
    tool_calls: list[dict] = []
    final_answer = ""

    tool_msg_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            tool_msg_map[msg.tool_call_id] = (
                content[:600] + " …[truncated]" if len(content) > 600 else content
            )

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool"  : tc["name"],
                        "input" : str(tc["args"]),
                        "output": tool_msg_map.get(tc["id"], "(no output)"),
                    })
            elif msg.content:
                final_answer = msg.content

    return final_answer, tool_calls


def run_agent(
    question: str,
    business_id: Optional[str] = None,
    max_iterations: int = 10,
) -> dict:
    full_question = question
    if business_id:
        full_question = f"{question}\n\n[Target business_id: {business_id}]"

    print(f"\n{'='*60}")
    from yelp_rag_agent.tools.summarizer_tool import _backend
    model_label = _backend.model if _backend else "unknown"
    print(f"Agent Pipeline  |  model={model_label}")
    print(f"Question: {question}")
    if business_id:
        print(f"Business ID: {business_id}")
    print(f"{'='*60}")

    agent = _get_agent()

    t0 = time.time()
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_question),
            ]
        },
        config={"recursion_limit": max_iterations * 2},
    )
    elapsed = round(time.time() - t0, 2)

    messages = result["messages"]
    final_answer, tool_calls = _extract_trace(messages)

    print(f"\n  Tool calls ({len(tool_calls)}):")
    for i, tc in enumerate(tool_calls, 1):
        print(f"    [{i}] {tc['tool']}({tc['input'][:60]}…)")
    print(f"  Elapsed: {elapsed}s")

    return {
        "question"       : question,
        "business_id"    : business_id,
        "final_answer"   : final_answer,
        "tool_calls"     : tool_calls,
        "steps"          : len(tool_calls),
        "elapsed_seconds": elapsed,
    }
