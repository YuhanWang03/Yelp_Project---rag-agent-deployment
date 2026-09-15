"""Comparison-first Gradio UI and primary application entry point.

The original single-system demo is preserved in ``legacy_app.py``.  This page
runs the same question through several systems and presents their answers and
execution metrics side by side.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import gradio as gr

# Reuse the legacy application's bootstrap, catalogue, backend configuration,
# and pipelines. Importing it does not launch its UI because its launch block
# is guarded by ``if __name__ == "__main__"``.
import legacy_app as legacy


SYSTEMS = [
    "Direct LLM",
    "RAG Baseline",
    "ReAct",
    "Plan-and-Solve",
    "ReWOO",
    "Reflection",
]

RETRIEVAL_SYSTEMS = set(SYSTEMS) - {"Direct LLM"}
_retrieval_warmed = False

DEEPSEEK_MODELS = {
    "DeepSeek V4 Flash": "configs/deepseek_v4_flash.yaml",
    "DeepSeek V4 Pro": "configs/deepseek_v4_pro.yaml",
}
DEFAULT_DEEPSEEK_MODEL = "DeepSeek V4 Flash"

SYSTEM_META = {
    "Direct LLM": ("CONTROL", "No retrieval", "🟠"),
    "RAG Baseline": ("FIXED FLOW", "Stats → Search → Summarize", "🟢"),
    "ReAct": ("AGENT", "Reason ↔ Act", "🔵"),
    "Plan-and-Solve": ("PLANNER", "Plan → Sequential tools → Solve", "🟣"),
    "ReWOO": ("PLANNER", "Plan → Parallel tools → Solve", "🩵"),
    "Reflection": ("SELF-REFINE", "Answer → Critique → Revise", "🩷"),
}

EMPTY_CARD = "<div class='empty-card'>Select this system and run a comparison.</div>"


def _safe_parse(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return legacy._parse_tool_output(value)


def _extract_chunks(tool_calls: list[dict]) -> list[dict]:
    for call in tool_calls:
        if "search_review_chunks" not in call.get("tool", ""):
            continue
        parsed = _safe_parse(call.get("output"))
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    return []


def _activate_model(model_label: str):
    """Create a fresh non-thinking DeepSeek backend for one comparison run."""
    config_path = DEEPSEEK_MODELS.get(model_label)
    if config_path is None:
        raise ValueError(f"Unsupported model: {model_label}")
    backend = legacy.load_backend(config_path)
    # DeepSeek defaults to thinking in some API modes, so keep this explicit.
    backend.thinking = False
    legacy.backend = backend
    legacy.set_backend(backend)
    return backend


def _run_one(system: str, question: str, business_id: str | None) -> dict:
    """Run one existing pipeline and normalize its result for the new UI."""
    backend = legacy.backend
    if hasattr(backend, "reset_usage"):
        backend.reset_usage()

    if system == "Direct LLM":
        context = f" about Yelp business ID {business_id}" if business_id else ""
        prompt = (
            "You are a Yelp review analyst. "
            f"Answer this question{context}:\n\n{question}\n\n"
            "Answer based only on your general knowledge."
        )
        started = time.time()
        answer = backend.generate(prompt, temperature=0.1)
        raw = {
            "final_answer": answer,
            "tool_calls": [],
            "llm_calls": 1,
            "elapsed_seconds": round(time.time() - started, 2),
        }
    elif system == "RAG Baseline":
        result = legacy.run_rag_pipeline(question, business_id=business_id)
        synthesis = result.get("synthesis", {})
        findings = synthesis.get("main_findings", [])
        uncertainty = synthesis.get("uncertainties", [])
        answer_parts = [f"- {item}" for item in findings]
        if uncertainty:
            answer_parts += ["\n**Uncertainties**"] + [f"- {item}" for item in uncertainty]
        chunks = result.get("retrieved_chunks", [])
        stats = result.get("business_stats")
        tool_details = []
        for name in result.get("tools_called", []):
            if name == "get_business_stats":
                tool_input = {"business_id": business_id}
                tool_output = stats
            elif name == "search_review_chunks_by_business":
                tool_input = {"business_id": business_id, "query": question}
                tool_output = f"{len(chunks)} review chunks returned"
            elif name == "search_review_chunks_global":
                tool_input = {"query": question}
                tool_output = f"{len(chunks)} review chunks returned"
            else:
                tool_input = {"question": question}
                tool_output = synthesis
            tool_details.append({
                "tool": name,
                "input": json.dumps(tool_input, ensure_ascii=False),
                "output": json.dumps(tool_output, ensure_ascii=False)
                if not isinstance(tool_output, str) else tool_output,
            })
        raw = {
            **result,
            "final_answer": "\n".join(answer_parts) or "No answer returned.",
            "tool_calls": tool_details,
            "llm_calls": 1,
        }
        raw["chunks"] = chunks
    elif system == "ReAct":
        raw = legacy.run_agent(
            question, business_id=business_id, max_iterations=6, thinking=False
        )
    elif system in legacy.PARADIGM_RUNNERS:
        raw = legacy.PARADIGM_RUNNERS[system](
            question, business_id=business_id, thinking=False
        )
    else:
        raise ValueError(f"Unknown system: {system}")

    backend_usage = backend.get_usage() if hasattr(backend, "get_usage") else {}
    raw_usage = raw.get("token_usage", {})
    # ReAct talks to the model through LangChain rather than backend.generate,
    # so its usage lives on the pipeline result.  Plain pipelines use the
    # backend accumulator.  Prefer whichever source contains actual counts.
    backend_total = backend_usage.get("input_tokens", 0) + backend_usage.get("output_tokens", 0)
    raw_total = raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)
    usage = raw_usage if raw_total and not backend_total else backend_usage
    tool_calls = raw.get("tool_calls", [])
    chunks = raw.get("chunks") or _extract_chunks(tool_calls)
    return {
        "system": system,
        "answer": raw.get("final_answer", "No answer returned."),
        "elapsed": raw.get("elapsed_seconds", 0),
        "llm_calls": raw.get("llm_calls", usage.get("calls", 0)),
        "tool_calls": tool_calls,
        "tool_count": len(tool_calls),
        "chunks": chunks,
        "usage": usage,
        "model": backend.model,
        "revisions": raw.get("revisions"),
    }


def _slug(system: str) -> str:
    return system.lower().replace(" ", "-").replace("-and-", "-")


def _metric(value, label: str, target: str | None = None) -> str:
    content = f"<b>{value}</b><span>{label}</span>"
    if target and value:
        return f"<a class='metric-box metric-link' href='#{target}'>{content}</a>"
    return f"<div class='metric-box'>{content}</div>"


def _detail_modals(result: dict, system: str) -> str:
    slug = _slug(system)
    tool_rows = []
    for index, call in enumerate(result.get("tool_calls", []), 1):
        name = html.escape(str(call.get("tool", "unknown")))
        tool_input = html.escape(str(call.get("input", ""))[:800] or "Pipeline-managed input")
        output = html.escape(str(call.get("output", ""))[:1200] or "No output captured")
        tool_rows.append(
            "<div class='modal-item'>"
            f"<div class='modal-item-title'><span>{index}</span><code>{name}</code></div>"
            f"<small>INPUT</small><pre>{tool_input}</pre>"
            f"<small>OUTPUT</small><pre>{output}</pre></div>"
        )

    evidence_rows = []
    for index, chunk in enumerate(result.get("chunks", [])[:12], 1):
        evidence_rows.append(
            "<div class='modal-item evidence-modal-item'>"
            f"<div class='modal-item-title'><span>E{index}</span>"
            f"<b>★ {html.escape(str(chunk.get('stars', '?')))}</b>"
            f"<em>Similarity {html.escape(str(chunk.get('similarity', '—')))}</em></div>"
            f"<p>{html.escape(str(chunk.get('chunk_text', '')))}</p></div>"
        )

    def modal(modal_id: str, title: str, subtitle: str, content: str) -> str:
        return (
            f"<div id='{modal_id}' class='metric-modal'>"
            "<a class='modal-backdrop' href='#' aria-label='Close'></a>"
            "<section class='modal-window'>"
            f"<header><div><span>{html.escape(system)}</span><h2>{title}</h2>"
            f"<p>{subtitle}</p></div><a class='modal-close' href='#' aria-label='Close'>×</a></header>"
            f"<div class='modal-scroll'>{content}</div></section></div>"
        )

    parts = []
    if tool_rows:
        parts.append(modal(
            f"tools-{slug}", "Tool calls",
            f"{len(tool_rows)} calls in execution order", "".join(tool_rows),
        ))
    if evidence_rows:
        parts.append(modal(
            f"evidence-{slug}", "Retrieved evidence",
            f"{len(result.get('chunks', []))} review excerpts returned", "".join(evidence_rows),
        ))
    return "".join(parts)


def _card(result: dict | None, system: str, state: str = "ready") -> str:
    tag, flow, icon = SYSTEM_META[system]
    modals = ""
    if state == "waiting":
        body = "<div class='waiting-card'><span></span> Waiting to run…</div>"
        metrics = ""
    elif state == "running":
        body = "<div class='waiting-card'><span></span> Running this pipeline…</div>"
        metrics = ""
    elif state == "error":
        body = f"<div class='error-card'>{html.escape(str(result.get('error', 'Unknown error')))}</div>"
        metrics = ""
    elif result:
        usage = result.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        tool_count = result["tool_count"]
        evidence_count = len(result.get("chunks", []))
        slug = _slug(system)
        metrics = (
            "<div class='metric-strip'>"
            + _metric(result["elapsed"], "seconds")
            + _metric(result["llm_calls"], "LLM calls")
            + _metric(tool_count, "tool calls", f"tools-{slug}" if tool_count else None)
            + _metric(tokens or "—", "tokens")
            + _metric(evidence_count, "evidence", f"evidence-{slug}" if evidence_count else None)
            + "</div>"
        )
        body = result["answer"]
        modals = _detail_modals(result, system)
    else:
        body, metrics = EMPTY_CARD, ""
        modals = ""

    return (
        f"<div class='system-head'><div><span class='system-icon'>{icon}</span>"
        f"<b>{system}</b><span class='system-tag'>{tag}</span></div>"
        # Keep generated Markdown and modal HTML in separate blocks.  Without
        # this boundary the Markdown renderer may nest the modal in the final
        # answer paragraph and move later evidence items back into the card.
        f"<small>{flow}</small></div>{metrics}\n\n{body}\n\n{modals}"
    )


def _evidence_markdown(results: list[dict]) -> str:
    unique: dict[tuple, dict] = {}
    sources: dict[tuple, set[str]] = {}
    for result in results:
        for chunk in result.get("chunks", []):
            key = (chunk.get("review_id"), chunk.get("chunk_idx"))
            unique[key] = chunk
            sources.setdefault(key, set()).add(result["system"])
    if not unique:
        return (
            "<div class='empty-detail'><b>No retrieved evidence</b>"
            "<span>The selected systems did not return review excerpts.</span></div>"
        )

    retrieving_systems = sorted({name for names in sources.values() for name in names})
    cards = []
    for index, (key, chunk) in enumerate(unique.items(), 1):
        text = html.escape(str(chunk.get("chunk_text", "")).strip())
        source_badges = "".join(
            f"<span>{html.escape(name)}</span>" for name in sorted(sources[key])
        )
        cards.append(
            "<article class='evidence-card'>"
            "<div class='evidence-card-head'>"
            f"<b>E{index}</b>"
            f"<span class='star-badge'>★ {html.escape(str(chunk.get('stars', '?')))}</span>"
            f"<span class='similarity-badge'>Similarity {html.escape(str(chunk.get('similarity', '—')))}</span>"
            "</div>"
            f"<div class='source-badges'><small>USED BY</small>{source_badges}</div>"
            f"<div class='evidence-text-scroll'><blockquote>{text}</blockquote></div>"
            "</article>"
        )
    return (
        "<section class='detail-section'>"
        "<div class='detail-heading'><div><span class='detail-kicker'>RETRIEVAL EVIDENCE</span>"
        f"<h2>Shared evidence pool</h2><p>{len(unique)} unique excerpts collected across "
        f"{len(retrieving_systems)} systems. All excerpts are included.</p></div>"
        f"<div class='detail-count'>{len(unique)}<span>EXCERPTS</span></div></div>"
        f"<div class='evidence-pool-scroll'><div class='evidence-grid'>{''.join(cards)}"
        "</div></div></section>"
    )


def _trace_markdown(results: list[dict]) -> str:
    if not results:
        return "<div class='empty-detail'><b>No execution traces yet</b><span>Run a comparison to inspect tool calls.</span></div>"
    system_cards = []
    for result in results:
        system = result["system"]
        tag, flow, icon = SYSTEM_META[system]
        steps = []
        if not result["tool_calls"]:
            steps.append("<div class='no-tools'>No tools used — direct generation only.</div>")
        else:
            for index, call in enumerate(result["tool_calls"], 1):
                tool = html.escape(str(call.get("tool", "unknown")))
                input_text = html.escape(str(call.get("input", ""))[:240] or "Pipeline-managed input")
                steps.append(
                    "<div class='trace-step'>"
                    f"<span class='step-number'>{index}</span>"
                    f"<div><code>{tool}</code><p>{input_text}</p></div>"
                    "</div>"
                )
        system_cards.append(
            "<article class='trace-card'>"
            f"<div class='trace-card-head'><div><span>{icon}</span><b>{html.escape(system)}</b></div>"
            f"<span>{result['tool_count']} CALL{'S' if result['tool_count'] != 1 else ''}</span></div>"
            f"<small class='trace-flow'>{html.escape(flow)}</small>"
            f"<div class='trace-steps'>{''.join(steps)}</div></article>"
        )
    return (
        "<section class='detail-section'><div class='detail-heading'><div>"
        "<span class='detail-kicker'>EXECUTION PATHS</span><h2>Tool traces by system</h2>"
        "<p>Inputs are abbreviated; steps remain in execution order.</p>"
        f"</div></div><div class='trace-grid'>{''.join(system_cards)}</div></section>"
    )


def _summary_rows(results: list[dict]) -> list[list]:
    return [
        [
            r["system"],
            r["model"],
            r["elapsed"],
            r["llm_calls"],
            r["tool_count"],
            "Yes" if r["chunks"] else "No",
        ]
        for r in results
    ]


def _warmup_retrieval() -> float:
    """Load and exercise retrieval once, outside every system timer.

    Loading the FAISS files and SentenceTransformer is intentionally lazy in
    the original application.  Without this warm-up, whichever retrieval-based
    system runs first is charged the entire process cold-start cost.
    """
    global _retrieval_warmed
    if _retrieval_warmed:
        return 0.0

    from yelp_rag_agent.tools import retrieval_tool

    started = time.time()
    retrieval_tool._load_store()
    # Loading the model is not enough: the first encode also initializes the
    # PyTorch/CUDA execution path.  Exercise that path before formal timing.
    retrieval_tool._encode_query("retrieval warm-up")
    _retrieval_warmed = True
    return round(time.time() - started, 2)


def compare(question: str, business_id: str, selected: list[str], model_label: str):
    question = (question or "").strip()
    business_id = (business_id or "").strip() or None
    selected = selected or []
    if not question:
        yield ["Please enter a question."] + [_card(None, s) for s in SYSTEMS] + [[], "", ""]
        return
    if len(selected) < 2:
        yield ["Select at least two systems for a meaningful comparison."] + [
            _card(None, s) for s in SYSTEMS
        ] + [[], "", ""]
        return

    try:
        _activate_model(model_label)
    except Exception as exc:
        message = f"Model initialization failed: {type(exc).__name__}: {exc}"
        yield [message] + [_card(None, s) for s in SYSTEMS] + [[], "", ""]
        return

    cards = {
        system: _card(None, system, "waiting") if system in selected else _card(None, system)
        for system in SYSTEMS
    }
    results: list[dict] = []
    warmup_seconds = 0.0
    if any(system in RETRIEVAL_SYSTEMS for system in selected):
        yield [f"{model_label} · warming up retrieval and embedding model…"] + [
            cards[s] for s in SYSTEMS
        ] + [[], "", ""]
        try:
            warmup_seconds = _warmup_retrieval()
        except Exception as exc:
            message = f"Retrieval warm-up failed: {type(exc).__name__}: {exc}"
            yield [message] + [cards[s] for s in SYSTEMS] + [[], "", ""]
            return
        warmup_label = (
            f"Retrieval ready · cold-start warm-up {warmup_seconds}s (excluded from results)"
            if warmup_seconds else
            "Retrieval already warm · starting timed comparison"
        )
        yield [warmup_label] + [cards[s] for s in SYSTEMS] + [[], "", ""]
    else:
        yield ["No retrieval warm-up needed · Direct LLM only"] + [
            cards[s] for s in SYSTEMS
        ] + [[], "", ""]

    # Deliberately sequential: the existing app has a singleton backend and a
    # module-level latest-evidence cache, so parallel execution is not isolated.
    for index, system in enumerate(selected, 1):
        cards[system] = _card(None, system, "running")
        yield [f"Running {index}/{len(selected)} · {system}"] + [cards[s] for s in SYSTEMS] + [
            _summary_rows(results), _evidence_markdown(results), _trace_markdown(results)
        ]
        try:
            result = _run_one(system, question, business_id)
            results.append(result)
            cards[system] = _card(result, system)
        except Exception as exc:
            cards[system] = _card({"error": f"{type(exc).__name__}: {exc}"}, system, "error")

        yield [f"Completed {index}/{len(selected)} systems"] + [cards[s] for s in SYSTEMS] + [
            _summary_rows(results), _evidence_markdown(results), _trace_markdown(results)
        ]

    warmup_note = (
        f" · retrieval warm-up {warmup_seconds}s excluded" if warmup_seconds else ""
    )
    yield [f"Comparison complete · {model_label} · {len(results)}/{len(selected)} succeeded{warmup_note}"] + [
        cards[s] for s in SYSTEMS
    ] + [_summary_rows(results), _evidence_markdown(results), _trace_markdown(results)]


def _benchmark_rows() -> list[list]:
    path = Path("results/paradigm_study.json")
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("system", "unknown"), []).append(row)
    output = []
    labels = {
        "direct_llm": "Direct LLM", "rag_baseline": "RAG Baseline",
        "full_agent": "ReAct", "plan_and_solve": "Plan-and-Solve",
        "rewoo": "ReWOO", "reflection": "Reflection",
    }
    for name, group in groups.items():
        latency = [r.get("elapsed_seconds") for r in group if r.get("elapsed_seconds") is not None]
        costs = [r.get("cost_usd") for r in group if r.get("cost_usd") is not None]
        quality = []
        for row in group:
            scores = [row.get(f"score_{k}") for k in ("correctness", "evidence", "groundedness", "tool_use")]
            if all(v is not None for v in scores):
                quality.append(sum(scores))
        output.append([
            labels.get(name, name), len(group),
            round(sum(quality) / len(quality), 2) if quality else None,
            round(sum(latency) / len(latency), 2) if latency else None,
            round(sum(costs) / len(costs), 6) if costs else None,
        ])
    return sorted(output, key=lambda row: SYSTEMS.index(row[0]) if row[0] in SYSTEMS else 99)


CSS = """
.gradio-container { max-width: none !important; width: 100% !important;
  padding-left: clamp(18px, 2vw, 38px) !important;
  padding-right: clamp(18px, 2vw, 38px) !important;
  background: #f6f7fb !important; }
.gradio-container main.app, .gradio-container > main { max-width: none !important;
  width: 100% !important; margin-left: 0 !important; margin-right: 0 !important;
  padding-left: 0 !important; padding-right: 0 !important; }
.hero { padding: 26px 30px; border-radius: 22px; color: white;
  background: radial-gradient(circle at 90% 0%, #846ef7 0, transparent 34%),
              linear-gradient(125deg, #161b38 0%, #30376e 100%); }
.hero h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -.8px; color: #ffffff !important; }
.hero p { color: #d9ddff; margin: 0; font-size: 15px; }
.eyebrow { color: #aeb8ff; font-size: 12px; font-weight: 800; letter-spacing: 1.5px; }
.query-panel { background: white; border: 1px solid #e4e7f0; border-radius: 18px; padding: 18px; }
.examples-panel { background: white; border: 1px solid #e4e7f0; border-radius: 18px;
  padding: 12px 14px !important; min-width: 0; }
.examples-panel .table-wrap, .examples-panel .dataset { max-height: 294px !important;
  overflow: auto !important; border-radius: 10px; }
.examples-panel table { font-size: 11px !important; }
.examples-panel th { position: sticky !important; top: 0; z-index: 2;
  background: #f7f8fc !important; }
.examples-panel td { white-space: normal !important; line-height: 1.35 !important; }
.section-title h2 { margin-bottom: 2px; }
.section-title p { color: #687087; margin-top: 0; }
.block.system-card { height: 520px !important; min-height: 520px !important; max-height: 520px !important;
  overflow: hidden !important; background: white;
  border: 1px solid #e2e5ee !important; border-radius: 18px !important;
  padding: 0 !important; box-shadow: 0 5px 20px rgba(30,38,70,.05); }
.block.system-card > div:has(> .prose.system-card) { width:100%; height:100% !important;
  min-height:0 !important; overflow:hidden !important; }
.block.system-card .prose.system-card { box-sizing:border-box; width:100%; height:100% !important;
  min-height:0 !important; max-height:none !important; overflow-y:auto !important;
  overflow-x:hidden !important; padding:19px !important; background:transparent !important;
  border:0 !important; border-radius:0 !important; box-shadow:none !important;
  scrollbar-gutter:stable; }
.block.system-card .prose.system-card::-webkit-scrollbar { width:7px; }
.block.system-card .prose.system-card::-webkit-scrollbar-track { background:transparent; }
.block.system-card .prose.system-card::-webkit-scrollbar-thumb {
  background:#d5d8e6; border-radius:999px; }
.system-head { border-bottom: 1px solid #eceef5; margin: -3px 0 14px; padding-bottom: 12px; }
.system-head > div { display: flex; align-items: center; gap: 8px; font-size: 17px; }
.system-head small { display: block; color: #788097; margin-top: 5px; }
.system-icon { font-size: 15px; }
.system-tag { font-size: 9px; letter-spacing: .8px; color: #525d86; background: #eef0ff;
  padding: 4px 7px; border-radius: 999px; margin-left: auto; }
.metric-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 7px; margin: 7px 0 18px; }
.metric-box { display:block; background:#f5f6fa; border-radius:10px; padding:8px 4px;
  text-align:center; text-decoration:none !important; border:1px solid transparent; }
.metric-box b { display:block; color:#252b49; font-size:14px; }
.metric-box span { display:block; color:#8a90a3; font-size:8px; text-transform:uppercase; }
.metric-link { cursor:pointer; background:#efefff; border-color:#dedcff; transition:.15s ease; }
.metric-link:hover { background:#e3e2ff; border-color:#aaa6fb; transform:translateY(-1px); }
.metric-link b, .metric-link span { color:#5652cf; }
.metric-modal { display:none; position:fixed; inset:0; z-index:1000; align-items:center;
  justify-content:center; padding:24px; }
.metric-modal:target { display:flex; }
.modal-backdrop { position:absolute; inset:0; background:rgba(17,22,45,.58);
  backdrop-filter:blur(3px); }
.modal-window { position:relative; z-index:1; width:min(720px,92vw); height:min(760px,78vh);
  max-height:78vh; min-height:0;
  display:flex; flex-direction:column; background:white; border:1px solid #e1e4ed;
  border-radius:18px; box-shadow:0 24px 80px rgba(12,18,48,.3); overflow:hidden; }
.modal-window > header { display:flex; align-items:flex-start; justify-content:space-between;
  padding:18px 20px 14px; border-bottom:1px solid #e9ebf2; }
.modal-window header span { color:#6560dc; font-size:10px; font-weight:800;
  letter-spacing:.8px; text-transform:uppercase; }
.modal-window header h2 { color:#242a47; font-size:20px; margin:2px 0; }
.modal-window header p { color:#858b9e; font-size:11px; margin:0; }
.modal-close { width:31px; height:31px; display:flex; align-items:center; justify-content:center;
  border-radius:9px; background:#f1f2f7; color:#555d73 !important; text-decoration:none !important;
  font-size:20px; }
.modal-scroll { flex:1 1 auto; min-height:0; overflow-y:scroll; overflow-x:hidden;
  overscroll-behavior:contain; padding:14px 20px 20px; scrollbar-gutter:stable; }
.modal-scroll::-webkit-scrollbar { width:8px; }
.modal-scroll::-webkit-scrollbar-track { background:#f1f2f7; }
.modal-scroll::-webkit-scrollbar-thumb { background:#b8bdd0; border-radius:999px; }
.modal-item { background:#f8f9fc; border:1px solid #e8eaf1; border-radius:12px;
  padding:12px; margin-bottom:10px; }
.modal-item-title { display:flex; align-items:center; gap:8px; margin-bottom:9px; }
.modal-item-title > span { min-width:23px; height:23px; display:flex; align-items:center;
  justify-content:center; border-radius:7px; background:#625ee0; color:white; font-size:9px; }
.modal-item-title code { color:#4f4bb7; font-size:11px; overflow-wrap:anywhere; }
.modal-item > small { color:#999fb0; font-size:8px; letter-spacing:.8px; }
.modal-item pre { margin:3px 0 9px; padding:8px; border-radius:8px; background:white;
  color:#555d72; font-size:10px; white-space:pre-wrap; overflow-wrap:anywhere; }
.evidence-modal-item .modal-item-title b { color:#896817; font-size:11px; }
.evidence-modal-item .modal-item-title em { margin-left:auto; color:#858b9c; font-size:9px; }
.evidence-modal-item p { color:#454c61; line-height:1.55; font-size:12px; margin:0; }
.empty-card { color: #9ba0af; display:flex; align-items:center; justify-content:center;
  min-height:220px; text-align:center; }
.waiting-card { color: #69718a; min-height: 220px; display:flex; align-items:center; justify-content:center; gap:9px; }
.waiting-card span { width:9px; height:9px; border-radius:50%; background:#665eea; animation:pulse 1s infinite; }
.error-card { color: #a93232; background:#fff2f2; border-radius:10px; padding:12px; }
.detail-output { background: transparent !important; border: 0 !important; padding: 0 !important; }
.detail-section { margin: 18px 0 30px; }
.detail-heading { display:flex; justify-content:space-between; align-items:flex-end; gap:20px;
  margin-bottom:16px; }
.detail-heading h2 { margin:3px 0 2px; color:#202640; font-size:22px; }
.detail-heading p { margin:0; color:#747c91; font-size:13px; }
.detail-kicker { color:#6964e8; font-size:10px; font-weight:800; letter-spacing:1.25px; }
.detail-count { min-width:82px; background:#252c57; color:white; border-radius:14px;
  padding:10px 14px; text-align:center; font-size:20px; font-weight:800; }
.detail-count span { display:block; color:#b9c0e9; font-size:8px; letter-spacing:1px; }
.evidence-pool-scroll { height:620px; overflow-y:auto; overflow-x:hidden; padding:2px 10px 8px 2px;
  scrollbar-gutter:stable; overscroll-behavior:contain; }
.evidence-pool-scroll::-webkit-scrollbar { width:8px; }
.evidence-pool-scroll::-webkit-scrollbar-track { background:#eceef5; border-radius:999px; }
.evidence-pool-scroll::-webkit-scrollbar-thumb { background:#b8bdd0; border-radius:999px; }
.evidence-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  align-items:start; gap:14px; }
.evidence-card { box-sizing:border-box; height:250px; min-height:250px; max-height:250px;
  display:flex; flex-direction:column; overflow:hidden; background:white;
  border:1px solid #e2e6f0; border-radius:16px; padding:16px;
  box-shadow:0 4px 16px rgba(32,40,76,.045); }
.evidence-card-head { display:flex; align-items:center; gap:8px; margin-bottom:12px; }
.evidence-card-head > b { color:#5b57dc; font-size:13px; }
.star-badge, .similarity-badge { border-radius:999px; padding:4px 8px; font-size:10px; }
.star-badge { background:#fff4d9; color:#8a6412; }
.similarity-badge { background:#f0f2f8; color:#697187; }
.evidence-text-scroll { flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:hidden;
  margin-top:11px; padding-right:7px; overscroll-behavior:contain; scrollbar-gutter:stable; }
.evidence-text-scroll::-webkit-scrollbar { width:5px; }
.evidence-text-scroll::-webkit-scrollbar-track { background:transparent; }
.evidence-text-scroll::-webkit-scrollbar-thumb { background:#d3d6e4; border-radius:999px; }
.evidence-card blockquote { margin:0; border-left:3px solid #aaa8ff; padding:4px 0 4px 12px;
  color:#3f465c; font-size:13px; line-height:1.6; }
.source-badges { min-height:20px; display:flex; flex-wrap:wrap; align-items:center; gap:5px; margin:0; }
.source-badges small { color:#969caf; font-size:8px; letter-spacing:.8px; margin-right:3px; }
.source-badges span { background:#f0efff; color:#5d59c8; border-radius:999px;
  padding:3px 7px; font-size:9px; }
.trace-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
.trace-card { background:white; border:1px solid #e2e6f0; border-radius:16px; padding:15px;
  box-shadow:0 4px 16px rgba(32,40,76,.045); }
.trace-card-head { display:flex; justify-content:space-between; align-items:center; }
.trace-card-head > div { display:flex; align-items:center; gap:7px; }
.trace-card-head > span { color:#6662d8; background:#f0efff; border-radius:999px;
  padding:4px 8px; font-size:9px; font-weight:750; }
.trace-flow { color:#8b91a4; display:block; margin:3px 0 12px; }
.trace-steps { border-top:1px solid #edf0f5; padding-top:8px; }
.trace-step { display:grid; grid-template-columns:24px 1fr; gap:8px; padding:8px 0;
  border-bottom:1px solid #f0f2f6; }
.trace-step:last-child { border-bottom:0; }
.step-number { width:22px; height:22px; display:flex; align-items:center; justify-content:center;
  border-radius:7px; background:#262d56; color:white; font-size:10px; font-weight:700; }
.trace-step code { color:#4f4bb8; background:#f2f1ff; border-radius:5px; padding:2px 5px;
  font-size:10px; overflow-wrap:anywhere; }
.trace-step p { margin:5px 0 0; color:#6d7488; font-size:10px; line-height:1.45;
  overflow-wrap:anywhere; }
.no-tools { color:#9298a9; background:#f7f8fb; padding:12px; border-radius:9px; font-size:11px; }
.empty-detail { min-height:180px; display:flex; flex-direction:column; align-items:center;
  justify-content:center; background:white; border:1px dashed #d9ddea; border-radius:16px; color:#51586d; }
.empty-detail span { color:#969bad; font-size:12px; margin-top:4px; }
#run-btn { min-height: 47px; font-weight: 750; border-radius: 12px; }
#status-box { border: 0; background: transparent; color: #596078; }
@keyframes pulse { 50% { opacity:.25; transform:scale(.7); } }
@media (max-width: 900px) {
  .metric-strip { grid-template-columns:repeat(2, 1fr); }
  .evidence-grid, .trace-grid { grid-template-columns:1fr; }
  .detail-heading { align-items:flex-start; }
}
"""


def build_comparison_ui():
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    with gr.Blocks(title="Yelp Agent · System Comparison", theme=theme, css=CSS) as demo:
        gr.HTML(
            "<div class='hero'><div class='eyebrow'>YELP BUSINESS INTELLIGENCE LAB</div>"
            "<h1>One question. Six reasoning systems.</h1>"
            "<p>Compare grounded answers, latency, model calls, tools, and evidence on the same Yelp query.</p></div>"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="query-panel"):
                question = gr.Textbox(
                    label="Research question", lines=3,
                    placeholder="What do customers complain about most at this business?",
                )
                business = gr.Dropdown(
                    choices=legacy.DROPDOWN_CHOICES,
                    value="(Global search — no specific business)",
                    label=f"Business · {len(legacy.CATALOGUE)} available", filterable=True,
                )
                business_id = gr.Textbox(visible=False)
            with gr.Column(scale=4, elem_classes="query-panel"):
                selected = gr.CheckboxGroup(
                    choices=SYSTEMS,
                    value=["Direct LLM", "RAG Baseline", "ReAct"],
                    label="Systems to compare",
                )
                model = gr.Dropdown(
                    choices=list(DEEPSEEK_MODELS),
                    value=DEFAULT_DEEPSEEK_MODEL,
                    label="DeepSeek model",
                    info="One model is shared by all selected reasoning systems.",
                    interactive=True,
                )
                run = gr.Button("Run comparison", variant="primary", elem_id="run-btn")
                status = gr.Markdown("Ready · choose at least two systems", elem_id="status-box")

            with gr.Column(scale=5, elem_classes="examples-panel"):
                gr.Examples(
                    examples=[[row[0], row[1]] for row in legacy.EXAMPLES],
                    inputs=[question, business],
                    label="Question examples",
                    examples_per_page=len(legacy.EXAMPLES),
                )

        business.change(legacy.on_business_select, business, business_id)

        with gr.Tabs():
            with gr.Tab("Live comparison"):
                gr.HTML("<div class='section-title'><h2>Answers side by side</h2><p>Every card receives the same question and business context.</p></div>")
                card_outputs = []
                for row_systems in (SYSTEMS[:3], SYSTEMS[3:]):
                    with gr.Row(equal_height=True):
                        for system in row_systems:
                            card_outputs.append(
                                gr.Markdown(_card(None, system), elem_classes="system-card")
                            )
                gr.Markdown("## Run summary")
                summary = gr.Dataframe(
                    headers=["System", "Model", "Latency (s)", "LLM calls", "Tool calls", "Evidence"],
                    datatype=["str", "str", "number", "number", "number", "str"],
                    interactive=False, wrap=True,
                )

            with gr.Tab("Evidence & traces"):
                evidence = gr.Markdown(
                    "<div class='empty-detail'><b>No retrieved evidence yet</b>"
                    "<span>Run a comparison to build the shared evidence pool.</span></div>",
                    elem_classes="detail-output",
                )
                traces = gr.Markdown(
                    "<div class='empty-detail'><b>No execution traces yet</b>"
                    "<span>Run a comparison to inspect tool calls.</span></div>",
                    elem_classes="detail-output",
                )

            with gr.Tab("Benchmark snapshot"):
                gr.Markdown(
                    "## Existing paradigm study\n"
                    "Aggregated from the repository's recorded experiment rows. Live runs above are separate."
                )
                gr.Dataframe(
                    value=_benchmark_rows(),
                    headers=["System", "Runs", "Quality /8", "Avg latency (s)", "Avg cost ($)"],
                    datatype=["str", "number", "number", "number", "number"],
                    interactive=False,
                )

        outputs = [status] + card_outputs + [summary, evidence, traces]
        run.click(
            compare,
            inputs=[question, business_id, selected, model],
            outputs=outputs,
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    comparison_demo = build_comparison_ui()
    if legacy.IS_HF_SPACE:
        comparison_demo.launch(server_name="0.0.0.0")
    else:
        comparison_demo.launch(
            server_name="0.0.0.0",
            server_port=legacy.args.port,
            share=legacy.args.share,
            show_error=True,
        )
