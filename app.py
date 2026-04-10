"""
Yelp Business Intelligence Agent — Gradio Demo

Usage:
    python app.py                                   # Ollama backend (default)
    python app.py --config configs/lmdeploy.yaml    # LMDeploy backend
    python app.py --config configs/ollama.yaml --model qwen2.5:14b  # override model
    python app.py --share                           # public Gradio link
"""

import argparse
import json
import os
import pickle
import time

import gradio as gr

from yelp_rag_agent.config import VECTORSTORE_META, BUSINESS_JSON, load_config
from yelp_rag_agent.backends import load_backend
from yelp_rag_agent.tools.summarizer_tool import set_backend
from yelp_rag_agent.pipelines.rag_baseline import run_rag_pipeline
from yelp_rag_agent.pipelines.agent_runner import run_agent

# ---------------------------------------------------------------------------
# Parse args early so backend is ready before UI loads
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/ollama.yaml")
parser.add_argument("--model",  default=None, help="Override model in YAML")
parser.add_argument("--share",  action="store_true")
parser.add_argument("--port",   type=int, default=7860)
args = parser.parse_args()

overrides = {"model": args.model} if args.model else None
backend   = load_backend(args.config, overrides=overrides)
set_backend(backend)
print(f"Backend loaded: {args.config}")

# ---------------------------------------------------------------------------
# Business catalogue
# ---------------------------------------------------------------------------

def _load_business_catalogue() -> dict:
    with open(VECTORSTORE_META, "rb") as f:
        store = pickle.load(f)
    b2i      = store["business_to_indices"]
    eligible = {bid: len(idxs) for bid, idxs in b2i.items() if len(idxs) > 50}

    catalogue = {}
    with open(BUSINESS_JSON, encoding="utf-8") as f:
        for line in f:
            biz = json.loads(line)
            bid = biz.get("business_id", "")
            if bid in eligible:
                cats        = (biz.get("categories") or "").split(", ")
                primary_cat = cats[0] if cats else "Business"
                catalogue[bid] = {
                    "name"       : biz.get("name", bid),
                    "city"       : biz.get("city", ""),
                    "stars"      : biz.get("stars", 0),
                    "categories" : primary_cat,
                    "chunk_count": eligible[bid],
                }
    return catalogue


print("Loading business catalogue…")
CATALOGUE = _load_business_catalogue()

_sorted_bizs    = sorted(CATALOGUE.items(), key=lambda x: -x[1]["chunk_count"])
DROPDOWN_CHOICES = ["(Global search — no specific business)"] + [
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)"
    for _, info in _sorted_bizs
]
LABEL_TO_ID = {
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)": bid
    for bid, info in _sorted_bizs
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
print(f"Catalogue loaded: {len(CATALOGUE)} businesses available.")

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["What do customers complain about most at this business?",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "RAG Baseline"],
    ["Analyze the main weaknesses of this business based on reviews.",
     "Santa Barbara Shellfish Company — Santa Barbara (★4.0, 203 chunks)",
     "Full Agent"],
    ["What aspects do customers praise and criticize about food and service?",
     "(Global search — no specific business)", "RAG Baseline"],
    ["Give me an overall profile of this business based on customer reviews.",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "Full Agent"],
    ["Do Yelp reviews show any patterns between wait time complaints and star ratings?",
     "(Global search — no specific business)", "Full Agent"],
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def on_business_select(label: str) -> str:
    if label == "(Global search — no specific business)" or not label:
        return ""
    return LABEL_TO_ID.get(label, "")


def _format_stats_dict(s: dict) -> str:
    if not s or s.get("review_count", 0) == 0:
        return ""
    bid      = s.get("business_id", "")
    dist     = s.get("star_distribution", {})
    dist_str = " &nbsp;|&nbsp; ".join(f"★{k}: **{v}**" for k, v in sorted(dist.items()))
    label    = ID_TO_LABEL.get(bid, "")
    name     = label.split(" — ")[0] if label else bid
    city     = label.split(" — ")[1].split(" (")[0] if " — " in label else ""
    return (
        f"### 📊 Business Stats\n\n"
        f"| Metric | Details |\n| :--- | :--- |\n"
        f"| **🏢 Name** | {name} |\n"
        f"| **🏙️ City** | {city} |\n"
        f"| **🆔 ID** | `{bid}` |\n"
        f"| **📝 Indexed Reviews** | {s['review_count']} chunks |\n"
        f"| **⭐ Avg Stars** | {s['avg_stars']} |\n"
        f"| **📈 Distribution** | {dist_str} |\n"
    )


def _format_stats_from_id(business_id: str | None) -> str:
    if not business_id or business_id not in CATALOGUE:
        return ""
    info = CATALOGUE[business_id]
    return (
        f"### 📊 Business Info\n\n"
        f"| Metric | Details |\n| :--- | :--- |\n"
        f"| **🏢 Name** | {info['name']} |\n"
        f"| **🏙️ City** | {info['city']} |\n"
        f"| **🏷️ Category** | {info['categories']} |\n"
        f"| **⭐ Yelp Stars** | {info['stars']} |\n"
        f"| **📝 Indexed Reviews** | {info['chunk_count']} chunks |\n"
    )

# ---------------------------------------------------------------------------
# Core query handler
# ---------------------------------------------------------------------------

def run_query(question: str, business_id: str, system: str):
    question    = question.strip()
    business_id = business_id.strip() or None

    if not question:
        yield "Please enter a question.", "", "", ""
        return

    yield (
        f"⏳ **Processing… ({system})**",
        "⏳ *Waiting for tools…*",
        "⏳ *Waiting for retrieval…*",
        "⏳ *Fetching stats…*",
    )

    # Direct LLM -------------------------------------------------------
    if system == "Direct LLM":
        import requests as _req
        cfg = load_config(args.config)
        t0  = time.time()
        try:
            resp = _req.post(
                f"{cfg['base_url']}/api/chat",
                json={"model": cfg["model"],
                      "messages": [{"role": "user", "content": question}],
                      "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
        except Exception as e:
            answer = f"Error: {e}"
        elapsed    = round(time.time() - t0, 2)
        answer_md  = f"### 🤖 Answer *(Direct LLM — no retrieval)*\n\n{answer}"
        tools_md   = f"No tools used.\n\n*⏱️ Elapsed: **{elapsed}s***"
        evid_md    = "*Direct LLM does not retrieve evidence.*"
        stats_md   = _format_stats_from_id(business_id)
        yield answer_md, tools_md, evid_md, stats_md
        return

    # RAG Baseline -----------------------------------------------------
    if system == "RAG Baseline":
        try:
            result = run_rag_pipeline(question, business_id=business_id)
        except Exception as e:
            yield f"Pipeline error: {e}", "", "", ""
            return

        syn      = result.get("synthesis", {})
        findings = syn.get("main_findings", [])
        evidence = syn.get("supporting_evidence", [])
        uncertain= syn.get("uncertainties", [])
        elapsed  = result.get("elapsed_seconds", "?")
        tools    = result.get("tools_called", [])
        stats    = result.get("business_stats")
        chunks   = result.get("retrieved_chunks", [])

        answer_lines = ["### 🎯 Main Findings\n"] + [f"- {f}" for f in findings]
        if uncertain:
            answer_lines += ["\n### ❓ Uncertainties\n"] + [f"- {u}" for u in uncertain]
        answer_md = "\n".join(answer_lines)

        tool_lines = [f"### ⚙️ Tools Called ({len(tools)} steps)\n"]
        tool_lines += [f"{i}. `{t}`" for i, t in enumerate(tools, 1)]
        tool_lines.append(f"\n*⏱️ Elapsed: **{elapsed}s***")
        tools_md = "\n".join(tool_lines)

        evid_lines = [f"### 📑 Retrieved Evidence ({len(chunks)} total)\n"]
        for item in evidence[:5]:
            evid_lines.append(f"**📌 Claim:** {item.get('claim', '')}\n")
            for quote in item.get("evidence", [])[:2]:
                evid_lines.append(f"> ❝ *{quote}* ❞\n")
            evid_lines.append("---\n")
        if not evidence and chunks:
            for c in chunks[:3]:
                evid_lines.append(
                    f"> ❝ *{c['chunk_text'][:200]}…* ❞\n> \n> — *(★{c['stars']})*\n\n---\n"
                )
        evid_md  = "\n".join(evid_lines)
        stats_md = _format_stats_dict(stats) if stats else _format_stats_from_id(business_id)
        yield answer_md, tools_md, evid_md, stats_md
        return

    # Full Agent -------------------------------------------------------
    if system == "Full Agent":
        try:
            result = run_agent(question, business_id=business_id, max_iterations=6)
        except Exception as e:
            yield f"Agent error: {e}", "", "", ""
            return

        answer     = result.get("final_answer") or result.get("answer", "*(No answer)*")
        tool_calls = result.get("tool_calls", [])
        elapsed    = result.get("elapsed_seconds", "?")

        answer_md  = f"### 🤖 Agent Answer\n\n{answer}"

        tool_lines = [f"### ⚙️ Action Trajectory ({len(tool_calls)} steps)\n"]
        for i, tc in enumerate(tool_calls, 1):
            tool_lines.append(f"**Step {i}: `{tc.get('tool', '?')}`**")
            tool_lines.append(f"- **In:** `{str(tc.get('input', ''))[:120]}`")
            if tc.get("output"):
                tool_lines.append(f"- **Out:** `{str(tc['output'])[:150]}…`")
            tool_lines.append("")
        tool_lines.append(f"*⏱️ Elapsed: **{elapsed}s***")
        tools_md = "\n".join(tool_lines)

        evid_lines  = ["### 📑 Retrieved Evidence *(from tools)*\n"]
        has_evidence = False
        for tc in tool_calls:
            if "search" in tc.get("tool", ""):
                try:
                    chunks = json.loads(tc["output"])
                    for c in chunks[:3]:
                        evid_lines.append(
                            f"> ❝ *{c['chunk_text'][:200]}…* ❞\n> \n"
                            f"> — *(★{c['stars']}, ID: `{c['business_id'][:8]}…`)*\n\n---\n"
                        )
                    has_evidence = True
                    break
                except Exception:
                    pass
        if not has_evidence:
            evid_lines.append("*No retrieval chunks found in context.*")
        evid_md = "\n".join(evid_lines)

        stats_md = ""
        for tc in tool_calls:
            if tc.get("tool") == "get_business_stats":
                try:
                    stats_md = _format_stats_dict(json.loads(tc["output"]))
                except Exception:
                    pass
                break
        if not stats_md:
            stats_md = _format_stats_from_id(business_id)

        yield answer_md, tools_md, evid_md, stats_md
        return

    yield "Unknown system.", "", "", ""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
#input-row { align-items: stretch !important; }
#examples-col .dataset, #examples-col .table-wrap {
    max-height: 680px !important;
    overflow-y: auto !important;
}
#examples-col thead th {
    position: sticky !important; top: 0 !important;
    background-color: white !important; z-index: 1 !important;
}
"""

def build_ui():
    with gr.Blocks(title="Yelp Business Intelligence Agent",
                   theme=gr.themes.Soft(), css=_CSS) as demo:

        with gr.Row(elem_id="input-row"):
            with gr.Column(scale=3):
                gr.Markdown(
                    "# Yelp Business Intelligence Agent\n"
                    "**FAISS Retrieval · LangGraph ReAct · Local LLM Serving**\n\n"
                    "Ask questions about Yelp businesses using three systems:\n"
                    "- **Direct LLM** — no retrieval baseline\n"
                    "- **RAG Baseline** — fixed pipeline (Stats → Search → Summarize)\n"
                    "- **Full Agent** — LangGraph ReAct with autonomous tool selection\n\n"
                    "> Vector store: 60,823 chunks · 160 businesses · all-MiniLM-L6-v2"
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What do customers complain about most?",
                    lines=2,
                )
                business_dropdown = gr.Dropdown(
                    choices=DROPDOWN_CHOICES,
                    value="(Global search — no specific business)",
                    label="Select Business (160 available, sorted by review count)",
                    filterable=True,
                )
                business_id_input = gr.Textbox(
                    label="Business ID (auto-filled from dropdown, or type manually)",
                    placeholder="e.g. ORL4JE6tz3rJxVqkdKfegA",
                )
                system_input = gr.Dropdown(
                    choices=["RAG Baseline", "Full Agent", "Direct LLM"],
                    value="RAG Baseline", label="System",
                )
                submit_btn = gr.Button("Run Query", variant="primary")
                business_dropdown.change(
                    fn=on_business_select,
                    inputs=business_dropdown,
                    outputs=business_id_input,
                )

            with gr.Column(scale=2, elem_id="examples-col"):
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_input, business_dropdown, system_input],
                    label="Quick Examples",
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                answer_output   = gr.Markdown(label="Answer")
                evidence_output = gr.Markdown(label="Retrieved Evidence")
            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Business Info")
                tools_output = gr.Markdown(label="Tool Calls")

        submit_btn.click(
            fn=run_query,
            inputs=[question_input, business_id_input, system_input],
            outputs=[answer_output, tools_output, evidence_output, stats_output],
        )

        gr.Markdown(
            "---\n"
            "**Stage 4 Evaluation Results** (20 questions × 3 systems, human scored)\n\n"
            "| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total /10** |\n"
            "|---|---|---|---|---|---|---|\n"
            "| Direct LLM | 0.25 | 0.00 | 0.00 | 0.00 | 1.70 | **1.95** |\n"
            "| RAG Baseline | 0.95 | 1.60 | 1.75 | 0.95 | 1.65 | **6.90** |\n"
            "| Full Agent | 1.05 | 1.15 | 1.15 | 1.30 | 0.10 | **4.75** |\n\n"
            "Hallucination Rate: Direct LLM **100%** → Full Agent **25%** → RAG Baseline **5%**"
        )
    return demo


if __name__ == "__main__":
    print(f"Starting Yelp Business Intelligence Agent demo…")
    print(f"  Config : {args.config}")
    print(f"  Port   : {args.port}")
    print(f"  Share  : {args.share}")
    print(f"  Businesses in dropdown: {len(CATALOGUE)}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
