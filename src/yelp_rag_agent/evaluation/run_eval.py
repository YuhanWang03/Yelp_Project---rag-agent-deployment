"""
Three-Way Evaluation

Runs all 20 questions against three systems and saves results to CSV.

Systems:
    direct_llm    — LLM answers from memory, no retrieval, no tools
    rag_baseline  — Fixed pipeline (stats → search → summarize)
    full_agent    — LangGraph ReAct agent

Usage:
    # Generate answers
    python -m yelp_rag_agent.evaluation.run_eval --run --config configs/ollama.yaml

    # After manually filling score columns, compute summary
    python -m yelp_rag_agent.evaluation.run_eval --summarise
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import requests

from yelp_rag_agent.config import RESULTS_DIR, load_config
from yelp_rag_agent.backends import load_backend
from yelp_rag_agent.tools.summarizer_tool import set_backend
from yelp_rag_agent.pipelines.rag_baseline import run_rag_pipeline
from yelp_rag_agent.pipelines.agent_runner import run_agent
from yelp_rag_agent.pipelines.plan_and_solve import run_plan_and_solve
from yelp_rag_agent.pipelines.rewoo import run_rewoo
from yelp_rag_agent.pipelines.reflection import run_reflection
from yelp_rag_agent.pipelines._paradigm_common import apply_thinking
from yelp_rag_agent.evaluation.metrics import compute_cost

QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"

FIELDNAMES = [
    "question_id", "question_type", "business_id", "question",
    "system",
    "thinking",
    "answer",
    "tools_called", "tool_count", "llm_calls", "elapsed_seconds",
    "input_tokens", "output_tokens", "reasoning_tokens", "cost_usd",
    "has_evidence",
    "answer_length",
    "score_correctness",
    "score_evidence",
    "score_groundedness",
    "score_tool_use",
    "score_efficiency",
    "notes",
]

EVIDENCE_SIGNALS = ['"', "'", "review", "customer said", "one reviewer", "excerpt"]


def _has_evidence(answer: str) -> bool:
    lowered = answer.lower()
    return any(sig in lowered for sig in EVIDENCE_SIGNALS)


def _join_tools(tool_calls: list) -> str:
    return " → ".join(tc.get("tool", "?") for tc in tool_calls)


def run_direct_llm(question: str, business_id: Optional[str],
                   thinking: bool = False) -> dict:
    from yelp_rag_agent.tools.summarizer_tool import _backend
    apply_thinking(_backend, thinking)
    if business_id:
        prompt = (f"You are a Yelp review analyst.\n"
                  f"Answer the following question about Yelp business ID: {business_id}\n\n"
                  f"Question: {question}\n\nAnswer based only on your general knowledge.")
    else:
        prompt = (f"You are a Yelp review analyst.\n"
                  f"Question: {question}\n\nAnswer based only on your general knowledge.")

    t0 = time.time()
    try:
        answer = _backend.generate(prompt, temperature=0)
    except Exception as e:
        answer = f"[ERROR] {e}"

    return {
        "answer": answer, "tools_called": "", "tool_count": 0,
        "llm_calls": 1, "elapsed_seconds": round(time.time() - t0, 2),
    }


def run_rag(question: str, business_id: Optional[str],
            thinking: bool = False) -> dict:
    from yelp_rag_agent.tools.summarizer_tool import _backend
    apply_thinking(_backend, thinking)
    result = run_rag_pipeline(question=question, business_id=business_id, top_k=8)
    syn    = result["synthesis"]

    parts = []
    findings = syn.get("main_findings", [])
    if findings:
        parts.append("\n".join(f"• {f}" for f in findings))
    evidence = syn.get("supporting_evidence", [])
    if evidence:
        ev_lines = []
        for item in evidence:
            ev_lines.append(f'  Claim: "{item.get("claim", "")}"')
            for q in item.get("evidence", [])[:2]:
                ev_lines.append(f'    – "{q}"')
        parts.append("Supporting evidence:\n" + "\n".join(ev_lines))
    uncertainties = syn.get("uncertainties", [])
    if uncertainties:
        parts.append("Uncertainties:\n" + "\n".join(f"? {u}" for u in uncertainties))

    return {
        "answer"         : "\n".join(parts).strip(),
        "tools_called"   : " → ".join(result["tools_called"]),
        "tool_count"     : len(result["tools_called"]),
        "llm_calls"      : 1,  # one summarize_evidence LLM call
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_full_agent(question: str, business_id: Optional[str],
                   thinking: bool = False) -> dict:
    result = run_agent(question=question, business_id=business_id, thinking=thinking)
    return {
        "answer"         : result["final_answer"],
        "tools_called"   : _join_tools(result["tool_calls"]),
        "tool_count"     : result["steps"],
        "llm_calls"      : result.get("llm_calls", ""),
        "token_usage"    : result.get("token_usage"),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _paradigm_eval(runner, question, business_id, thinking) -> dict:
    result = runner(question, business_id=business_id, thinking=thinking)
    return {
        "answer"         : result["final_answer"],
        "tools_called"   : _join_tools(result["tool_calls"]),
        "tool_count"     : result["steps"],
        "llm_calls"      : result.get("llm_calls", ""),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_plan_and_solve_eval(question, business_id, thinking=False) -> dict:
    return _paradigm_eval(run_plan_and_solve, question, business_id, thinking)


def run_rewoo_eval(question, business_id, thinking=False) -> dict:
    return _paradigm_eval(run_rewoo, question, business_id, thinking)


def run_reflection_eval(question, business_id, thinking=False) -> dict:
    return _paradigm_eval(run_reflection, question, business_id, thinking)


ALL_SYSTEMS = {
    "direct_llm"    : run_direct_llm,
    "rag_baseline"  : run_rag,
    "full_agent"    : run_full_agent,
    "plan_and_solve": run_plan_and_solve_eval,
    "rewoo"         : run_rewoo_eval,
    "reflection"    : run_reflection_eval,
}

# ReAct cannot use DeepSeek thinking mode (reasoning_content replay across
# multi-turn tool calls is unsupported by the agent client). Skip its
# thinking-on cell rather than emit a degraded/duplicate row.
_NO_THINKING_SYSTEMS = {"full_agent"}


def run_evaluation(config_path: str, output_name: str = "eval_results.csv",
                   resume: bool = True,
                   overrides: Optional[dict] = None,
                   only_systems: Optional[list] = None,
                   thinking_modes: Optional[list] = None,
                   limit: Optional[int] = None) -> None:
    cfg        = load_config(config_path)
    if overrides:
        cfg.update(overrides)
    model_name = cfg.get("model", "unknown")

    backend = load_backend(config_path, overrides=overrides)
    set_backend(backend)

    if thinking_modes is None:
        thinking_modes = [False]

    systems = [(name, ALL_SYSTEMS[name]) for name in
               (only_systems or list(ALL_SYSTEMS)) if name in ALL_SYSTEMS]
    if not systems:
        raise ValueError(f"No valid systems selected. Available: {list(ALL_SYSTEMS)}")

    questions  = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
        print(f"--limit {limit}: running first {len(questions)} question(s) only.")
    output_csv = RESULTS_DIR / output_name

    completed: set[tuple] = set()
    if resume and output_csv.exists():
        with open(output_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((row["question_id"], row["system"],
                               row.get("thinking", "False")))
        print(f"Resuming — {len(completed)} rows already completed.")

    # Build the full task list (system × thinking), skipping invalid combos.
    tasks = []
    for sys_name, sys_fn in systems:
        for think in thinking_modes:
            if think and sys_name in _NO_THINKING_SYSTEMS:
                continue
            tasks.append((sys_name, sys_fn, think))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or not resume

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(questions) * len(tasks)
        done  = 0

        for q in questions:
            qid      = q["id"]
            qtype    = q["type"]
            biz_id   = q.get("business_id")
            question = q["question"]

            for sys_name, sys_fn, think in tasks:
                done += 1
                key = (qid, sys_name, str(think))
                if key in completed:
                    print(f"  [{done}/{total}] SKIP  {qid} | {sys_name} | think={think}")
                    continue

                print(f"\n  [{done}/{total}] {qid} | {sys_name} | think={think}  —  {question[:50]}…")
                if hasattr(backend, "reset_usage"):
                    backend.reset_usage()
                try:
                    sys_result = sys_fn(question, biz_id, think)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    sys_result = {"answer": f"[ERROR] {e}", "tools_called": "",
                                  "tool_count": 0, "llm_calls": "", "elapsed_seconds": 0}

                # Token usage: prefer the pipeline-reported usage (ReAct, via
                # LangChain); else read the backend accumulator (generate-based
                # paradigms). compute_cost no-ops for non-DeepSeek models.
                usage = sys_result.get("token_usage")
                if not usage and hasattr(backend, "get_usage"):
                    usage = backend.get_usage()
                usage = usage or {}
                cost  = compute_cost(usage, model_name)

                row = {
                    "question_id"      : qid,
                    "question_type"    : qtype,
                    "business_id"      : biz_id or "",
                    "question"         : question,
                    "system"           : sys_name,
                    "thinking"         : think,
                    "answer"           : sys_result["answer"].replace("\n", " | "),
                    "tools_called"     : sys_result["tools_called"],
                    "tool_count"       : sys_result["tool_count"],
                    "llm_calls"        : sys_result.get("llm_calls", ""),
                    "elapsed_seconds"  : sys_result["elapsed_seconds"],
                    "input_tokens"     : usage.get("input_tokens", ""),
                    "output_tokens"    : usage.get("output_tokens", ""),
                    "reasoning_tokens" : usage.get("reasoning_tokens", ""),
                    "cost_usd"         : cost["total_cost_usd"] if cost["total_cost_usd"] is not None else "",
                    "has_evidence"     : _has_evidence(sys_result["answer"]),
                    "answer_length"    : len(sys_result["answer"]),
                    "score_correctness" : "",
                    "score_evidence"    : "",
                    "score_groundedness": "",
                    "score_tool_use"    : "",
                    "score_efficiency"  : "",
                    "notes"             : "",
                }
                writer.writerow(row)
                f.flush()
                print(f"    → {sys_name}: {len(sys_result['answer'])} chars, "
                      f"{sys_result['tool_count']} tools, "
                      f"{sys_result.get('llm_calls','?')} llm, {sys_result['elapsed_seconds']}s")

    print(f"\nResults saved to: {output_csv}")


def _group_key(row) -> tuple:
    return (row["system"], row.get("thinking", "False"))


def _ordered_groups(rows) -> list[tuple]:
    """Stable, readable ordering of (system, thinking) groups."""
    order = list(ALL_SYSTEMS)
    seen  = []
    for r in rows:
        k = _group_key(r)
        if k not in seen:
            seen.append(k)
    return sorted(seen, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))


def _label(group: tuple) -> str:
    sys, think = group
    return f"{sys}{' +think' if think == 'True' else ''}"


def summarise(output_name: str = "eval_results.csv") -> None:
    output_csv = RESULTS_DIR / output_name
    if not output_csv.exists():
        print(f"No results file found at {output_csv}. Run with --run first.")
        return

    rows = []
    with open(output_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    score_cols = ["score_correctness", "score_evidence", "score_groundedness",
                  "score_tool_use", "score_efficiency"]
    groups = _ordered_groups(rows)

    print(f"\n{'='*78}\nEvaluation Summary — {output_csv.name}\n{'='*78}")

    print("\n--- Auto Metrics ---")
    header = (f"{'System':<22} {'Avg Tools':>10} {'Avg LLM':>8} "
              f"{'Avg Time(s)':>12} {'Avg Out Tok':>12} {'Avg $':>10} {'Evidence%':>10}")
    print(header)
    print("-" * len(header))
    for g in groups:
        gr = [r for r in rows if _group_key(r) == g]
        if not gr:
            continue
        def _avg_int(col):
            vals = [int(r[col]) for r in gr if str(r.get(col, "")).strip().lstrip("-").isdigit()]
            return sum(vals) / len(vals) if vals else 0
        def _avg_float(col):
            vals = [float(r[col]) for r in gr if str(r.get(col, "")).strip() not in ("", "None")]
            return sum(vals) / len(vals) if vals else 0
        print(f"{_label(g):<22} "
              f"{_avg_int('tool_count'):>10.1f} "
              f"{_avg_int('llm_calls'):>8.1f} "
              f"{_avg_float('elapsed_seconds'):>12.1f} "
              f"{_avg_int('output_tokens'):>12.0f} "
              f"{_avg_float('cost_usd'):>10.5f} "
              f"{sum(1 for r in gr if r['has_evidence']=='True')/len(gr):>9.0%}")

    scored_rows = [r for r in rows if r["score_correctness"] != ""]
    if not scored_rows:
        print("\n[No manual scores found. Fill score columns then re-run --summarise.]")
        return

    print(f"\n--- Manual Scores ({len(scored_rows)} rows scored) ---")
    header2 = (f"{'System':<22} {'Correct':>8} {'Evidence':>9} {'Ground':>7} "
               f"{'Tool':>6} {'Effic':>6} {'TOTAL':>7}")
    print(header2)
    print("-" * len(header2))
    for g in groups:
        gr = [r for r in scored_rows if _group_key(r) == g]
        if not gr:
            continue
        avgs  = {c: sum(float(r[c]) for r in gr if r[c] != "") /
                    max(1, sum(1 for r in gr if r[c] != ""))
                 for c in score_cols}
        total = sum(avgs.values())
        print(f"{_label(g):<22} "
              f"{avgs['score_correctness']:>8.2f} {avgs['score_evidence']:>9.2f} "
              f"{avgs['score_groundedness']:>7.2f} {avgs['score_tool_use']:>6.2f} "
              f"{avgs['score_efficiency']:>6.2f} {total:>7.2f}")

    print(f"\n--- Hallucination Rate (score_groundedness == 0) ---")
    for g in groups:
        gr = [r for r in scored_rows
              if _group_key(r) == g and r["score_groundedness"] != ""]
        if not gr:
            continue
        rate = sum(1 for r in gr if float(r["score_groundedness"]) == 0) / len(gr)
        print(f"  {_label(g):<22}: {rate:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way evaluation runner")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",       action="store_true")
    group.add_argument("--summarise", action="store_true")
    parser.add_argument("--config",    default="configs/ollama.yaml")
    parser.add_argument("--output",    default="eval_results.csv",
                        help="Output CSV filename inside results/")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--paradigm",  default=None,
                        help="Comma-separated subset of systems/paradigms to run: "
                             "direct_llm, rag_baseline, full_agent, plan_and_solve, "
                             "rewoo, reflection. Default: all six.")
    parser.add_argument("--systems",   default=None,
                        help="Deprecated alias for --paradigm.")
    parser.add_argument("--thinking",  default="off",
                        choices=["off", "on", "both"],
                        help="DeepSeek V4 thinking mode dimension. 'both' runs each "
                             "system twice (off+on). full_agent always runs off only.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions (for dry-run testing).")
    args = parser.parse_args()

    selector = args.paradigm or args.systems
    only_systems = None
    if selector:
        only_systems = [s.strip() for s in selector.split(",") if s.strip()]

    thinking_modes = {"off": [False], "on": [True], "both": [False, True]}[args.thinking]

    if args.run:
        run_evaluation(args.config, args.output, resume=not args.no_resume,
                       only_systems=only_systems, thinking_modes=thinking_modes,
                       limit=args.limit)
    elif args.summarise:
        summarise(args.output)
