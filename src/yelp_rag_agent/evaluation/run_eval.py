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

from yelp_rag_agent.config import OLLAMA_BASE_URL, OLLAMA_MODEL, RESULTS_DIR, load_config
from yelp_rag_agent.backends import load_backend
from yelp_rag_agent.tools.summarizer_tool import set_backend
from yelp_rag_agent.pipelines.rag_baseline import run_rag_pipeline
from yelp_rag_agent.pipelines.agent_runner import run_agent

QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"

FIELDNAMES = [
    "question_id", "question_type", "business_id", "question",
    "system",
    "answer",
    "tools_called", "tool_count", "elapsed_seconds",
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


def run_direct_llm(question: str, business_id: Optional[str],
                   base_url: str, model: str) -> dict:
    if business_id:
        prompt = (f"You are a Yelp review analyst.\n"
                  f"Answer the following question about Yelp business ID: {business_id}\n\n"
                  f"Question: {question}\n\nAnswer based only on your general knowledge.")
    else:
        prompt = (f"You are a Yelp review analyst.\n"
                  f"Question: {question}\n\nAnswer based only on your general knowledge.")

    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"temperature": 0}},
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
    except Exception as e:
        answer = f"[ERROR] {e}"

    return {
        "answer": answer, "tools_called": "",
        "tool_count": 0, "elapsed_seconds": round(time.time() - t0, 2),
    }


def run_rag(question: str, business_id: Optional[str]) -> dict:
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
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_full_agent(question: str, business_id: Optional[str]) -> dict:
    result = run_agent(question=question, business_id=business_id)
    return {
        "answer"         : result["final_answer"],
        "tools_called"   : " → ".join(tc["tool"] for tc in result["tool_calls"]),
        "tool_count"     : result["steps"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_evaluation(config_path: str, output_name: str = "eval_results.csv",
                   resume: bool = True) -> None:
    cfg        = load_config(config_path)
    base_url   = cfg.get("base_url", OLLAMA_BASE_URL)
    model_name = cfg.get("model", OLLAMA_MODEL)

    backend = load_backend(config_path)
    set_backend(backend)

    questions  = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    output_csv = RESULTS_DIR / output_name

    completed: set[tuple] = set()
    if resume and output_csv.exists():
        with open(output_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((row["question_id"], row["system"]))
        print(f"Resuming — {len(completed)} rows already completed.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or not resume

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(questions) * 3
        done  = 0

        for q in questions:
            qid      = q["id"]
            qtype    = q["type"]
            biz_id   = q.get("business_id")
            question = q["question"]

            systems = [
                ("direct_llm",
                 lambda qu, bi: run_direct_llm(qu, bi, base_url, model_name)),
                ("rag_baseline", run_rag),
                ("full_agent",   run_full_agent),
            ]

            for sys_name, sys_fn in systems:
                done += 1
                if (qid, sys_name) in completed:
                    print(f"  [{done}/{total}] SKIP  {qid} | {sys_name}")
                    continue

                print(f"\n  [{done}/{total}] {qid} | {sys_name}  —  {question[:55]}…")
                try:
                    sys_result = sys_fn(question, biz_id)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    sys_result = {"answer": f"[ERROR] {e}",
                                  "tools_called": "", "tool_count": 0,
                                  "elapsed_seconds": 0}

                row = {
                    "question_id"      : qid,
                    "question_type"    : qtype,
                    "business_id"      : biz_id or "",
                    "question"         : question,
                    "system"           : sys_name,
                    "answer"           : sys_result["answer"].replace("\n", " | "),
                    "tools_called"     : sys_result["tools_called"],
                    "tool_count"       : sys_result["tool_count"],
                    "elapsed_seconds"  : sys_result["elapsed_seconds"],
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
                      f"{sys_result['tool_count']} tools, {sys_result['elapsed_seconds']}s")

    print(f"\nResults saved to: {output_csv}")


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
    systems    = ["direct_llm", "rag_baseline", "full_agent"]

    print(f"\n{'='*70}\nEvaluation Summary — {output_csv.name}\n{'='*70}")

    print("\n--- Auto Metrics ---")
    header = f"{'System':<18} {'Avg Tools':>10} {'Avg Time(s)':>12} {'Evidence%':>10} {'Avg Length':>11}"
    print(header)
    print("-" * len(header))
    for sys in systems:
        sys_rows = [r for r in rows if r["system"] == sys]
        if not sys_rows:
            continue
        print(f"{sys:<18} "
              f"{sum(int(r['tool_count']) for r in sys_rows)/len(sys_rows):>10.1f} "
              f"{sum(float(r['elapsed_seconds']) for r in sys_rows)/len(sys_rows):>12.1f} "
              f"{sum(1 for r in sys_rows if r['has_evidence']=='True')/len(sys_rows):>9.0%} "
              f"{sum(int(r['answer_length']) for r in sys_rows)/len(sys_rows):>11.0f}")

    scored_rows = [r for r in rows if r["score_correctness"] != ""]
    if not scored_rows:
        print("\n[No manual scores found. Fill score columns then re-run --summarise.]")
        return

    print(f"\n--- Manual Scores ({len(scored_rows)} rows scored) ---")
    header2 = (f"{'System':<18} {'Correct':>8} {'Evidence':>9} {'Ground':>7} "
               f"{'Tool':>6} {'Effic':>6} {'TOTAL':>7}")
    print(header2)
    print("-" * len(header2))
    for sys in systems:
        sys_rows = [r for r in scored_rows if r["system"] == sys]
        if not sys_rows:
            continue
        avgs  = {c: sum(float(r[c]) for r in sys_rows if r[c] != "") /
                    max(1, sum(1 for r in sys_rows if r[c] != ""))
                 for c in score_cols}
        total = sum(avgs.values())
        print(f"{sys:<18} "
              f"{avgs['score_correctness']:>8.2f} {avgs['score_evidence']:>9.2f} "
              f"{avgs['score_groundedness']:>7.2f} {avgs['score_tool_use']:>6.2f} "
              f"{avgs['score_efficiency']:>6.2f} {total:>7.2f}")

    print(f"\n--- Hallucination Rate (score_groundedness == 0) ---")
    for sys in systems:
        sys_rows = [r for r in scored_rows
                    if r["system"] == sys and r["score_groundedness"] != ""]
        if not sys_rows:
            continue
        rate = sum(1 for r in sys_rows if float(r["score_groundedness"]) == 0) / len(sys_rows)
        print(f"  {sys:<18}: {rate:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way evaluation runner")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",       action="store_true")
    group.add_argument("--summarise", action="store_true")
    parser.add_argument("--config",    default="configs/ollama.yaml")
    parser.add_argument("--output",    default="eval_results.csv",
                        help="Output CSV filename inside results/")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.run:
        run_evaluation(args.config, args.output, resume=not args.no_resume)
    elif args.summarise:
        summarise(args.output)
