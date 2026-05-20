"""
LLM-as-judge scoring for the Stage E paradigm study.

Scores each answer row on the 4 QUALITY rubric dimensions (correctness,
evidence, groundedness, tool_use), 0-2 each, using DeepSeek-V4 as the judge.
Efficiency is intentionally NOT LLM-judged — it is objective and lives in the
raw elapsed_seconds / cost_usd / tool_count columns for Stage F.

Methodology caveats (note in the report):
  - Self-preference bias: a DeepSeek-V4 judge scores DeepSeek-V4 answers.
    It is the strongest judge available here (Groq Llama 3.1 8B is too weak);
    the bias applies uniformly across paradigms so RELATIVE comparisons hold.
  - Groundedness is assessed from internal consistency (do the answer's
    claims match the quotes it cites?), since retrieved chunks are not stored.

Scores are written back into the CSV in place (resumable: rows that already
have score_correctness are skipped). Re-run aggregate_paradigm_study.py
afterwards to refresh results/paradigm_study.json with the scores.

Usage:
    set DEEPSEEK_API_KEY=...
    python scripts/llm_judge.py                         # scores both CSVs
    python scripts/llm_judge.py --csv results/paradigm_v4flash.csv
    python scripts/llm_judge.py --judge-config configs/deepseek_v4_pro.yaml
    python scripts/llm_judge.py --limit 5               # dry-run a few rows
"""

import argparse
import csv
import sys
from pathlib import Path

from yelp_rag_agent.backends import load_backend
from yelp_rag_agent.pipelines._paradigm_common import extract_json

DEFAULT_CSVS = ["results/paradigm_v4flash.csv", "results/paradigm_v4pro.csv"]

_SCORE_FIELDS = {
    "correctness" : "score_correctness",
    "evidence"    : "score_evidence",
    "groundedness": "score_groundedness",
    "tool_use"    : "score_tool_use",
}

_JUDGE_PROMPT = """You are a strict, fair evaluator scoring an AI system's answer to a \
question about a corpus of Yelp reviews. Score ONLY the four dimensions below, each 0-2.

Dimension definitions:
1. correctness (0-2): Does the answer directly and accurately address the question?
   2=fully & correctly; 1=partial/incomplete/slightly off; 0=off-topic or wrong.
2. evidence (0-2): Does it cite specific review quotes or data (e.g. star counts)?
   2=2+ specific quotes/data points; 1=references reviews only in general terms; 0=generic, no review content.
3. groundedness (0-2): Are the claims supported by the evidence the answer itself cites?
   2=all major claims backed by cited quotes/data; 1=mostly grounded, 1-2 claims overreach; 0=multiple unsupported/contradicted claims.
4. tool_use (0-2): Given the tools called, was tool selection sensible for this question?
   2=called the right tools in a logical order; 1=relevant tools but missed one or suboptimal order; 0=no tools used, wrong tools, or critical input errors.
   NOTE: if TOOLS CALLED is empty, tool_use MUST be 0.

Question-type guidance:
- Complaint Mining: reward NAMED complaint categories (e.g. "long waits", "rude staff") + at least one negative quote.
- Aspect Analysis: must focus on the REQUESTED aspect; off-aspect content earns no evidence credit.
- Business Profiling: a good profile covers BOTH strengths and weaknesses.
- Cross-Business Pattern: judge whether patterns are concrete and quote-backed.

QUESTION TYPE: {qtype}
QUESTION: {question}
TOOLS CALLED: {tools_called}  (tool_count={tool_count})

ANSWER TO SCORE:
{answer}

Respond with ONLY this JSON (integers 0/1/2, one-sentence rationale):
{{"correctness": <0-2>, "evidence": <0-2>, "groundedness": <0-2>, "tool_use": <0-2>, "rationale": "<one sentence>"}}"""


def _clamp_score(v):
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return max(0, min(2, n))


def judge_row(backend, row: dict, retries: int = 1):
    """Return (scores_dict, rationale) or (None, error_str) on failure."""
    prompt = _JUDGE_PROMPT.format(
        qtype=row.get("question_type", ""),
        question=row.get("question", ""),
        tools_called=row.get("tools_called", "") or "(none)",
        tool_count=row.get("tool_count", ""),
        answer=row.get("answer", "")[:4000],
    )
    last_err = "judge parse failed"
    for _ in range(retries + 1):
        try:
            raw = backend.generate(prompt, temperature=0, max_tokens=512)
        except Exception as e:
            # API hiccup (e.g. Groq rate limit) → leave row unscored so a
            # re-run picks it up, rather than crashing the whole batch.
            last_err = f"judge API error: {type(e).__name__}"
            continue
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and "correctness" in parsed:
            scores = {k: _clamp_score(parsed.get(k)) for k in _SCORE_FIELDS}
            if all(v is not None for v in scores.values()):
                # direct_llm has no tools → enforce tool_use = 0 per rubric.
                if row.get("system") == "direct_llm":
                    scores["tool_use"] = 0
                return scores, str(parsed.get("rationale", ""))[:300]
    return None, last_err


def _target_cols(judge_tag: str) -> tuple[dict, str]:
    """Map score dims + notes to CSV columns. A judge_tag suffixes them
    (e.g. score_correctness__groq) so a second judge's scores coexist with
    the primary (untagged) V4 scores instead of overwriting them."""
    suffix = f"__{judge_tag}" if judge_tag else ""
    score_cols = {k: f"score_{k}{suffix}" for k in _SCORE_FIELDS}  # k = correctness, ...
    return score_cols, f"notes{suffix}"


def score_csv(path: Path, backend, limit=None, system=None, judge_tag="",
              questions=None) -> tuple[int, int]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fieldnames = list(rows[0].keys()) if rows else []
    score_cols, notes_col = _target_cols(judge_tag)
    # ensure target columns exist (a tagged second-judge run adds new ones)
    for col in list(score_cols.values()) + [notes_col]:
        if col not in fieldnames:
            fieldnames.append(col)
            for r in rows:
                r.setdefault(col, "")

    check_col = score_cols["correctness"]
    unscored = [r for r in rows if not str(r.get(check_col, "")).strip()]
    already  = len(rows) - len(unscored)
    todo = unscored
    if system:
        todo = [r for r in todo if r.get("system") == system]
    if questions:
        todo = [r for r in todo if r.get("question_id") in questions]
    if limit:
        todo = todo[:limit]
    print(f"\n{path.name}: {len(rows)} rows | {already} already scored | "
          f"{len(todo)} to score now → cols {check_col!r}…"
          + (f" (system={system})" if system else ""))

    scored, failed = 0, 0
    for i, row in enumerate(todo, 1):
        scores, rationale = judge_row(backend, row)
        label = f"{row['question_id']}|{row['system']}|t={row['thinking']}"
        if scores is None:
            failed += 1
            row[notes_col] = (row.get(notes_col, "") + " " + rationale).strip()
            print(f"  [{i}/{len(todo)}] {label}  FAIL: {rationale}")
        else:
            for k, col in score_cols.items():
                row[col] = scores[k]
            row[notes_col] = (row.get(notes_col, "") + " [judge] " + rationale).strip()
            scored += 1
            total = sum(scores.values())
            print(f"  [{i}/{len(todo)}] {label}  -> C{scores['correctness']} "
                  f"E{scores['evidence']} G{scores['groundedness']} "
                  f"T{scores['tool_use']}  (q/8={total})")
        # rewrite after each row for resumability (file is small)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return scored, failed


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge scorer")
    parser.add_argument("--csv", default=None,
                        help="Single CSV to score. Default: both V4 study CSVs.")
    parser.add_argument("--judge-config", default="configs/deepseek_v4_flash.yaml",
                        help="Backend config for the judge model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N unscored rows (dry-run).")
    parser.add_argument("--system", default=None,
                        help="Only score rows for this system (discrimination check).")
    parser.add_argument("--judge-tag", default="",
                        help="Suffix score columns (e.g. 'groq') so a second, "
                             "independent judge's scores coexist with the V4 ones "
                             "for cross-judge bias validation.")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question_ids to score (e.g. "
                             "CM_01,AA_01,BP_01,CP_01) — a stratified sample for "
                             "cross-judge validation within rate limits.")
    args = parser.parse_args()
    qset = {q.strip() for q in args.questions.split(",")} if args.questions else None

    backend = load_backend(args.judge_config)
    if hasattr(backend, "thinking"):
        backend.thinking = False  # judging does not need thinking; keep it fast
    print(f"Judge model: {getattr(backend, 'model', '?')}")

    targets = [Path(args.csv)] if args.csv else [Path(p) for p in DEFAULT_CSVS]
    total_scored, total_failed = 0, 0
    for path in targets:
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        s, fl = score_csv(path, backend, limit=args.limit, system=args.system,
                          judge_tag=args.judge_tag, questions=qset)
        total_scored += s
        total_failed += fl

    print(f"\nDone. scored={total_scored} failed={total_failed}")
    print("Now run: python scripts/aggregate_paradigm_study.py  to refresh the JSON.")
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
