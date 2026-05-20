"""
Aggregate Stage E paradigm-study CSVs into one canonical JSON for Stage F.

Merges the V4-Flash and V4-Pro eval CSVs, tags each row with its model
(the CSVs themselves carry no model column), coerces numeric/boolean columns
to proper types, validates counts/cleanliness, and writes
results/paradigm_study.json — the single input artifact for the Stage F
analysis notebook and report chapter.

Usage:
    python scripts/aggregate_paradigm_study.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

RESULTS = Path("results")
SOURCES = {
    "deepseek-v4-flash": RESULTS / "paradigm_v4flash.csv",
    "deepseek-v4-pro"  : RESULTS / "paradigm_v4pro.csv",
}
OUT = RESULTS / "paradigm_study.json"

INT_COLS   = ["tool_count", "llm_calls", "input_tokens", "output_tokens",
              "reasoning_tokens", "answer_length"]
FLOAT_COLS = ["elapsed_seconds", "cost_usd"]
SCORE_COLS = ["score_correctness", "score_evidence", "score_groundedness",
              "score_tool_use", "score_efficiency"]
PLANNING   = {"plan_and_solve", "rewoo", "reflection"}


def _to_int(v):
    v = str(v).strip()
    return int(v) if v.lstrip("-").isdigit() else None


def _to_float(v):
    v = str(v).strip()
    if v in ("", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def load_rows(model: str, path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            row["model"]        = model
            row["thinking"]     = _to_bool(r.get("thinking", "False"))
            row["has_evidence"] = _to_bool(r.get("has_evidence", "False"))
            for c in INT_COLS:
                row[c] = _to_int(r.get(c, ""))
            for c in FLOAT_COLS:
                row[c] = _to_float(r.get(c, ""))
            for c in SCORE_COLS:
                row[c] = _to_float(r.get(c, ""))
            rows.append(row)
    return rows


def main():
    all_rows: list[dict] = []
    counts: dict[str, int] = {}
    for model, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source CSV: {path}")
        rows = load_rows(model, path)
        all_rows.extend(rows)
        counts[model] = len(rows)
        print(f"{model:<20} {len(rows):>4} rows  ({path.name})")

    # --- validation ---
    empty_plans = [r for r in all_rows
                   if r["system"] in PLANNING and r["tool_count"] == 0]
    errors      = [r for r in all_rows if str(r["answer"]).startswith("[ERROR]")]
    cost_by_model: dict[str, float] = {}
    for r in all_rows:
        cost_by_model[r["model"]] = cost_by_model.get(r["model"], 0.0) + (r["cost_usd"] or 0.0)
    cost_by_model = {k: round(v, 4) for k, v in cost_by_model.items()}

    print(f"\nValidation: {len(all_rows)} total rows | "
          f"empty_plans={len(empty_plans)} | error_rows={len(errors)}")
    print(f"Cost by model (USD, list price): {cost_by_model}")
    if empty_plans:
        print("WARNING: empty plans present:",
              [(r["model"], r["system"], r["question_id"]) for r in empty_plans])
    if errors:
        print("WARNING: error rows present:",
              [(r["model"], r["system"], r["question_id"]) for r in errors])

    out = {
        "generated_at"          : datetime.now().isoformat(timespec="seconds"),
        "source_files"          : {m: str(p) for m, p in SOURCES.items()},
        "row_count"             : len(all_rows),
        "counts_by_model"       : counts,
        "cost_by_model_usd_list": cost_by_model,
        "scored"                : any(r["score_correctness"] is not None for r in all_rows),
        "rows"                  : all_rows,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
