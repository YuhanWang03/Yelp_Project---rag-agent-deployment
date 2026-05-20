"""
Cross-judge bias check (Stage H).

Compares the primary DeepSeek-V4 judge (unsuffixed score_* columns) against an
independent judge written into score_*__<tag> columns by:

    python scripts/llm_judge.py --judge-config configs/openai_judge.yaml --judge-tag openai

If the two judges agree on the per-system ranking and correlate row-by-row,
the self-preference-bias concern (V4 judging V4) does not change the study's
conclusions.

Usage:
    python scripts/compare_judges.py             # tag defaults to 'openai'
    python scripts/compare_judges.py --tag groq  # if you also ran the Groq judge
"""

import argparse

import pandas as pd

from yelp_rag_agent.evaluation.paradigm_figures import (
    load_study, DIMS, PARADIGM_ORDER, LABELS,
)

CSVS = ["results/paradigm_v4flash.csv", "results/paradigm_v4pro.csv"]


def _read_csvs() -> pd.DataFrame:
    frames = []
    for p in CSVS:
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="openai", help="Independent judge column suffix.")
    args = ap.parse_args()

    df = _read_csvs()
    primary = DIMS                                   # score_correctness, ...
    other   = [f"{d}__{args.tag}" for d in DIMS]     # score_correctness__groq, ...

    missing = [c for c in other if c not in df.columns]
    if missing:
        print(f"Independent judge columns not found: {missing}\n"
              f"Run first:\n  python scripts/llm_judge.py "
              f"--judge-config configs/groq_judge.yaml --judge-tag {args.tag}")
        return

    # keep only rows scored by BOTH judges
    df = df.dropna(subset=primary + other)
    for c in primary + other:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=primary + other)
    df["q_v4"]   = df[primary].sum(axis=1)
    df["q_other"] = df[other].sum(axis=1)
    n = len(df)
    print(f"Rows scored by both judges: {n}\n")

    # --- per-system quality under each judge ---
    print(f"{'System':<16}{'V4 /8':>8}{args.tag + ' /8':>10}{'Δ':>7}")
    print("-" * 41)
    order = [s for s in PARADIGM_ORDER if s in set(df["system"])]
    for s in order:
        sub = df[df["system"] == s]
        a, b = sub["q_v4"].mean(), sub["q_other"].mean()
        print(f"{LABELS.get(s, s):<16}{a:>8.2f}{b:>10.2f}{b-a:>+7.2f}")

    # --- agreement metrics ---
    pear = df["q_v4"].corr(df["q_other"], method="pearson")
    spear = df["q_v4"].corr(df["q_other"], method="spearman")
    mad  = (df["q_v4"] - df["q_other"]).abs().mean()
    rank_v4    = [df[df.system == s]["q_v4"].mean() for s in order]
    rank_other = [df[df.system == s]["q_other"].mean() for s in order]
    rank_match = (pd.Series(rank_v4).rank().tolist()
                  == pd.Series(rank_other).rank().tolist())

    print(f"\nRow-level Pearson r : {pear:.3f}")
    print(f"Row-level Spearman ρ: {spear:.3f}")
    print(f"Mean |Δ quality/8|  : {mad:.2f}")
    print(f"Per-system ranking identical: {rank_match}")

    print("\nVerdict:", (
        "judges agree — conclusions robust to judge choice (self-bias does not "
        "flip the paradigm comparison)."
        if (spear >= 0.6 or rank_match) and mad <= 1.0 else
        "judges diverge — inspect; self-bias may matter, report both."))


if __name__ == "__main__":
    main()
