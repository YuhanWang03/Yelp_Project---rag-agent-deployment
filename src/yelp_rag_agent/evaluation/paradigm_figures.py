"""
Shared figures + summary tables for the Stage E paradigm study (Stage F).

Single source of truth reused by:
  - notebooks/paradigm_analysis.ipynb   (interactive viewing)
  - scripts/build_paradigm_report.py    (embeds figures as base64 in the HTML report)

Story the figures tell: on this RAG-QA task answer quality is SATURATED
(~8/8) for every retrieval-grounded paradigm; only direct_llm (no retrieval)
fails. Paradigm choice, thinking mode, and model size move cost/latency a
lot but not quality — so the paradigm decision is an efficiency decision.

Do not call matplotlib.use() here — the caller picks the backend (the report
builder sets Agg before importing; the notebook uses the inline backend).
"""

import json

import pandas as pd
import matplotlib.pyplot as plt

from yelp_rag_agent.config import RESULTS_DIR

STUDY_PATH = RESULTS_DIR / "paradigm_study.json"

DIMS = ["score_correctness", "score_evidence",
        "score_groundedness", "score_tool_use"]

PARADIGM_ORDER = ["direct_llm", "rag_baseline", "full_agent",
                  "plan_and_solve", "rewoo", "reflection"]
LABELS = {
    "direct_llm"    : "Direct LLM",
    "rag_baseline"  : "RAG Baseline",
    "full_agent"    : "ReAct",
    "plan_and_solve": "Plan-and-Solve",
    "rewoo"         : "ReWOO",
    "reflection"    : "Reflection",
}
COLORS = {
    "direct_llm"    : "#9e9e9e",
    "rag_baseline"  : "#90a4ae",
    "full_agent"    : "#ef5350",
    "plan_and_solve": "#42a5f5",
    "rewoo"         : "#66bb6a",
    "reflection"    : "#ab47bc",
}


def load_study(path=STUDY_PATH) -> pd.DataFrame:
    """Load paradigm_study.json into a DataFrame with a quality8 column."""
    d = json.loads(path.read_text(encoding="utf-8") if hasattr(path, "read_text")
                   else open(path, encoding="utf-8").read())
    df = pd.DataFrame(d["rows"])
    df["quality8"] = df[DIMS].sum(axis=1)
    return df


def _present_systems(df) -> list[str]:
    return [s for s in PARADIGM_ORDER if s in set(df["system"])]


# ---------------------------------------------------------------------------
# Figures (each returns a matplotlib Figure)
# ---------------------------------------------------------------------------

def fig_quality_ceiling(df) -> plt.Figure:
    """Bar: avg quality /8 by system (Flash). Shows the ceiling + direct_llm drop."""
    sub = df[df["model"] == "deepseek-v4-flash"]
    systems = _present_systems(sub)
    means = [sub[sub["system"] == s]["quality8"].mean() for s in systems]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([LABELS[s] for s in systems], means,
           color=[COLORS[s] for s in systems])
    ax.axhline(8, ls="--", lw=1, color="#bbb", zorder=0)
    ax.set_ylim(0, 8.6)
    ax.set_ylabel("Avg quality (/8)")
    ax.set_title("Answer quality saturates for every retrieval paradigm\n"
                 "(DeepSeek-V4-Flash; only Direct LLM, which has no retrieval, fails)")
    for i, v in enumerate(means):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def fig_quality_vs_latency(df) -> plt.Figure:
    """Scatter: quality vs latency per (model,system,thinking) group. The money chart —
    quality flat near the ceiling while latency spans an order of magnitude."""
    grp = (df.groupby(["model", "system", "thinking"])
             .agg(quality=("quality8", "mean"),
                  latency=("elapsed_seconds", "mean")).reset_index())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for _, r in grp.iterrows():
        marker = "o" if r["model"] == "deepseek-v4-flash" else "^"
        edge   = "k" if r["thinking"] else "none"
        ax.scatter(r["latency"], r["quality"], s=90, marker=marker,
                   color=COLORS.get(r["system"], "#777"),
                   edgecolors=edge, linewidths=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Avg latency per question (s, log scale)")
    ax.set_ylabel("Avg quality (/8)")
    ax.set_title("Quality is flat; latency varies 30×+\n"
                 "(○ Flash  △ Pro · black edge = thinking on · color = paradigm)")
    # legend for paradigms
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=COLORS[s],
                          label=LABELS[s]) for s in _present_systems(df)]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_latency_thinking_tax(df) -> plt.Figure:
    """Grouped bars: latency by paradigm, thinking off vs on, one panel per model."""
    models = ["deepseek-v4-flash", "deepseek-v4-pro"]
    systems = ["plan_and_solve", "rewoo", "reflection"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        off = [sub[(sub.system == s) & (~sub.thinking)]["elapsed_seconds"].mean() for s in systems]
        on  = [sub[(sub.system == s) & (sub.thinking)]["elapsed_seconds"].mean() for s in systems]
        x = range(len(systems))
        ax.bar([i - 0.2 for i in x], off, width=0.4, label="thinking off", color="#90caf9")
        ax.bar([i + 0.2 for i in x], on,  width=0.4, label="thinking on",  color="#f48fb1")
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[s] for s in systems], rotation=15, ha="right")
        ax.set_ylabel("Avg latency (s)")
        ax.set_title(model.replace("deepseek-", ""))
        ax.legend(fontsize=8)
    fig.suptitle("The thinking tax: ~3-4× latency for no quality gain")
    fig.tight_layout()
    return fig


def fig_token_breakdown(df) -> plt.Figure:
    """Stacked bars: input / output(non-reasoning) / reasoning tokens by paradigm
    (Flash, thinking on where available). Shows ReAct's input overhead and the
    reasoning-token cost of thinking."""
    sub = df[df["model"] == "deepseek-v4-flash"].copy()
    # use thinking-on rows for paradigms that have them, else thinking-off
    rows = []
    for s in ["full_agent", "plan_and_solve", "rewoo", "reflection"]:
        cand = sub[(sub.system == s) & (sub.thinking)]
        if cand.empty:
            cand = sub[sub.system == s]
        rows.append((s, cand))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels, inp, out_nonreason, reason = [], [], [], []
    for s, cand in rows:
        labels.append(LABELS[s] + ("\n(think)" if cand["thinking"].any() else ""))
        inp.append(cand["input_tokens"].mean())
        r = cand["reasoning_tokens"].mean()
        out_nonreason.append(cand["output_tokens"].mean() - r)
        reason.append(r)
    ax.bar(labels, inp, label="input", color="#90a4ae")
    ax.bar(labels, out_nonreason, bottom=inp, label="output (answer)", color="#66bb6a")
    bottom2 = [a + b for a, b in zip(inp, out_nonreason)]
    ax.bar(labels, reason, bottom=bottom2, label="reasoning", color="#ab47bc")
    ax.set_ylabel("Avg tokens per question")
    ax.set_title("Token cost breakdown (Flash)\nReAct re-sends history (high input); thinking adds reasoning tokens")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tables (return DataFrames for display / HTML rendering)
# ---------------------------------------------------------------------------

def table_main(df) -> pd.DataFrame:
    """Per (model, system, thinking): n, quality/8, latency, cost, tokens."""
    g = (df.groupby(["model", "system", "thinking"])
           .agg(n=("quality8", "size"),
                quality8=("quality8", "mean"),
                latency_s=("elapsed_seconds", "mean"),
                cost_usd=("cost_usd", "mean"),
                input_tok=("input_tokens", "mean"),
                output_tok=("output_tokens", "mean"),
                reasoning_tok=("reasoning_tokens", "mean"))
           .round(3).reset_index())
    g["system"] = pd.Categorical(g["system"], PARADIGM_ORDER, ordered=True)
    return g.sort_values(["model", "system", "thinking"]).reset_index(drop=True)


def table_thinking_tax(df) -> pd.DataFrame:
    """Thinking off vs on (avg over the 3 dual-mode paradigms), per model:
    quality delta (≈0) vs latency/cost multipliers."""
    sub = df[df["system"].isin(["plan_and_solve", "rewoo", "reflection"])]
    rows = []
    for model in ["deepseek-v4-flash", "deepseek-v4-pro"]:
        m = sub[sub["model"] == model]
        off = m[~m["thinking"]]
        on  = m[m["thinking"]]
        rows.append({
            "model"          : model,
            "quality_off"    : round(off["quality8"].mean(), 2),
            "quality_on"     : round(on["quality8"].mean(), 2),
            "quality_delta"  : round(on["quality8"].mean() - off["quality8"].mean(), 2),
            "latency_x"      : round(on["elapsed_seconds"].mean() / off["elapsed_seconds"].mean(), 1),
            "cost_x"         : round(on["cost_usd"].mean() / max(off["cost_usd"].mean(), 1e-9), 1),
        })
    return pd.DataFrame(rows)
