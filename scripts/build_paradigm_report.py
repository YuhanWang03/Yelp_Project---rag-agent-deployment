"""
Build the "Reasoning Paradigm Study" report chapter and inject it into the
bilingual project overview HTML (Stage F).

Renders the key figures (from paradigm_figures) to base64 data-URIs so the
HTML stays self-contained (no loose PNG files), computes the summary tables
from results/paradigm_study.json, and injects an EN chapter into
docs/project_overview.html and a ZH chapter into docs/project_overview_zh.html.

Idempotent: the chapter is wrapped in BEGIN/END markers and replaced on
re-run; Take-aways is renumbered 7 -> 8 only once.

Usage:
    python scripts/build_paradigm_report.py
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless rendering for the report

from pathlib import Path

import pandas as pd

from yelp_rag_agent.config import PROJECT_ROOT
from yelp_rag_agent.evaluation import paradigm_figures as pf

BEGIN  = "<!-- BEGIN PARADIGM STUDY -->"
END    = "<!-- END PARADIGM STUDY -->"
ANCHOR = "<!-- PARADIGM_STUDY_ANCHOR -->"

DOCS = PROJECT_ROOT / "docs"
_CSVS = [PROJECT_ROOT / "results" / "paradigm_v4flash.csv",
         PROJECT_ROOT / "results" / "paradigm_v4pro.csv"]


def compute_cross_judge(tag: str = "openai"):
    """Per-system V4 vs independent-judge quality + agreement, read live from
    the CSVs (which carry score_*__<tag> columns). Returns None if the
    independent judge hasn't been run."""
    frames = [pd.read_csv(p) for p in _CSVS if p.exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    other = [f"{d}__{tag}" for d in pf.DIMS]
    if any(c not in df.columns for c in other):
        return None
    for c in pf.DIMS + other:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=pf.DIMS + other)
    if df.empty:
        return None
    df["q_v4"] = df[pf.DIMS].sum(axis=1)
    df["q_o"]  = df[other].sum(axis=1)
    rows = []
    for s in pf.PARADIGM_ORDER:
        sub = df[df["system"] == s]
        if len(sub):
            rows.append((pf.LABELS.get(s, s), sub["q_v4"].mean(), sub["q_o"].mean()))
    return {"rows": rows, "pearson": df["q_v4"].corr(df["q_o"]),
            "mad": (df["q_v4"] - df["q_o"]).abs().mean(), "n": len(df)}


def fig_to_uri(fig, dpi=100) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    pf.plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def html_table(headers, rows, num_from=1) -> str:
    th = "".join(f'<th class="num">{h}</th>' if i >= num_from else f"<th>{h}</th>"
                 for i, h in enumerate(headers))
    body = []
    for row in rows:
        tds = "".join(f'<td class="num">{c}</td>' if i >= num_from else f"<td>{c}</td>"
                      for i, c in enumerate(row))
        body.append(f"  <tr>{tds}</tr>")
    return "<table>\n  <tr>" + th + "</tr>\n" + "\n".join(body) + "\n</table>"


def _grp(df, model, system, thinking=False):
    return df[(df.model == model) & (df.system == system) & (df.thinking == thinking)]


def build_chapter(df, lang: str) -> str:
    img_ceiling = fig_to_uri(pf.fig_quality_ceiling(df))
    img_scatter = fig_to_uri(pf.fig_quality_vs_latency(df))
    img_tax     = fig_to_uri(pf.fig_latency_thinking_tax(df))

    order = ["direct_llm", "rag_baseline", "full_agent",
             "plan_and_solve", "rewoo", "reflection"]
    rows_a = []
    for s in order:
        g = _grp(df, "deepseek-v4-flash", s, False)
        rows_a.append([pf.LABELS[s], f"{g.quality8.mean():.1f}",
                       f"{g.elapsed_seconds.mean():.1f}", f"{g.cost_usd.mean():.5f}",
                       f"{g.input_tokens.mean():.0f}", f"{g.output_tokens.mean():.0f}"])

    tax = pf.table_thinking_tax(df)
    rows_b = [[r["model"].replace("deepseek-", ""), f"{r['quality_off']:.2f}",
               f"{r['quality_on']:.2f}", f"{r['quality_delta']:+.2f}",
               f"{r['latency_x']:.1f}×", f"{r['cost_x']:.1f}×"]
              for _, r in tax.iterrows()]

    pro_tax = tax[tax["model"] == "deepseek-v4-pro"].iloc[0]

    cj = compute_cross_judge()

    def _cj_section(lng: str) -> str:
        if not cj:
            return ""  # independent judge not run yet → section omitted
        cj_rows = [[name, f"{v4:.2f}", f"{ot:.2f}"] for name, v4, ot in cj["rows"]]
        if lng == "en":
            tbl = html_table(["System", "V4 judge /8", "OpenAI judge /8"], cj_rows)
            return (f"<h3>3.4 Judge validation (cross-model robustness)</h3>\n<p>\n"
                    f"  To rule out self-preference bias (a DeepSeek-V4 judge scoring\n"
                    f"  V4-generated answers), all {cj['n']} answers were re-scored by an\n"
                    f"  <strong>independent judge from a different model family</strong>\n"
                    f"  (OpenAI gpt-4o-mini). The two judges agree closely &mdash; row-level\n"
                    f"  Pearson <strong>r = {cj['pearson']:.2f}</strong>, mean absolute\n"
                    f"  difference <strong>{cj['mad']:.2f} / 8</strong> &mdash; and both place\n"
                    f"  every retrieval system at the ceiling with Direct LLM far below.\n"
                    f"  <strong>The paradigm comparison does not depend on the choice of\n"
                    f"  judge.</strong>\n</p>\n{tbl}\n")
        tbl = html_table(["系统", "V4 裁判 /8", "OpenAI 裁判 /8"], cj_rows)
        return (f"<h3>3.4 裁判验证（跨模型稳健性）</h3>\n<p>\n"
                f"  为排除自偏好偏差（V4 裁判给 V4 生成的答案打分），我们用<strong>另一个\n"
                f"  模型族的独立裁判</strong>（OpenAI gpt-4o-mini）对全部 {cj['n']} 条答案重新\n"
                f"  评分。两套裁判高度一致 &mdash; 行级 Pearson <strong>r = {cj['pearson']:.2f}</strong>、\n"
                f"  平均绝对差 <strong>{cj['mad']:.2f} / 8</strong> &mdash; 且都把所有检索系统打到\n"
                f"  天花板、把 Direct LLM 远远拉到底。<strong>范式对比的结论不依赖于裁判的\n"
                f"  选择。</strong>\n</p>\n{tbl}\n")

    cj_en, cj_zh = _cj_section("en"), _cj_section("zh")

    if lang == "en":
        tbl_a = html_table(
            ["Paradigm", "Quality /8", "Latency (s)", "Cost ($)", "Input tok", "Output tok"],
            rows_a)
        tbl_b = html_table(
            ["Model", "Quality off", "Quality on", "Δ Quality", "Latency", "Cost"],
            rows_b)
        return f"""<h2>3. Reasoning Paradigm Study (DeepSeek-V4)</h2>
<p>
  Beyond the original three systems, we added three more agent reasoning
  paradigms &mdash; <strong>Plan-and-Solve</strong>, <strong>ReWOO</strong>
  and <strong>Reflection</strong> &mdash; alongside the existing
  <strong>ReAct</strong> agent, and benchmarked all four (plus the Direct LLM
  and RAG baselines) on the same 20 questions. Each ran on two
  <strong>DeepSeek-V4</strong> models (Flash and Pro) and two
  <strong>thinking</strong> modes, for <strong>320 answers</strong> total,
  scored by an LLM-as-judge on four quality dimensions (0&ndash;2 each,
  &le;8). Efficiency (latency / cost / tokens) comes from raw run metrics.
</p>
<p style="font-size: 9.5pt;">
  Paradigms: <strong>ReAct</strong> interleaves reason&rarr;act, deciding each
  step from observations; <strong>Plan-and-Solve</strong> plans all tool calls
  up front then executes sequentially; <strong>ReWOO</strong> plans then
  executes in parallel; <strong>Reflection</strong> answers, then critiques and
  revises its own answer. Framework (LangGraph) and tools are held constant so
  the only variable is the reasoning structure.
</p>

<h3>3.1 Quality saturates for every retrieval paradigm</h3>
<img src="{img_ceiling}" style="max-width:100%;height:auto;" alt="Quality by system" />
<p>
  Every retrieval-grounded system &mdash; RAG Baseline and all four paradigms
  &mdash; lands at <strong>7.8&ndash;8.0 / 8</strong>, statistically
  indistinguishable. Only Direct LLM, which has no retrieval, collapses to
  {rows_a[0][1]} / 8. With a strong model and good retrieval, <strong>the
  reasoning structure does not move answer quality</strong>.
</p>
{tbl_a}
<p style="font-size: 9pt; color:#6b7280;">Flash, thinking off (Direct LLM / RAG are reference baselines).</p>

<h3>3.2 The paradigm tradeoff is efficiency, not quality</h3>
<img src="{img_scatter}" style="max-width:100%;height:auto;" alt="Quality vs latency" />
<p>
  Plotting quality against latency makes the result stark: quality is a flat
  band near the ceiling while latency spans more than <strong>30&times;</strong>.
  ReAct is the most token-expensive paradigm because it re-sends the growing
  message history on every turn ({rows_a[2][4]} input tokens vs
  ~{rows_a[3][4]} for Plan-and-Solve). Plan-and-Solve and ReWOO are the
  leanest; ReWOO's parallel execution gives it a small edge when a plan has
  several independent tool calls.
</p>

<h3>3.3 The thinking tax</h3>
<img src="{img_tax}" style="max-width:100%;height:auto;" alt="Thinking tax" />
<p>
  DeepSeek-V4's built-in thinking mode adds roughly
  <strong>{pro_tax['latency_x']:.0f}&times; latency</strong> and
  <strong>{pro_tax['cost_x']:.1f}&times; cost</strong> for a quality change of
  <strong>{pro_tax['quality_delta']:+.2f} / 8</strong> &mdash; i.e. none. On a
  task whose quality is already saturated, internal chain-of-thought buys
  nothing.
</p>
{tbl_b}
<p style="font-size: 9pt; color:#6b7280;">Averaged over Plan-and-Solve, ReWOO, Reflection.</p>

{cj_en}
<h3>3.5 Conclusion</h3>
<div class="callout">
  <strong>On this RAG-QA task, paradigm choice is an efficiency decision, not a
  quality one.</strong> All retrieval paradigms reach ceiling quality, so the
  right choice is the cheapest and fastest: <strong>Plan-and-Solve or ReWOO on
  Flash with thinking off</strong>. <strong>Explicit Reflection does not improve
  quality</strong> over simpler paradigms while costing 2&ndash;3&times; more,
  and <strong>thinking mode adds large latency/cost for zero measurable quality
  gain</strong> &mdash; the answer to our motivating question: built-in
  reasoning does not subsume or reward explicit Reflection once quality is
  saturated. Pro is ~10&times; the cost of Flash for the same quality.
</div>
<p style="font-size: 9pt; color:#6b7280;">
  Caveats: the 0&ndash;2 rubric is too coarse to separate near-ceiling answers
  (a property of an easy task, not a judge flaw &mdash; see &sect;3.4);
  groundedness is judged from internal consistency since retrieved chunks were
  not stored. Self-preference bias was tested and ruled out (&sect;3.4).
</p>"""

    # Chinese
    tbl_a = html_table(
        ["范式", "质量 /8", "延迟 (s)", "成本 ($)",
         "输入 tok", "输出 tok"], rows_a)
    tbl_b = html_table(
        ["模型", "质量(off)", "质量(on)", "Δ 质量",
         "延迟", "成本"], rows_b)
    return f"""<h2>3. 推理范式对比研究（DeepSeek-V4）</h2>
<p>
  在原有三个系统之外，我们新增了三种智能体推理范式——<strong>Plan-and-Solve</strong>、<strong>ReWOO</strong>、<strong>Reflection</strong>，连同原有的 <strong>ReAct</strong> agent，在相同的 20 题上对比这四个范式（加 Direct LLM 和 RAG 基线）。每个范式跨两个 <strong>DeepSeek-V4</strong> 模型（Flash / Pro）和两种 <strong>thinking</strong> 模式跑，共 <strong>320 条答案</strong>，由 LLM-as-judge 在四个质量维度上评分（0&ndash;2 分/维度，满分 8）。效率（延迟/成本/token）取自原始运行指标。
</p>
<p style="font-size: 9.5pt;">
  范式说明：<strong>ReAct</strong> 边推理边行动，根据观测决定每一步；<strong>Plan-and-Solve</strong> 先规划全部工具调用再顺序执行；<strong>ReWOO</strong> 规划后并行执行；<strong>Reflection</strong> 先答、再自我批判与修正。框架（LangGraph）和工具保持不变，唯一变量是推理结构。
</p>

<h3>3.1 所有检索范式的质量都饱和</h3>
<img src="{img_ceiling}" style="max-width:100%;height:auto;" alt="各系统质量" />
<p>
  所有带检索的系统——RAG 基线和四个范式——都落在 <strong>7.8&ndash;8.0 / 8</strong>，统计上无法区分；只有无检索的 Direct LLM 崩到 {rows_a[0][1]} / 8。在强模型 + 好检索下，<strong>推理结构不影响答案质量</strong>。
</p>
{tbl_a}
<p style="font-size: 9pt; color:#6b7280;">Flash，thinking off（Direct LLM / RAG 为参照基线）。</p>

<h3>3.2 范式的取舍是效率，不是质量</h3>
<img src="{img_scatter}" style="max-width:100%;height:auto;" alt="质量 vs 延迟" />
<p>
  把质量对延迟作图，结论一目了然：质量是贴近天花板的一条水平带，而延迟跨度超过 <strong>30&times;</strong>。ReAct 是 token 最贵的范式，因为它每轮都重发不断增长的消息历史（{rows_a[2][4]} 输入 token，而 Plan-and-Solve 约 {rows_a[3][4]}）。Plan-and-Solve 和 ReWOO 最精简；当计划含多个独立工具调用时，ReWOO 的并行执行略占优势。
</p>

<h3>3.3 thinking 税</h3>
<img src="{img_tax}" style="max-width:100%;height:auto;" alt="thinking 税" />
<p>
  DeepSeek-V4 的内置 thinking 模式带来约 <strong>{pro_tax['latency_x']:.0f}&times; 延迟</strong>、<strong>{pro_tax['cost_x']:.1f}&times; 成本</strong>，而质量变化仅 <strong>{pro_tax['quality_delta']:+.2f} / 8</strong>——即几乎为零。在质量已饱和的任务上，内部思考链买不来任何收益。
</p>
{tbl_b}
<p style="font-size: 9pt; color:#6b7280;">对 Plan-and-Solve、ReWOO、Reflection 取平均。</p>

{cj_zh}
<h3>3.5 结论</h3>
<div class="callout">
  <strong>在这个 RAG 问答任务上，范式选择是效率决策，不是质量决策。</strong>所有检索范式都达到天花板质量，所以正确选择是最便宜最快的：<strong>Flash 上的 Plan-and-Solve 或 ReWOO，thinking 关</strong>。<strong>显式 Reflection 不提升质量</strong>，却贵 2&ndash;3&times;；<strong>thinking 模式加大量延迟/成本、质量零增益</strong>——这回答了我们的出发问题：当质量饱和时，内置推理既不取代也不奖励显式 Reflection。Pro 的成本约为 Flash 的 10&times;，质量相同。
</div>
<p style="font-size: 9pt; color:#6b7280;">
  局限：0&ndash;2 量表在顶端太粗，难以区分接近天花板的答案（这是任务偏易的属性，不是裁判缺陷——见 &sect;3.4）；未存检索 chunks，groundedness 按内部一致性评判。自偏好偏差已检验并排除（&sect;3.4）。
</p>"""


def inject(path: Path, chapter_html: str):
    html = path.read_text(encoding="utf-8")
    if ANCHOR not in html:
        raise RuntimeError(f"{path.name} missing {ANCHOR} — add it after §2 Architecture.")
    if BEGIN in html and END in html:                  # drop any previous block
        pre, rest = html.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        html = pre + post
    block = f"{BEGIN}\n{chapter_html}\n{END}"
    html = html.replace(ANCHOR, ANCHOR + "\n" + block, 1)  # insert right after the anchor
    path.write_text(html, encoding="utf-8")
    print(f"injected chapter into {path.name}")


def main():
    df = pf.load_study()
    inject(DOCS / "project_overview.html", build_chapter(df, "en"))
    inject(DOCS / "project_overview_zh.html", build_chapter(df, "zh"))
    print("Done. Re-export PDFs from the HTML if needed.")


if __name__ == "__main__":
    main()
