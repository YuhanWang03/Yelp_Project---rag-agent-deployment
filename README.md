---
title: Yelp Business Intelligence Agent
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: true
---

# 🍽️ Yelp Business Intelligence Agent

A RAG-powered question-answering system over **60,823 Yelp review chunks**
(**50,000 reviews · 160 businesses**) that compares **six approaches** to
review QA — two baselines and four agent reasoning paradigms — all on
**DeepSeek-V4**.

> **🚀 Live demo:** https://huggingface.co/spaces/YUHAN03/yelp-rag-agent
> *(runs on DeepSeek-V4-Flash)*

> **📄 Full write-up (PDF):**
> [Project_Report.pdf](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/download/v1.0/Project_Report.pdf)
> · [Project_Report_zh.pdf (中文)](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/download/v1.0/Project_Report_zh.pdf)
> &mdash; full reasoning-paradigm study. Source HTML in
> `docs/project_overview.html`.

---

## The six systems compared

| System | Type | What it does |
|---|---|---|
| **Direct LLM** | baseline | Answers from parametric memory only — no retrieval |
| **RAG Baseline** | baseline | Fixed pipeline: stats → FAISS retrieve → structured summarize |
| **ReAct** | paradigm | Interleaved reason→act, autonomous tool selection (LangGraph) |
| **Plan-and-Solve** | paradigm | Plan all tool calls up front → sequential execute → solve |
| **ReWOO** | paradigm | Plan → **parallel** execute → solve |
| **Reflection** | paradigm | Answer → self-critique → revise |

The four paradigms share the same tools and retrieval; only the reasoning
structure varies. Framework (LangGraph) and model (DeepSeek-V4) are held
constant so the paradigm is the only variable.

---

## Key finding — quality saturates; paradigm choice is an efficiency decision

Each system was run on the same 20 questions across DeepSeek-V4-Flash and
V4-Pro and two "thinking" modes (320 answers), scored by an LLM-as-judge on
four quality dimensions (0–2 each, ≤8).

| System | Quality /8 | Latency (s) | Cost ($/q) | Output tok |
|---|---|---|---|---|
| Direct LLM | 0.2 | 3.8 | 0.00007 | 213 |
| **RAG Baseline** | **7.8** | **3.2** | **0.00016** | 286 |
| ReAct | 8.0 | 11.8 | 0.00118 | 765 |
| **Plan-and-Solve** | **7.9** | **6.4** | **0.00040** | 433 |
| **ReWOO** | **7.9** | **6.2** | **0.00036** | 404 |
| Reflection | 7.9 | 10.8 | 0.00081 | 704 |

*(DeepSeek-V4-Flash, thinking off.)*

- **Retrieval — not reasoning structure — drives quality.** Direct LLM (no
  retrieval) collapses to 0.2/8; every retrieval-grounded system reaches
  **7.8–8.0/8** and is statistically indistinguishable.
- **The paradigm tradeoff is cost & latency, not quality.** ReAct is the most
  token-expensive (it re-sends the growing history each turn); Plan-and-Solve
  and ReWOO are the leanest.
- **The thinking tax is real and unrewarded.** DeepSeek-V4 thinking mode adds
  ~3–4× latency and ~2× cost for **no measurable quality gain** (Pro:
  +0.00/8). V4-Pro costs ~10× V4-Flash for the same quality.
- **Recommendation:** Plan-and-Solve or ReWOO on Flash with thinking off —
  ceiling-quality answers at the lowest cost/latency.

> **Robustness — self-bias ruled out:** to check the DeepSeek-V4-judges-V4
> concern, all 320 answers were re-scored by an independent judge from a
> different family (OpenAI gpt-4o-mini). The two judges agree closely
> (Pearson **r = 0.97**, mean abs diff **0.11/8**) — both put every retrieval
> system at the ceiling and Direct LLM far below — so the conclusions do not
> depend on the judge.
>
> **Caveats:** the 0–2 rubric saturates near the ceiling (an easy-task
> property, not a judge flaw); groundedness judged from internal consistency
> since retrieved chunks were not stored.

---

## Architecture

```
User Question
     │
     ├─► Direct LLM       — answer from memory only (baseline)
     ├─► RAG Baseline     — Stats → FAISS Retrieval → Summarize (fixed pipeline)
     ├─► ReAct            — interleaved reason→act, autonomous tools
     ├─► Plan-and-Solve   — plan → sequential execute → solve
     ├─► ReWOO            — plan → parallel execute → solve
     └─► Reflection       — answer → self-critique → revise
```

**Stack:** FAISS · sentence-transformers (all-MiniLM-L6-v2) · LangGraph ·
LangChain · DeepSeek-V4 (Flash/Pro) · Gradio 5

### Backend abstraction layer

All pipelines depend only on a `BaseBackend.generate(prompt) -> str`
interface. Concrete backends are interchangeable via YAML config — **no
application code changes when migrating between them**:

| Backend | Use case | Config |
|---|---|---|
| `DeepSeekBackend` | Paradigm study + live demo | `configs/deepseek_v4_flash.yaml` / `_pro.yaml` |
| `OllamaBackend` | Local development | `configs/ollama.yaml` |
| `GroqBackend` | Alternative serverless | `configs/groq.yaml` |
| `LMDeployBackend` | Self-hosted OpenAI-compatible | `configs/lmdeploy.yaml` |

---

## Live Setup

```bash
git clone https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment
cd Yelp_Project---rag-agent-deployment
pip install -e . --no-deps -r requirements.txt

# DeepSeek-V4 (recommended — used for the study and the live demo)
export DEEPSEEK_API_KEY="your_key"
python app.py --config configs/deepseek_v4_flash.yaml

# Local Ollama (offline dev)
ollama pull qwen2.5:7b
python app.py --config configs/ollama.yaml
```

Pick a paradigm in the **System / Reasoning Paradigm** dropdown; toggle
**Thinking mode** (DeepSeek-V4 only) to compare.

---

## Reproduce the study

```bash
# 1. Run the full matrix (4 paradigms × thinking on/off × 20 questions) + baselines
python -m yelp_rag_agent.evaluation.run_eval --run \
    --config configs/deepseek_v4_flash.yaml \
    --paradigm full_agent,plan_and_solve,rewoo,reflection --thinking both \
    --output paradigm_v4flash.csv

# 2. LLM-as-judge scoring (4 quality dimensions)
python scripts/llm_judge.py

# 3. Aggregate to the canonical JSON
python scripts/aggregate_paradigm_study.py

# 4. Build the analysis + report chapter
jupyter notebook notebooks/paradigm_analysis.ipynb
python scripts/build_paradigm_report.py
```

---

## Project Structure

```
yelp-rag-agent-deployment/
├── src/yelp_rag_agent/
│   ├── backends/        # Base + DeepSeek · Ollama · Groq · LMDeploy · HF Inference
│   ├── tools/           # retrieval · stats · classifier · summarizer
│   ├── pipelines/       # rag_baseline · agent_runner (ReAct) · plan_and_solve · rewoo · reflection
│   └── evaluation/      # run_eval · metrics · paradigm_figures · rubric · test_questions
├── configs/             # deepseek_v4_flash/_pro · ollama · groq · lmdeploy · hf_spaces YAMLs
├── notebooks/           # paradigm_analysis · (legacy) benchmark/colab notebooks
├── scripts/             # llm_judge · aggregate_paradigm_study · build_paradigm_report · smoke_test
├── docs/                # project_overview.html (EN + ZH) + exported PDFs
└── app.py               # Gradio demo (6-way paradigm selector + thinking toggle)
```

---

## Engineering Highlights

1. **Reasoning paradigms behind one interface** — ReAct, Plan-and-Solve,
   ReWOO and Reflection all reuse the same four tools and a shared planner;
   only the execution/refinement structure differs.
2. **DeepSeek-V4 thinking mode handling** — thinking shares the `max_tokens`
   budget with reasoning (which is emitted first), so the backend adds
   reasoning headroom to avoid truncated answers; ReAct degrades thinking off
   because V4 requires `reasoning_content` to be replayed across multi-turn
   tool calls (a reasoning-model × agent-tooling gap).
3. **Token & cost accounting** — per-run input/output/reasoning tokens and USD
   cost, captured from the DeepSeek API (and LangChain `usage_metadata` for
   ReAct), enabling the efficiency comparison.
4. **Chunk-level pre-filtered retrieval** — a `business_to_indices` map enables
   per-business search on small embedding subsets, avoiding the global Top-K +
   post-filter failure mode for low-volume businesses.

See `docs/project_overview.html` or the
[Project_Report PDF](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/latest)
for the full study and engineering rationale.
