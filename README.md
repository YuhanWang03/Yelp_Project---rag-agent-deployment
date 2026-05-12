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

A RAG-powered question-answering system over **60,823 Yelp review chunks** from
**50,000 reviews · 160 businesses**, comparing three architectures (Direct LLM
vs RAG Baseline vs LangGraph ReAct Agent) across four interchangeable LLM
backends (Ollama · LMDeploy · Groq · HF Inference).

> **🚀 Live demo:** https://huggingface.co/spaces/YUHAN03/yelp-rag-agent
> *(runs free on Groq Llama 3.1 8B Instant — no sign-up required)*

> **📄 Full write-up:** see `docs/project_overview.html` (printable) and
> `docs/benchmark_report.md` for the original A100 deployment study.

---

## Architecture

```
User Question
     │
     ├─► Direct LLM       — Llama 3.1 8B from memory only (baseline)
     │
     ├─► RAG Baseline     — Stats → FAISS Retrieval → Summarize (fixed pipeline)
     │
     └─► Full Agent       — LangGraph ReAct with autonomous tool selection
```

**Stack:** FAISS · sentence-transformers (all-MiniLM-L6-v2) · LangGraph ·
LangChain · Llama 3.1 8B (Groq) for demo · Qwen2.5-7B-AWQ (LMDeploy) for the
original A100 research · Gradio 5

### Backend abstraction layer

All pipelines depend only on a `BaseBackend.generate(prompt) -> str` interface.
Four concrete backends are interchangeable via YAML config — **no application
code changes when migrating between them**:

| Backend | Use case | Config |
|---|---|---|
| `OllamaBackend` | Local development | `configs/ollama.yaml` |
| `LMDeployBackend` | A100 deployment study | `configs/lmdeploy.yaml` |
| `GroqBackend` | HF Spaces live demo | `configs/groq.yaml` |
| `HFInferenceBackend` | Fallback serverless | `configs/hf_spaces.yaml` |

---

## Original Research Results (Qwen2.5-7B on A100 80GB)

### Deployment performance

| Backend | TTFT | Throughput | Avg Response | VRAM |
|---|---|---|---|---|
| fp16 (pytorch) | 693.6 ms | 73.9 tok/s | 2.00 s | 64.55 GB |
| **AWQ (turbomind)** | **18.0 ms** | **166.9 tok/s** | **0.91 s** | 65.30 GB |

AWQ delivers **38.5× lower TTFT** and **2.26× higher throughput** with no
quality regression.

### Quality evaluation (20 questions, human-scored 0–2/dimension, 10 max)

| System | Backend | Correctness | Evidence | Groundedness | Tool Use | Efficiency | Total /10 |
|---|---|---|---|---|---|---|---|
| Direct LLM | A100 / Qwen | 0.20 | 0.00 | 0.00 | 0.00 | 0.95 | 1.15 |
| **RAG Baseline** | **A100 / Qwen** | **1.75** | **1.90** | **1.95** | **2.00** | **1.90** | **9.50** |
| ~~Full Agent~~ | ~~A100 / Qwen~~ | ~~0.00~~ | ~~0.00~~ | ~~0.00~~ | ~~1.00~~ | ~~0.00~~ | ~~1.00~~ |
| **Full Agent (fixed)** | **Groq / Llama 3.1 8B** | **1.80** | 1.30 | 1.25 | **1.85** | 1.40 | **7.60** |

> **The Full Agent "fix" — a 7.6× quality recovery from one backend swap:**
> In the original A100 study, Qwen2.5-7B emits
> `<tool_call>...</tool_call>` XML tags rather than OpenAI-standard
> `tool_calls` JSON, which LangChain's `ChatOpenAI` client cannot parse —
> the agent's tool calls are silently dropped, scoring near zero across all
> dimensions. Switching the LLM backend to Llama 3.1 8B Instant (Groq), which
> emits standard `tool_calls`, **recovered Full Agent functionality with zero
> changes to pipeline or tool code** — only ~30 lines of new
> `GroqBackend` wrapper plus a YAML config. **0% hallucination rate** in the
> 20-question evaluation. This is the engineering payoff of the backend
> abstraction layer.

---

## Live Setup

```bash
# Clone
git clone https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment
cd Yelp_Project---rag-agent-deployment
pip install -e . --no-deps -r requirements.txt

# Option A — Free serverless (Groq Llama 3.1 8B Instant, recommended)
export GROQ_API_KEY="your_key"
python app.py --config configs/groq.yaml

# Option B — Local Ollama
ollama pull qwen2.5:7b
python app.py --config configs/ollama.yaml

# Option C — A100 with LMDeploy turbomind (AWQ)
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct-AWQ \
    --server-port 23333 --backend turbomind --model-format awq
python app.py --config configs/lmdeploy.yaml
```

---

## Project Structure

```
yelp-rag-agent-deployment/
├── src/yelp_rag_agent/
│   ├── backends/        # Base + Ollama · LMDeploy · Groq · HF Inference
│   ├── tools/           # retrieval · stats · classifier · summarizer
│   ├── pipelines/       # rag_baseline · agent_runner (LangGraph)
│   └── evaluation/      # run_eval · metrics · rubric · test_questions
├── configs/             # ollama / lmdeploy / groq / hf_spaces YAMLs
├── notebooks/           # colab_awq_deploy · colab_quality_eval · benchmark_analysis
├── scripts/             # serve_lmdeploy.sh · smoke_test.py · upload_to_hf.py
├── docs/                # project_overview.html · benchmark_report.md · charts
└── app.py               # Gradio demo (auto-detects HF Spaces)
```

---

## Engineering Highlights

1. **Backend abstraction layer** — same code runs on 4 LLM serving stacks.
   The Qwen→Llama swap for the live demo required ~30 lines of new code.
2. **Thread-safe lazy singletons** — fixed a race condition where LangGraph's
   parallel tool execution corrupted PyTorch CUDA state during
   `SentenceTransformer.to('cuda')`.
3. **Chunk-level pre-filtered retrieval** — `business_to_indices` map enables
   per-business search on small embedding subsets, avoiding the "global Top-K
   + post-filter" failure mode for low-volume businesses.
4. **JSON-string tool outputs** — works around Groq's strict schema that
   rejects empty-list `ToolMessage.content`.

See `docs/project_overview.html` (or `docs/benchmark_report.md`) for the full
write-up and engineering rationale.
