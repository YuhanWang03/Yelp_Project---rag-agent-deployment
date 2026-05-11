---
title: Yelp Business Intelligence Agent
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
---

# Yelp Business Intelligence Agent

A RAG-powered question-answering system built on 60,823 Yelp review chunks, comparing three systems across deployment backends.

## Architecture

```
User Question
     │
     ├─► Direct LLM       — Qwen2.5-7B from memory only (baseline)
     │
     ├─► RAG Baseline     — Stats → FAISS Retrieval → Summarize (fixed pipeline)
     │
     └─► Full Agent       — LangGraph ReAct with 5 tools (autonomous)
```

**Stack:** FAISS · sentence-transformers (all-MiniLM-L6-v2) · LangGraph · Qwen2.5-7B-Instruct · Gradio

## Deployment Benchmark (A100 80GB)

| Backend | TTFT | Throughput | Avg Response | VRAM |
|---|---|---|---|---|
| fp16 (pytorch) | 693.6 ms | 73.9 tok/s | 2.00 s | 64.55 GB |
| **AWQ (turbomind)** | **18.0 ms** | **166.9 tok/s** | **0.91 s** | 65.30 GB |

AWQ delivers **38.5× lower TTFT** and **2.26× higher throughput** with no quality regression.

## Quality Evaluation (20 questions, human scored 0–5)

| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | Total |
|---|---|---|---|---|---|---|
| Direct LLM | 0.20 | 0.00 | 0.00 | 0.00 | 0.95 | 1.15 |
| **RAG Baseline** | **1.75** | **1.90** | **1.95** | **2.00** | **1.90** | **9.50** |
| Full Agent | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |

## Local Setup

```bash
# Clone and install
git clone https://github.com/your-username/yelp-rag-agent-deployment
cd yelp-rag-agent-deployment
pip install -e . --no-deps

# Start Ollama and run
ollama pull qwen2.5:7b
python app.py

# Or with LMDeploy on A100
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct-AWQ \
    --server-port 23333 --backend turbomind --model-format awq
python app.py --config configs/lmdeploy.yaml
```

## Project Structure

```
yelp-rag-agent-deployment/
├── src/yelp_rag_agent/
│   ├── backends/          # Ollama · LMDeploy · HF Inference
│   ├── tools/             # retrieval · stats · classifier · summarizer
│   ├── pipelines/         # rag_baseline · agent_runner (LangGraph)
│   └── evaluation/        # run_eval · metrics
├── configs/               # ollama.yaml · lmdeploy.yaml · hf_spaces.yaml
├── notebooks/             # colab_awq_deploy · colab_quality_eval · benchmark_analysis
├── scripts/               # serve_lmdeploy.sh · smoke_test.py
├── docs/                  # benchmark_report.md · PNG charts
└── app.py                 # Gradio demo
```
