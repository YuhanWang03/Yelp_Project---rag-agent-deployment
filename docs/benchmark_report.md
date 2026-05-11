# Yelp RAG Agent — Benchmark Report

**Hardware:** NVIDIA A100 80GB (Google Colab)  
**Model:** Qwen2.5-7B-Instruct (fp16) vs Qwen2.5-7B-Instruct-AWQ (int4)  
**Serving framework:** LMDeploy 0.7.x — pytorch backend (fp16) / turbomind backend (AWQ)  
**Date:** April 2026

---

## 1. Overview

This report benchmarks two deployment configurations of the Yelp RAG Agent across two dimensions:

- **Deployment performance** — how fast and memory-efficient each backend is
- **Answer quality** — how accurately each system answers Yelp-related questions

Three systems are compared for quality:

| System | Description |
|---|---|
| `direct_llm` | LLM answers from memory only, no retrieval |
| `rag_baseline` | Fixed pipeline: stats → retrieval → summarize |
| `full_agent` | LangGraph ReAct agent with 5 tools |

---

## 2. Deployment Performance

Measured on a cold A100 80GB GPU with no prior model cache.

| Metric | fp16 (pytorch) | AWQ (turbomind) | AWQ Advantage |
|---|---|---|---|
| TTFT (ms) | 693.6 | **18.0** | **38.5× faster** |
| Throughput (tokens/s) | 73.9 | **166.9** | **2.26× higher** |
| Avg response time (s) | 2.00 | **0.91** | **2.2× faster** |
| VRAM peak (GB) | 64.55 | 65.30 | comparable |

![Deployment Performance](deployment_performance.png)

**Key observations:**

- AWQ's TTFT advantage (38.5×) is dramatic and comes from turbomind's kernel fusion during the prefill phase, which is especially efficient for W4A16 quantized weights.
- Despite using 4-bit weights, AWQ's VRAM footprint is not smaller than fp16 on A100. LMDeploy's turbomind backend pre-allocates a large KV-cache buffer at startup, which dominates the total allocation. On smaller GPUs (e.g., T4 16GB), the difference would be significant.
- Throughput improvement (2.26×) is consistent with typical W4A16 benchmarks on A100, where the memory bandwidth saving from smaller weights accelerates the decode phase.

---

## 3. Quality Evaluation

### 3.1 Evaluation Setup

- **Questions:** 20 questions across 4 types: Complaint Mining, Aspect Analysis, Business Profiling, Cross-Business Pattern
- **Scoring:** Human-scored on 5 dimensions, 0–5 scale each
- **Backend used for quality eval:** Both fp16 and AWQ

### 3.2 Score Dimensions

| Dimension | Definition |
|---|---|
| Correctness | Is the answer factually accurate? |
| Evidence | Does it cite specific review excerpts? |
| Groundedness | Is the answer supported by retrieved data? |
| Tool Use | Were the right tools called in a reasonable order? |
| Efficiency | Was the answer produced with minimal unnecessary steps? |

### 3.3 Results — fp16 Backend

| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total** |
|---|---|---|---|---|---|---|
| Direct LLM | 0.20 | 0.00 | 0.00 | 0.00 | 0.95 | **1.15** |
| RAG Baseline | 1.75 | 1.90 | 1.95 | 2.00 | 1.90 | **9.50** |
| Full Agent | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | **1.00** |

![Quality Scores fp16](quality_scores_fp16.png)

![Quality Radar](quality_radar.png)

### 3.4 fp16 vs AWQ Quality Comparison (RAG Baseline)

| Backend | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total** |
|---|---|---|---|---|---|---|
| fp16 (pytorch) | 1.75 | 1.90 | 1.95 | 2.00 | 1.90 | **9.50** |
| AWQ (turbomind) | 1.80 | 2.00 | 2.00 | 2.00 | 1.95 | **9.75** |

![fp16 vs AWQ Quality](quality_fp16_vs_awq.png)

AWQ scores are marginally higher (+0.25 total) despite using 4-bit quantization, which is within human scoring variance and indicates no meaningful quality degradation from quantization at this scale.

---

## 4. Discussion

### RAG Baseline outperforms both alternatives

The RAG Baseline achieved the highest quality scores by a large margin (total 9.50 vs 1.15 for Direct LLM). The fixed pipeline — `get_business_stats` → `search_review_chunks` → `summarize_evidence` — consistently retrieved relevant review chunks and produced grounded, evidence-backed answers. This demonstrates that even a simple retrieval-augmented pipeline substantially outperforms an unaided LLM on domain-specific question answering.

### Direct LLM produces generic, ungrounded answers

Direct LLM scored near zero on Evidence and Groundedness because it has no access to the actual Yelp review corpus. Its answers are plausible-sounding but generic (e.g., "customers commonly complain about service and food quality"), earning a small Efficiency score for being concise but failing on all factual dimensions.

### Full Agent tool calling compatibility issue

The Full Agent system scored near zero across all dimensions due to a known incompatibility between Qwen2.5's native `<tool_call>` format and LangChain's `ChatOpenAI` tool-calling protocol when served via LMDeploy. The model correctly identified which tools to call (as evidenced by the `<tool_call>` tags in the output), but LangGraph did not parse and execute these calls, leaving the agent unable to retrieve any evidence. This is a framework integration limitation, not a model capability limitation.

### AWQ delivers speed without quality loss

Across both deployment metrics and quality evaluation, AWQ (turbomind) strictly dominates or matches fp16 (pytorch):
- **2.26× higher throughput** and **38.5× lower TTFT** make it substantially more suitable for interactive applications
- **No meaningful quality degradation** — the 0.25-point quality difference is within human scoring variance
- For production deployment on a single A100, AWQ is the recommended configuration

---

## 5. Limitations

1. **Full Agent tool calling** — Qwen2.5-7B with LMDeploy HTTP mode does not correctly integrate with LangChain's function-calling protocol. Resolving this would require either a custom output parser for Qwen's `<tool_call>` format, or using LMDeploy's native Python API instead of the OpenAI-compatible HTTP endpoint.

2. **Small evaluation set** — 20 questions across 4 types is sufficient for directional conclusions but too small for statistically significant comparisons. Human scoring variance further limits precision.

3. **Single hardware configuration** — All experiments ran on A100 80GB. VRAM comparisons between fp16 and AWQ would be more meaningful on a memory-constrained GPU (e.g., T4 16GB), where AWQ's smaller weight footprint would matter more.

4. **Quality scores are moderate** — Even the best system (RAG Baseline) averaged only ~1.9/5 per dimension. This reflects the difficulty of the task (specific business-level questions) and the 7B model's limited reasoning capacity without additional fine-tuning or prompt engineering.

---

## 6. Conclusion

This project demonstrates a complete deployment pipeline for a Yelp-domain RAG agent, from local Ollama prototyping to cloud A100 deployment with fp16 and AWQ backends via LMDeploy.

**Main findings:**

- AWQ (W4A16, turbomind backend) achieves **2.26× throughput** and **38.5× lower TTFT** compared to fp16 with no quality regression, making it the preferred production configuration.
- The **RAG Baseline pipeline** is the most effective system, substantially outperforming both the unaided LLM and the (non-functional) agent on all quality dimensions.
- The **Full Agent** approach is architecturally sound but blocked by a Qwen-LangChain tool-calling compatibility issue that would need to be resolved for production use.

---

*Generated from `notebooks/benchmark_analysis.ipynb`. Raw data in `results/`.*
