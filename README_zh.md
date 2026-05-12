# 🍽️ Yelp 商家智能问答 Agent

基于 **60,823 条 Yelp 评论 chunk**（来自 50,000 条原始评论 · 160 家商家）
构建的 RAG 问答系统，对比三种问答架构（Direct LLM / RAG Baseline / LangGraph
ReAct Agent）在四种可互换的 LLM 后端上的表现（Ollama · LMDeploy · Groq · HF
Inference）。

> **🚀 在线 Demo：** https://huggingface.co/spaces/YUHAN03/yelp-rag-agent
> *（使用 Groq Llama 3.1 8B Instant，免费、免注册）*

> **📄 详细报告：** 见 `docs/project_overview_zh.html`（或导出的
> `docs/Project_Report_zh.pdf`），含项目概览、架构、A100 部署研究、
> Full Agent parser bug 修复故事等完整内容。

---

## 项目架构

```
用户问题
   │
   ├─► Direct LLM     —— Llama 3.1 8B 纯参数化记忆作答（baseline）
   │
   ├─► RAG Baseline   —— 固定流水线：Stats → FAISS 检索 → Summarize
   │
   └─► Full Agent     —— LangGraph ReAct，LLM 自主调度工具
```

**技术栈：** FAISS · sentence-transformers (all-MiniLM-L6-v2) · LangGraph ·
LangChain · Demo 用 Llama 3.1 8B (Groq) · 原 A100 研究用 Qwen2.5-7B-AWQ
(LMDeploy) · Gradio 5

### 后端抽象层（核心工程亮点）

所有 pipeline 只依赖 `BaseBackend.generate(prompt) -> str` 接口，**通过 YAML
配置即可在四种后端间切换，应用层代码零修改**：

| 后端 | 用途 | 配置文件 |
|---|---|---|
| `OllamaBackend` | 本地开发 | `configs/ollama.yaml` |
| `LMDeployBackend` | A100 部署研究 | `configs/lmdeploy.yaml` |
| `GroqBackend` | HF Spaces 在线 demo | `configs/groq.yaml` |
| `HFInferenceBackend` | 备用 serverless 方案 | `configs/hf_spaces.yaml` |

---

## 研究成果（Qwen2.5-7B 在 A100 80GB 上）

### 部署性能对比

| 后端 | TTFT | 吞吐 | 平均响应 | VRAM |
|---|---|---|---|---|
| fp16 (pytorch) | 693.6 ms | 73.9 tok/s | 2.00 s | 64.55 GB |
| **AWQ (turbomind)** | **18.0 ms** | **166.9 tok/s** | **0.91 s** | 65.30 GB |

AWQ 实现了 **38.5× 的 TTFT 降低** 和 **2.26× 的吞吐提升**，且**质量无回退**。

### 质量评测（20 题 · 人工 0–2 分 / 维度评分，单题满分 10）

| 系统 | 后端 | 正确性 | 证据 | 可溯源 | 工具使用 | 效率 | 总分 /10 |
|---|---|---|---|---|---|---|---|
| Direct LLM | A100 / Qwen | 0.20 | 0.00 | 0.00 | 0.00 | 0.95 | 1.15 |
| **RAG Baseline** | **A100 / Qwen** | **1.75** | **1.90** | **1.95** | **2.00** | **1.90** | **9.50** |
| ~~Full Agent~~ | ~~A100 / Qwen~~ | ~~0.00~~ | ~~0.00~~ | ~~0.00~~ | ~~1.00~~ | ~~0.00~~ | ~~1.00~~ |
| **Full Agent（修复后）** | **Groq / Llama 3.1 8B** | **1.80** | 1.30 | 1.25 | **1.85** | 1.40 | **7.60** |

> **Full Agent 的"修复"—— 一次后端切换带来 7.6× 质量回升：**
> 原 A100 研究中，Qwen2.5-7B 输出 `<tool_call>...</tool_call>` XML 标签而非
> OpenAI 标准的 `tool_calls` JSON，LangChain 的 `ChatOpenAI` 客户端无法解析，
> Agent 的 tool 调用被静默丢弃，导致全维度接近 0 分。把 LLM 后端切换到 Groq
> 的 Llama 3.1 8B Instant（输出标准 `tool_calls`）后，**pipeline 和 tool 代码
> 零修改即恢复 Full Agent 全功能** —— 只新增了约 30 行 `GroqBackend` 封装 +
> 一份 YAML 配置。20 题评测中**幻觉率为 0%**。这正是后端抽象层的工程价值兑现。

---

## 本地运行

```bash
# 克隆代码
git clone https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment
cd Yelp_Project---rag-agent-deployment
pip install -e . --no-deps -r requirements.txt

# 方案 A — 免费 serverless（Groq Llama 3.1 8B Instant，推荐）
export GROQ_API_KEY="你的 key"
python app.py --config configs/groq.yaml

# 方案 B — 本地 Ollama
ollama pull qwen2.5:7b
python app.py --config configs/ollama.yaml

# 方案 C — A100 + LMDeploy turbomind（AWQ）
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct-AWQ \
    --server-port 23333 --backend turbomind --model-format awq
python app.py --config configs/lmdeploy.yaml
```

---

## 项目结构

```
yelp-rag-agent-deployment/
├── src/yelp_rag_agent/
│   ├── backends/        # Base + Ollama · LMDeploy · Groq · HF Inference
│   ├── tools/           # retrieval · stats · classifier · summarizer
│   ├── pipelines/       # rag_baseline · agent_runner (LangGraph)
│   └── evaluation/      # run_eval · metrics · rubric · test_questions
├── configs/             # ollama / lmdeploy / groq / hf_spaces 各 YAML
├── notebooks/           # colab_awq_deploy · colab_quality_eval · benchmark_analysis
├── scripts/             # serve_lmdeploy.sh · smoke_test.py · upload_to_hf.py
├── docs/                # project_overview.html（中英）+ 导出的 PDF
└── app.py               # Gradio 演示页（自动识别 HF Spaces 环境）
```

---

## 工程亮点

1. **后端抽象层**：同一份代码在 4 个 LLM 服务栈上运行；为在线 Demo 把
   Qwen→Llama 整体迁移仅新增约 30 行代码。
2. **线程安全的懒加载单例**：修复了 LangGraph 并行 tool 执行时，多线程同时
   调用 `SentenceTransformer.to('cuda')` 破坏 PyTorch CUDA 状态的竞态。
3. **Chunk 级 + 商家预过滤检索**：`business_to_indices` 映射让小评论量商家
   也能命中相关内容，避开"全局 Top-K + 后过滤"的反模式。
4. **JSON 字符串化工具输出**：绕开 Groq API 对空 list `ToolMessage.content`
   的严格校验拒收。

详见 `docs/project_overview_zh.html`（或导出的 `docs/Project_Report_zh.pdf`）。
