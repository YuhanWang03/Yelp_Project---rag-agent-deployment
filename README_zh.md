# 🍽️ Yelp 商业智能 Agent

一个基于 RAG 的问答系统，覆盖 **60,823 条 Yelp 评论 chunk**（**50,000 条评论 · 160 家商家**），对比 **六种**评论问答方法——两个基线 + 四种智能体推理范式——全部基于 **DeepSeek-V4**。

> **🚀 在线 Demo：** https://huggingface.co/spaces/YUHAN03/yelp-rag-agent
> *（运行在 DeepSeek-V4-Flash 上）*

> **📄 完整报告（PDF）：**
> [Project_Report.pdf](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/download/v1.0/Project_Report.pdf)
> · [Project_Report_zh.pdf（中文）](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/download/v1.0/Project_Report_zh.pdf)
> &mdash; 完整的推理范式对比研究。源 HTML 见 `docs/project_overview_zh.html`。

---

## 对比的六个系统

| 系统 | 类型 | 做什么 |
|---|---|---|
| **Direct LLM** | 基线 | 仅凭参数化记忆作答——无检索 |
| **RAG Baseline** | 基线 | 固定流水线：stats → FAISS 检索 → 结构化合成 |
| **ReAct** | 范式 | 边推理边行动，自主调度工具（LangGraph） |
| **Plan-and-Solve** | 范式 | 先规划全部工具调用 → 顺序执行 → solve |
| **ReWOO** | 范式 | 规划 → **并行**执行 → solve |
| **Reflection** | 范式 | 作答 → 自我批判 → 修正 |

四种范式共用相同的工具和检索；唯一变化的是推理结构。框架（LangGraph）和模型（DeepSeek-V4）保持不变，使范式成为唯一变量。

---

## 核心发现 —— 质量饱和，范式选择是效率决策

每个系统都在相同的 20 题上、跨 DeepSeek-V4-Flash 与 V4-Pro、两种 thinking 模式各跑一遍（共 320 条答案），由 LLM-as-judge 在四个质量维度上评分（0–2 分/维度，满分 8）。

| 系统 | 质量 /8 | 延迟 (s) | 成本 ($/题) | 输出 tok |
|---|---|---|---|---|
| Direct LLM | 0.2 | 3.8 | 0.00007 | 213 |
| **RAG Baseline** | **7.8** | **3.2** | **0.00016** | 286 |
| ReAct | 8.0 | 11.8 | 0.00118 | 765 |
| **Plan-and-Solve** | **7.9** | **6.4** | **0.00040** | 433 |
| **ReWOO** | **7.9** | **6.2** | **0.00036** | 404 |
| Reflection | 7.9 | 10.8 | 0.00081 | 704 |

*（DeepSeek-V4-Flash，thinking 关。）*

- **决定质量的是检索，不是推理结构。** 无检索的 Direct LLM 崩到 0.2/8；所有带检索的系统都达到 **7.8–8.0/8**，统计上无法区分。
- **范式的取舍在成本和延迟，不在质量。** ReAct 最贵（每轮重发历史）；Plan-and-Solve 和 ReWOO 最精简。
- **thinking 税真实且无回报。** DeepSeek-V4 thinking 模式多花 ~3–4× 延迟、~2× 成本，**质量零增益**（Pro：+0.00/8）。V4-Pro 成本约为 V4-Flash 的 10×，质量相同。
- **建议：** Flash 上的 Plan-and-Solve 或 ReWOO、thinking 关——以最低成本/延迟拿到天花板质量。

> **稳健性 —— 自偏好偏差已排除：** 为检验"V4 评 V4"的疑虑，全部 320 条答案用另一个模型族的独立裁判（OpenAI gpt-4o-mini）重评。两套裁判高度一致（Pearson **r = 0.97**，平均绝对差 **0.11/8**），结论不依赖裁判选择。
>
> **局限：** 0–2 量表在顶端饱和（任务偏易的属性，非裁判缺陷）；未存检索 chunks，groundedness 按内部一致性评判。

---

## 系统架构

```
用户问题
     │
     ├─► Direct LLM       — 仅凭记忆作答（基线）
     ├─► RAG Baseline     — Stats → FAISS 检索 → 合成（固定流水线）
     ├─► ReAct            — 边推理边行动，自主调用工具
     ├─► Plan-and-Solve   — 规划 → 顺序执行 → solve
     ├─► ReWOO            — 规划 → 并行执行 → solve
     └─► Reflection       — 作答 → 自我批判 → 修正
```

**技术栈：** FAISS · sentence-transformers (all-MiniLM-L6-v2) · LangGraph ·
LangChain · DeepSeek-V4 (Flash/Pro) · Gradio 5

### 后端抽象层

所有 pipeline 只依赖 `BaseBackend.generate(prompt) -> str` 接口。具体后端通过
YAML 配置即可互换——**迁移时无需改任何应用层代码**：

| 后端 | 用途 | 配置 |
|---|---|---|
| `DeepSeekBackend` | 范式研究 + 在线 demo | `configs/deepseek_v4_flash.yaml` / `_pro.yaml` |
| `OllamaBackend` | 本地开发 | `configs/ollama.yaml` |
| `GroqBackend` | 备用 serverless | `configs/groq.yaml` |
| `LMDeployBackend` | 自托管 OpenAI 兼容服务 | `configs/lmdeploy.yaml` |

---

## 运行

```bash
git clone https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment
cd Yelp_Project---rag-agent-deployment
pip install -e . --no-deps -r requirements.txt

# DeepSeek-V4（推荐——研究和在线 demo 都用它）
export DEEPSEEK_API_KEY="你的key"
python app.py --config configs/deepseek_v4_flash.yaml

# 本地 Ollama（离线开发）
ollama pull qwen2.5:7b
python app.py --config configs/ollama.yaml
```

在 **System / Reasoning Paradigm** 下拉框里选范式；切换 **Thinking mode**
（仅 DeepSeek-V4）做对比。

---

## 复现研究

```bash
# 1. 跑全矩阵（4 范式 × thinking 开/关 × 20 题）+ 基线
python -m yelp_rag_agent.evaluation.run_eval --run \
    --config configs/deepseek_v4_flash.yaml \
    --paradigm full_agent,plan_and_solve,rewoo,reflection --thinking both \
    --output paradigm_v4flash.csv

# 2. LLM-as-judge 评分（4 个质量维度）
python scripts/llm_judge.py

# 3. 聚合成规范 JSON
python scripts/aggregate_paradigm_study.py

# 4. 生成分析 + 报告章节
jupyter notebook notebooks/paradigm_analysis.ipynb
python scripts/build_paradigm_report.py

# 可选：独立裁判交叉验证
python scripts/llm_judge.py --judge-config configs/openai_judge.yaml --judge-tag openai
python scripts/compare_judges.py
```

---

## 项目结构

```
yelp-rag-agent-deployment/
├── src/yelp_rag_agent/
│   ├── backends/        # Base + DeepSeek · Ollama · Groq · LMDeploy · HF Inference
│   ├── tools/           # retrieval · stats · classifier · summarizer
│   ├── pipelines/       # rag_baseline · agent_runner (ReAct) · plan_and_solve · rewoo · reflection
│   └── evaluation/      # run_eval · metrics · paradigm_figures · rubric · test_questions
├── configs/             # deepseek_v4_flash/_pro · ollama · groq · lmdeploy · hf_spaces · *_judge
├── notebooks/           # paradigm_analysis.ipynb
├── scripts/             # llm_judge · aggregate_paradigm_study · build_paradigm_report · compare_judges · smoke_test
├── docs/                # project_overview.html（中英）+ 导出的 PDF
└── app.py               # Gradio demo（6 路范式选择器 + thinking 开关）
```

---

## 工程亮点

1. **一个接口下的多种推理范式** —— ReAct、Plan-and-Solve、ReWOO、Reflection
   复用同一套四个工具和共享 planner；只有执行/精炼结构不同。
2. **DeepSeek-V4 thinking 模式处理** —— thinking 与推理共享 `max_tokens` 预算
   （推理先生成），后端额外加推理余量避免答案被截断；ReAct 降级为 thinking 关，
   因为 V4 要求多轮工具调用回传 `reasoning_content`（推理模型 × agent 工具链的落差）。
3. **Token 与成本核算** —— 逐次记录 input/output/reasoning token 和美元成本
   （ReAct 经 LangChain `usage_metadata`），支撑效率对比。
4. **跨模型裁判验证** —— 用 OpenAI gpt-4o-mini 作独立裁判交叉验证（r=0.97），
   排除"V4 评 V4"的自偏好偏差。

完整研究与工程细节见 `docs/project_overview_zh.html` 或 GitHub Releases 上的
[Project_Report PDF](https://github.com/YuhanWang03/Yelp_Project---rag-agent-deployment/releases/latest)。
