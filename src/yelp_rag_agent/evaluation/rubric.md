# Stage 4 Evaluation Rubric

## Overview

Each answer is scored on 5 dimensions (0–2 scale).
All three systems are evaluated on identical questions.

**Systems compared:**
- `direct_llm` — Qwen2.5-7B answers from parametric memory only, no tools
- `rag_baseline` — Fixed pipeline: stats → retrieve → summarise
- `full_agent` — LangGraph ReAct: autonomous tool selection

---

## Scoring Dimensions

### 1. Answer Correctness (0–2)
Does the answer address the question and provide a relevant, accurate response?

| Score | Criteria |
|-------|----------|
| 2 | Directly and fully answers the question; key points are correct |
| 1 | Partially answers the question; some relevant content but incomplete or slightly off-topic |
| 0 | Does not answer the question, is off-topic, or is clearly wrong |

### 2. Evidence Support (0–2)
Does the answer cite specific review excerpts or data to support its claims?

| Score | Criteria |
|-------|----------|
| 2 | Provides 2+ specific review quotes or data points (e.g. star counts, named excerpts) |
| 1 | References reviews in general terms but without direct quotes or specific data |
| 0 | No reference to any review content; purely generic statements |

### 3. Groundedness / Faithfulness (0–2)
Are the answer's claims actually supported by the retrieved review evidence?
*(For `direct_llm`, score based on whether claims are plausible vs. fabricated.)*

| Score | Criteria |
|-------|----------|
| 2 | All major claims can be traced to specific retrieved chunks or stated facts |
| 1 | Most claims are grounded; 1–2 statements appear to go beyond the evidence |
| 0 | Multiple claims are ungrounded or contradict the retrieved evidence (hallucination) |

### 4. Tool Use Appropriateness (0–2)
*(For `direct_llm`: always 0 — no tools available.)*
Did the system call the right tools in a sensible order for this question type?

| Score | Criteria |
|-------|----------|
| 2 | Called all necessary tools; tool selection and order is logical |
| 1 | Called relevant tools but missed one useful tool or used a suboptimal order |
| 0 | Called wrong tools, skipped essential tools, or tool input had critical errors (e.g. wrong business_id) |

### 5. Efficiency (0–2)
How fast and lean was the response?
*(Scored relative to other systems on the same question.)*

| Score | Criteria |
|-------|----------|
| 2 | Fastest response with fewest tool calls needed to answer the question |
| 1 | Reasonable response time; no obviously redundant tool calls |
| 0 | Notably slow, or made redundant/unnecessary tool calls |

---

## Question Type–Specific Notes

### Complaint Mining
- A good answer names **specific complaint categories** (e.g. "long wait times", "rude staff"), not just "bad service".
- Evidence Support requires at least one direct quote from a negative review.

### Aspect Analysis
- The answer should focus on the **requested aspect** (service / food / price / etc.).
- Off-topic content about other aspects should not receive Evidence Support credit.

### Business Profiling
- A good profile covers **both positive and negative** aspects even if one dominates.
- Answers that only list negatives for a low-rated business should lose 1 point on Correctness.

### Cross-Business Pattern Search
- `direct_llm` is expected to score 0 on Evidence Support because it has no access to the corpus.
- The interesting comparison is `rag_baseline` vs `full_agent` on Groundedness and Correctness.

---

## How to Score

1. Open `s4_agent/results/eval_results.csv`.
2. For each row, read the `answer` column.
3. For `rag_baseline` and `full_agent`, also read `tools_called` and `tool_count`.
4. Fill in the five score columns: `score_correctness`, `score_evidence`, `score_groundedness`, `score_tool_use`, `score_efficiency`.
5. Add brief notes in the `notes` column for any surprising or borderline cases.
6. Save. Run `python s4_agent/evaluation/run_eval.py --summarise` to compute aggregate tables.

---

## Aggregate Metrics (computed by run_eval.py)

After human scoring, the script computes per-system averages:

| Metric | Formula |
|--------|---------|
| Total Score | Sum of all 5 dimensions (max 10) |
| Avg Correctness | Mean across all questions |
| Avg Evidence | Mean across all questions |
| Avg Groundedness | Mean across all questions |
| Avg Tool Use | Mean across all questions (N/A for direct_llm) |
| Avg Efficiency | Mean across all questions |
| Evidence Rate | % of answers with score_evidence >= 1 |
| Hallucination Rate | % of answers with score_groundedness == 0 |
