"""
End-to-end paradigm smoke test (Stage E2.4).

Runs each reasoning paradigm once against a real backend (DeepSeek-V4 by
default) to verify the FULL chain works — plan -> tools -> solve, the
critique/revise loop, and ReAct tool-calling — and prints an answer preview
plus steps / llm_calls / latency per paradigm.

This is the first run that costs real API calls. On V4-Flash the whole
sweep is a fraction of a cent.

Usage:
    set DEEPSEEK_API_KEY=...
    python scripts/smoke_paradigms.py
    python scripts/smoke_paradigms.py --thinking
    python scripts/smoke_paradigms.py --paradigm reflection
    python scripts/smoke_paradigms.py --config configs/deepseek_v4_pro.yaml
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepseek_v4_flash.yaml")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable DeepSeek V4 thinking mode")
    parser.add_argument("--paradigm", default="all",
                        choices=["all", "react", "plan_and_solve", "rewoo", "reflection"])
    parser.add_argument("--question",
                        default="What do customers complain about most at this business?")
    parser.add_argument("--global", dest="use_global", action="store_true",
                        help="Run a global (no business_id) question instead")
    args = parser.parse_args()

    from yelp_rag_agent.backends import load_backend
    from yelp_rag_agent.tools.summarizer_tool import set_backend
    from yelp_rag_agent.tools.retrieval_tool import _load_store
    from yelp_rag_agent.pipelines.agent_runner import run_agent
    from yelp_rag_agent.pipelines.plan_and_solve import run_plan_and_solve
    from yelp_rag_agent.pipelines.rewoo import run_rewoo
    from yelp_rag_agent.pipelines.reflection import run_reflection

    backend = load_backend(args.config)
    set_backend(backend)
    print(f"Backend: {getattr(backend, 'model', '?')}  |  thinking={args.thinking}")

    biz = None
    if not args.use_global:
        store, _, _ = _load_store()
        biz = next(bid for bid, idxs in store["business_to_indices"].items()
                   if len(idxs) > 50)
    print(f"business_id: {biz or '(global)'}")
    print(f"question   : {args.question}")

    runners = {
        "react"         : lambda: run_agent(args.question, business_id=biz, thinking=args.thinking),
        "plan_and_solve": lambda: run_plan_and_solve(args.question, business_id=biz, thinking=args.thinking),
        "rewoo"         : lambda: run_rewoo(args.question, business_id=biz, thinking=args.thinking),
        "reflection"    : lambda: run_reflection(args.question, business_id=biz, thinking=args.thinking),
    }
    selected = list(runners) if args.paradigm == "all" else [args.paradigm]

    rows = []
    for name in selected:
        print(f"\n{'#'*64}\n# {name}\n{'#'*64}")
        try:
            r = runners[name]()
            ans = (r.get("final_answer") or "").replace("\n", " ").strip()
            print(f"\n  ANSWER PREVIEW: {ans[:300]}…")
            rows.append([name, r.get("steps", "?"), r.get("llm_calls", "?"),
                         r.get("elapsed_seconds", "?"), len(ans), "OK"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append([name, "-", "-", "-", 0, f"FAIL:{type(e).__name__}"])

    print(f"\n{'='*70}\nSUMMARY  (backend={getattr(backend,'model','?')}, thinking={args.thinking})\n{'='*70}")
    print(f"{'paradigm':<16}{'steps':>6}{'llm':>5}{'elapsed':>9}{'ans_len':>9}  status")
    for name, steps, llm, el, alen, status in rows:
        print(f"{name:<16}{str(steps):>6}{str(llm):>5}{str(el):>9}{str(alen):>9}  {status}")

    if any(str(r[5]).startswith("FAIL") for r in rows):
        print("\nOne or more paradigms FAILED.")
        sys.exit(1)
    print("\nAll selected paradigms ran successfully.")


if __name__ == "__main__":
    main()
