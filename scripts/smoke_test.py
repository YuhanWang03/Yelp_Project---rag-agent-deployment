"""
Stage A/B Smoke Test

Verifies that the new package structure works correctly.

Usage:
    python scripts/smoke_test.py                              # skip LLM tests
    python scripts/smoke_test.py --full                       # requires Ollama
    python scripts/smoke_test.py --config configs/ollama.yaml --full

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import argparse
import sys
import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn, skip: bool = False):
    if skip:
        print(f"{SKIP} {name}")
        results.append((name, True, "skipped"))
        return
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, ""))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"       {e}")
        results.append((name, False, str(e)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true",
                        help="Run tests that require a running LLM backend")
    parser.add_argument("--config", default="configs/ollama.yaml",
                        help="Backend config for --full tests")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("Yelp RAG Agent — Smoke Test")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # 1. Package imports
    # ------------------------------------------------------------------
    def test_import_config():
        from yelp_rag_agent.config import (
            PROJECT_ROOT, VECTORSTORE_INDEX, VECTORSTORE_META,
            CLASSIFIER_DIR, DATA_PATH, EMBED_MODEL,
        )
        assert PROJECT_ROOT.exists(), f"PROJECT_ROOT not found: {PROJECT_ROOT}"

    def test_import_backends():
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.base import BaseBackend
        from yelp_rag_agent.backends.ollama import OllamaBackend
        from yelp_rag_agent.backends.lmdeploy import LMDeployBackend

    def test_import_tools():
        from yelp_rag_agent.tools.retrieval_tool import (
            search_review_chunks_global, search_review_chunks_by_business,
        )
        from yelp_rag_agent.tools.stats_tool import get_business_stats
        from yelp_rag_agent.tools.classifier_tool import classify_review
        from yelp_rag_agent.tools.summarizer_tool import summarize_evidence, set_backend

    def test_import_pipelines():
        from yelp_rag_agent.pipelines.rag_baseline import run_rag_pipeline
        from yelp_rag_agent.pipelines.agent_runner import run_agent

    def test_import_app():
        # Just verify app.py can be imported without crashing at module level.
        # We don't actually launch it.
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "app", pathlib.Path(__file__).parent.parent / "app.py"
        )
        # We only check that the file exists and is valid Python syntax
        app_path = pathlib.Path(__file__).parent.parent / "app.py"
        assert app_path.exists(), "app.py not found"
        compile(app_path.read_text(encoding="utf-8"), "app.py", "exec")

    check("Package: config imports",    test_import_config)
    check("Package: backends imports",  test_import_backends)
    check("Package: tools imports",     test_import_tools)
    check("Package: pipelines imports", test_import_pipelines)
    check("Package: app.py syntax",     test_import_app)

    # ------------------------------------------------------------------
    # 2. File presence
    # ------------------------------------------------------------------
    def test_vectorstore_files():
        from yelp_rag_agent.config import VECTORSTORE_INDEX, VECTORSTORE_META
        assert VECTORSTORE_INDEX.exists(), f"Missing: {VECTORSTORE_INDEX}"
        assert VECTORSTORE_META.exists(),  f"Missing: {VECTORSTORE_META}"

    def test_data_file():
        from yelp_rag_agent.config import DATA_PATH
        assert DATA_PATH.exists(), f"Missing: {DATA_PATH}"

    def test_business_json():
        from yelp_rag_agent.config import BUSINESS_JSON
        assert BUSINESS_JSON.exists(), f"Missing: {BUSINESS_JSON}"

    check("Files: vectorstore index + pkl", test_vectorstore_files)
    check("Files: review CSV",             test_data_file)
    check("Files: business JSON",          test_business_json)

    # ------------------------------------------------------------------
    # 3. Retrieval tool (no LLM needed)
    # ------------------------------------------------------------------
    def test_global_search():
        from yelp_rag_agent.tools.retrieval_tool import search_review_chunks_global
        results = search_review_chunks_global.invoke(
            {"query": "rude staff terrible service", "top_k": 3}
        )
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert "chunk_text" in results[0], "Missing 'chunk_text' key"
        assert "similarity" in results[0], "Missing 'similarity' key"

    def test_business_search():
        from yelp_rag_agent.tools.retrieval_tool import (
            search_review_chunks_by_business,
            _load_store,
        )
        store, _, _ = _load_store()
        # Pick any business with >5 chunks
        sample_biz = next(
            bid for bid, idxs in store["business_to_indices"].items()
            if len(idxs) > 5
        )
        results = search_review_chunks_by_business.invoke(
            {"business_id": sample_biz, "query": "food quality", "top_k": 3}
        )
        assert len(results) >= 1, "Expected at least 1 result"
        assert results[0]["business_id"] == sample_biz, "Wrong business returned"

    check("Retrieval: global search (3 chunks)", test_global_search)
    check("Retrieval: business-filtered search", test_business_search)

    # ------------------------------------------------------------------
    # 4. Stats tool (no LLM needed)
    # ------------------------------------------------------------------
    def test_stats_tool():
        from yelp_rag_agent.tools.stats_tool import get_business_stats
        from yelp_rag_agent.tools.retrieval_tool import _load_store
        store, _, _ = _load_store()
        sample_biz  = next(
            bid for bid, idxs in store["business_to_indices"].items()
            if len(idxs) > 50
        )
        stats = get_business_stats.invoke({"business_id": sample_biz})
        assert stats["review_count"] > 0,  "review_count should be > 0"
        assert stats["avg_stars"]    > 0,  "avg_stars should be > 0"
        assert len(stats["star_distribution"]) == 5, "star_distribution should have 5 keys"

    check("Stats: get_business_stats", test_stats_tool)

    # ------------------------------------------------------------------
    # 5. Backend factory (no server needed)
    # ------------------------------------------------------------------
    def test_backend_factory_ollama():
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.ollama import OllamaBackend
        b = load_backend("configs/ollama.yaml")
        assert isinstance(b, OllamaBackend)
        assert b.model == "qwen2.5:7b"

    def test_backend_factory_lmdeploy():
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.lmdeploy import LMDeployBackend
        b = load_backend("configs/lmdeploy.yaml")
        assert isinstance(b, LMDeployBackend)

    def test_backend_override():
        from yelp_rag_agent.backends import load_backend
        b = load_backend("configs/ollama.yaml", overrides={"model": "qwen2.5:14b"})
        assert b.model == "qwen2.5:14b", "Override not applied"

    check("Backend: load OllamaBackend from YAML",   test_backend_factory_ollama)
    check("Backend: load LMDeployBackend from YAML", test_backend_factory_lmdeploy)
    check("Backend: CLI override applied",           test_backend_override)

    # ------------------------------------------------------------------
    # 6. summarizer_tool raises without backend (no LLM needed)
    # ------------------------------------------------------------------
    def test_summarizer_no_backend():
        from yelp_rag_agent.tools import summarizer_tool
        summarizer_tool._backend = None   # force unset
        try:
            summarizer_tool.summarize_evidence.invoke(
                {"question": "test", "evidence_chunks": [{"chunk_text": "x", "stars": 3}]}
            )
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError:
            pass   # expected

    check("Summarizer: raises RuntimeError without backend",
          test_summarizer_no_backend)

    # ------------------------------------------------------------------
    # 7. Full LLM tests (requires running backend)
    # ------------------------------------------------------------------
    def test_backend_generate():
        from yelp_rag_agent.backends import load_backend
        b   = load_backend(args.config)
        out = b.generate("Reply with exactly: OK", temperature=0, max_tokens=10)
        assert isinstance(out, str) and len(out) > 0, "Empty response from backend"

    def test_summarize_with_backend():
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.tools.summarizer_tool import set_backend, summarize_evidence
        b = load_backend(args.config)
        set_backend(b)
        result = summarize_evidence.invoke({
            "question": "What do customers complain about?",
            "evidence_chunks": [
                {"chunk_text": "The wait was terrible and staff were rude.",
                 "stars": 1, "business_id": "TEST001"},
            ]
        })
        assert "main_findings" in result, "Missing 'main_findings'"
        assert len(result["main_findings"]) > 0, "main_findings is empty"

    def test_rag_pipeline_end_to_end():
        from yelp_rag_agent.tools.retrieval_tool import _load_store
        from yelp_rag_agent.pipelines.rag_baseline import run_rag_pipeline
        store, _, _ = _load_store()
        sample_biz  = next(
            bid for bid, idxs in store["business_to_indices"].items()
            if len(idxs) > 50
        )
        result = run_rag_pipeline(
            "What do customers say about this business?",
            business_id=sample_biz, top_k=3,
        )
        assert result["mode"] == "business"
        assert len(result["retrieved_chunks"]) > 0
        assert len(result["synthesis"]["main_findings"]) > 0

    check("LLM: backend.generate() returns non-empty string",
          test_backend_generate, skip=not args.full)
    check("LLM: summarize_evidence returns structured dict",
          test_summarize_with_backend, skip=not args.full)
    check("LLM: RAG pipeline end-to-end (Flow A)",
          test_rag_pipeline_end_to_end, skip=not args.full)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    passed  = sum(1 for _, ok, note in results if ok and note != "skipped")
    skipped = sum(1 for _, _, note in results if note == "skipped")
    failed  = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed  |  {skipped} skipped  |  {failed} failed")
    print(f"{'='*55}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
