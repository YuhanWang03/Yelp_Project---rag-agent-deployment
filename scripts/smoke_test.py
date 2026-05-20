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
    parser.add_argument("--deepseek", action="store_true",
                        help="Run DeepSeek-V4 tool-call format gate test "
                             "(requires DEEPSEEK_API_KEY env var)")
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
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend

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
        import json as _json
        from yelp_rag_agent.tools.retrieval_tool import search_review_chunks_global
        raw = search_review_chunks_global.invoke(
            {"query": "rude staff terrible service", "top_k": 3}
        )
        results = _json.loads(raw) if isinstance(raw, str) else raw
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert "chunk_text" in results[0], "Missing 'chunk_text' key"
        assert "similarity" in results[0], "Missing 'similarity' key"

    def test_business_search():
        import json as _json
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
        raw = search_review_chunks_by_business.invoke(
            {"business_id": sample_biz, "query": "food quality", "top_k": 3}
        )
        results = _json.loads(raw) if isinstance(raw, str) else raw
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

    def test_backend_override():
        from yelp_rag_agent.backends import load_backend
        b = load_backend("configs/ollama.yaml", overrides={"model": "qwen2.5:14b"})
        assert b.model == "qwen2.5:14b", "Override not applied"

    def test_backend_factory_groq():
        import os
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.groq import GroqBackend
        # Inject a dummy key so construction succeeds even without env var
        b = load_backend("configs/groq.yaml",
                         overrides={"api_key": os.environ.get("GROQ_API_KEY", "dummy")})
        assert isinstance(b, GroqBackend)
        assert b.model == "llama-3.1-8b-instant"

    def test_backend_factory_deepseek_flash():
        import os
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend
        b = load_backend("configs/deepseek_v4_flash.yaml",
                         overrides={"api_key": os.environ.get("DEEPSEEK_API_KEY", "dummy")})
        assert isinstance(b, DeepSeekBackend)
        assert b.model == "deepseek-v4-flash"
        assert b.thinking is False

    def test_backend_factory_deepseek_pro():
        import os
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend
        b = load_backend("configs/deepseek_v4_pro.yaml",
                         overrides={"api_key": os.environ.get("DEEPSEEK_API_KEY", "dummy")})
        assert isinstance(b, DeepSeekBackend)
        assert b.model == "deepseek-v4-pro"

    def test_backend_factory_openai_judge():
        import os
        from yelp_rag_agent.backends import load_backend
        from yelp_rag_agent.backends.openai_backend import OpenAIBackend
        b = load_backend("configs/openai_judge.yaml",
                         overrides={"api_key": os.environ.get("OPENAI_API_KEY", "dummy")})
        assert isinstance(b, OpenAIBackend)
        assert b.base_url == "https://api.openai.com/v1"

    def test_deepseek_thinking_payload():
        # Unit test: verify thinking toggle produces the correct nested field.
        # No API call. V4 defaults thinking ON, so 'off' MUST send 'disabled'
        # explicitly — assert both directions to catch a silent no-op.
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend
        b = DeepSeekBackend(model="deepseek-v4-flash", api_key="dummy",
                            thinking=False)
        payload = b._build_payload("hi", temperature=0.1, max_tokens=10)
        assert payload["thinking"] == {"type": "disabled"}, \
            "thinking=False must send {'type': 'disabled'} (default is ON)"
        assert "temperature" in payload, \
            "temperature should be sent when thinking is off"
        assert payload["max_tokens"] == 10, \
            "thinking-off must pass max_tokens through unchanged"
        b.thinking = True
        payload = b._build_payload("hi", temperature=0.1, max_tokens=10)
        assert payload["thinking"] == {"type": "enabled"}, \
            "thinking=True must send {'type': 'enabled'}"
        assert "temperature" not in payload, \
            "temperature is ignored in thinking mode; should be omitted"
        # Thinking shares max_tokens with reasoning_content (emitted first), so
        # the budget must be bumped or the answer gets truncated to empty.
        assert payload["max_tokens"] > 10, \
            "thinking-on must add reasoning headroom to max_tokens"

    def test_deepseek_usage_and_cost():
        # Usage accumulator sums across calls + resets; cost math matches the
        # published V4 list prices. No API call.
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend
        from yelp_rag_agent.evaluation.metrics import compute_cost
        b = DeepSeekBackend(model="deepseek-v4-flash", api_key="dummy")
        b._accumulate_usage({"prompt_tokens": 300, "completion_tokens": 500,
                             "prompt_cache_hit_tokens": 256,
                             "completion_tokens_details": {"reasoning_tokens": 400}})
        u = b.get_usage()
        assert u["calls"] == 1 and u["input_tokens"] == 300
        assert u["output_tokens"] == 500 and u["reasoning_tokens"] == 400
        b.reset_usage()
        assert b.get_usage()["calls"] == 0, "reset_usage must zero the accumulator"
        c = compute_cost({"input_tokens": 1_000_000, "input_cached_tokens": 0,
                          "output_tokens": 1_000_000}, "deepseek-v4-flash")
        assert abs(c["total_cost_usd"] - 0.42) < 1e-9, f"flash cost wrong: {c}"
        assert compute_cost({"input_tokens": 1}, "llama-3.1-8b")["total_cost_usd"] is None

    check("Backend: DeepSeek usage accumulator + cost math",  test_deepseek_usage_and_cost)
    check("Backend: load OllamaBackend from YAML",          test_backend_factory_ollama)
    check("Backend: load GroqBackend from YAML",            test_backend_factory_groq)
    check("Backend: load DeepSeekBackend (V4-Flash) YAML",  test_backend_factory_deepseek_flash)
    check("Backend: load DeepSeekBackend (V4-Pro) YAML",    test_backend_factory_deepseek_pro)
    check("Backend: load OpenAIBackend (judge) YAML",       test_backend_factory_openai_judge)
    check("Backend: DeepSeek thinking flag injects payload", test_deepseek_thinking_payload)
    check("Backend: CLI override applied",                  test_backend_override)

    # ------------------------------------------------------------------
    # 6. summarizer_tool raises without backend (no LLM needed)
    # ------------------------------------------------------------------
    def test_summarizer_no_backend():
        from yelp_rag_agent.tools import summarizer_tool
        summarizer_tool._backend = None   # force unset
        summarizer_tool.set_last_chunks(
            [{"chunk_text": "x", "stars": 3, "business_id": "TEST"}]
        )
        try:
            summarizer_tool.summarize_evidence.invoke({"question": "test"})
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
        from yelp_rag_agent.tools.summarizer_tool import (
            set_backend, set_last_chunks, summarize_evidence,
        )
        b = load_backend(args.config)
        set_backend(b)
        set_last_chunks([
            {"chunk_text": "The wait was terrible and staff were rude.",
             "stars": 1, "business_id": "TEST001"},
        ])
        result = summarize_evidence.invoke({"question": "What do customers complain about?"})
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
    # 8. DeepSeek-V4 tool-call format gate (Stage E)
    #
    # CRITICAL: this is the gate test for Stage E. Verifies V4 emits
    # standard OpenAI `tool_calls` JSON (not Qwen-style <tool_call> XML
    # embedded in `content`). If this fails, Stage E2 pipelines cannot
    # rely on LangChain's tool_calls parser and the Qwen-era bug repeats.
    # ------------------------------------------------------------------
    def test_deepseek_tool_call_format():
        import os, requests
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        assert api_key, "DEEPSEEK_API_KEY not set"
        tool_schema = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model"      : "deepseek-v4-flash",
                "messages"   : [{"role": "user",
                                 "content": "What's the weather in Tokyo?"}],
                "tools"      : tool_schema,
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens" : 256,
            },
            timeout=60,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # The critical assertion: tool calls must surface as a structured
        # `tool_calls` list, NOT as XML/JSON glob inside `content`.
        assert msg.get("tool_calls"), (
            f"V4 did NOT return tool_calls field. message={msg}. "
            f"If tool intent is hidden in content, Stage E2 pipelines will "
            f"break the same way Qwen2.5 did."
        )
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather", \
            f"Wrong tool name: {tc['function']['name']}"

    check("DeepSeek-V4: tool_calls format (Stage E gate test)",
          test_deepseek_tool_call_format, skip=not args.deepseek)

    def test_deepseek_thinking_mode_live():
        # Verifies the thinking toggle ACTUALLY works against the live API in
        # both directions (2 calls). Catches the silent-default-on failure:
        # thinking=enabled must return reasoning_content; disabled must not.
        import os, requests
        from yelp_rag_agent.backends.deepseek import DeepSeekBackend
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        assert api_key, "DEEPSEEK_API_KEY not set"

        def call(thinking: bool) -> dict:
            b = DeepSeekBackend(model="deepseek-v4-flash", api_key=api_key,
                                thinking=thinking)
            payload = b._build_payload(
                "What is 17 * 23? Show your reasoning.",
                temperature=0.1, max_tokens=512,
            )
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload, timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]

        on  = call(True)
        off = call(False)
        assert on.get("reasoning_content"), (
            "thinking=enabled returned no reasoning_content — the toggle is "
            "not activating thinking mode (check field name/format)."
        )
        assert not off.get("reasoning_content"), (
            "thinking=disabled still returned reasoning_content — V4's "
            "default-on was not overridden; thinking-off eval arm is invalid."
        )

    check("DeepSeek-V4: thinking toggle live (on/off reasoning_content)",
          test_deepseek_thinking_mode_live, skip=not args.deepseek)

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
