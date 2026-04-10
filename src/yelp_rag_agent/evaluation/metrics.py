"""
Deployment Performance Metrics

Measures how fast and memory-efficient an LLM backend is.
This module answers "how fast / how cheap" — NOT "how good".
Answer quality is evaluated separately via run_eval.py + human scoring.

Metrics collected:
    - VRAM usage (GB): load and peak
    - TTFT (ms): time to first token
    - Throughput (tokens/s): estimated from batch of prompts
    - Average response time (s)

Designed to run inside a Colab A100 notebook.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# VRAM helpers (requires nvidia-smi)
# ---------------------------------------------------------------------------

def get_vram_gb() -> dict:
    """
    Parse nvidia-smi output and return current VRAM usage.

    Returns:
        {"used_gb": float, "total_gb": float}
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        used, total = map(int, out.split(", "))
        return {
            "used_gb" : round(used  / 1024, 2),
            "total_gb": round(total / 1024, 2),
        }
    except Exception as e:
        return {"used_gb": None, "total_gb": None, "error": str(e)}


def wait_for_server(base_url: str, timeout: int = 120,
                    poll_interval: int = 5) -> bool:
    """
    Poll the /v1/models endpoint until the server is ready or timeout expires.

    Returns True if server is ready, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# TTFT — Time To First Token
# ---------------------------------------------------------------------------

def measure_ttft(base_url: str, model: str,
                 prompt: str = "Hello, who are you?") -> float:
    """
    Measure Time To First Token (ms) using streaming.

    Sends a streaming chat request and records elapsed time
    until the first chunk arrives.

    Returns:
        TTFT in milliseconds, or -1.0 on error.
    """
    t0 = time.perf_counter()
    try:
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model"    : model,
                "messages" : [{"role": "user", "content": prompt}],
                "stream"   : True,
                "max_tokens": 1,
            },
            stream=True,
            timeout=60,
        ) as resp:
            for chunk in resp.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    return round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------------------
# Throughput — tokens/s and average response time
# ---------------------------------------------------------------------------

def measure_throughput(base_url: str, model: str,
                       prompts: list[str],
                       max_tokens: int = 200) -> dict:
    """
    Run a batch of prompts and measure average response time and
    estimated tokens/s.

    Args:
        base_url:   LMDeploy server URL.
        model:      Model name string.
        prompts:    List of prompt strings to send.
        max_tokens: Max tokens per response.

    Returns:
        {
            "avg_response_s" : float,
            "estimated_tps"  : float,   # tokens per second (word-level estimate)
            "n_samples"      : int,
            "errors"         : int,
        }
    """
    times, word_counts, errors = [], [], 0

    for prompt in prompts:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model"    : model,
                    "messages" : [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            word_counts.append(len(content.split()))
        except Exception:
            errors += 1

    if not times:
        return {"avg_response_s": None, "estimated_tps": None,
                "n_samples": len(prompts), "errors": errors}

    return {
        "avg_response_s": round(sum(times) / len(times), 2),
        "estimated_tps" : round(sum(word_counts) / sum(times), 1),
        "n_samples"     : len(prompts),
        "errors"        : errors,
    }


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS = [
    "What do customers commonly complain about at restaurants?",
    "Summarize the typical experience at a 1-star rated business.",
    "What aspects of service do customers mention most in negative reviews?",
    "Describe patterns in reviews that mention wait time.",
    "What do customers say about food quality in positive reviews?",
]


def run_deployment_experiment(
    experiment_id: str,
    model: str,
    backend_label: str,
    base_url: str = "http://localhost:23333",
    vram_baseline_gb: Optional[float] = None,
) -> dict:
    """
    Run a complete deployment performance experiment for one model/backend.

    Args:
        experiment_id:      Short identifier, e.g. "lmdeploy_fp16".
        model:              Model name string for the API.
        backend_label:      Human-readable label, e.g. "lmdeploy-pytorch".
        base_url:           Server URL.
        vram_baseline_gb:   VRAM used before loading model (for net calculation).

    Returns:
        Dict with all collected metrics.
    """
    print(f"\n[metrics] Running experiment: {experiment_id}")

    # VRAM after model load
    vram_after = get_vram_gb()
    vram_load  = (
        round(vram_after["used_gb"] - vram_baseline_gb, 2)
        if vram_baseline_gb is not None and vram_after["used_gb"] is not None
        else vram_after["used_gb"]
    )

    # TTFT
    print("[metrics] Measuring TTFT …")
    ttft = measure_ttft(base_url, model)

    # Throughput
    print("[metrics] Measuring throughput …")
    throughput = measure_throughput(base_url, model, BENCHMARK_PROMPTS)

    # VRAM peak (after inference)
    vram_peak = get_vram_gb()

    result = {
        "experiment_id"  : experiment_id,
        "model"          : model,
        "backend"        : backend_label,
        "vram_load_gb"   : vram_load,
        "vram_peak_gb"   : vram_peak.get("used_gb"),
        "ttft_ms"        : ttft,
        "avg_response_s" : throughput["avg_response_s"],
        "estimated_tps"  : throughput["estimated_tps"],
        "n_samples"      : throughput["n_samples"],
        "errors"         : throughput["errors"],
    }

    print(f"[metrics] Done: {result}")
    return result


def save_results(results: list[dict], path: str = "results/deployment_perf.json",
                 hardware: str = "Colab A100 80GB") -> None:
    """Save a list of experiment results to JSON."""
    output = {
        "hardware"   : hardware,
        "experiments": results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[metrics] Results saved to {path}")
