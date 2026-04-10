"""
Central configuration for yelp-rag-agent.

All paths are derived from PROJECT_ROOT automatically.
Model/backend settings are loaded from a YAML config file at runtime
via load_config(); defaults below are used when no YAML is provided.
"""

from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Project root: src/yelp_rag_agent/ -> src/ -> project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Static paths (layout-dependent, never need to change)
# ---------------------------------------------------------------------------
VECTORSTORE_INDEX = PROJECT_ROOT / "vectorstore" / "review_chunks.index"
VECTORSTORE_META  = PROJECT_ROOT / "vectorstore" / "review_chunks.pkl"
CLASSIFIER_DIR    = PROJECT_ROOT / "artifacts"   / "roberta_5class_best"
DATA_PATH         = PROJECT_ROOT / "data" / "processed" / "yelp_reviews_sampled_50k.csv"
BUSINESS_JSON     = PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_business.json"
RESULTS_DIR       = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding model (must match what build_vectorstore.py used)
# ---------------------------------------------------------------------------
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 8

# ---------------------------------------------------------------------------
# Runtime backend config — populated by load_config() at startup
# ---------------------------------------------------------------------------
# These defaults mirror configs/ollama.yaml so the app works without
# an explicit --config argument.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"


def load_config(yaml_path: str) -> dict:
    """Load a backend YAML config file and return the raw dict."""
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
