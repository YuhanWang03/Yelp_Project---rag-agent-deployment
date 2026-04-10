"""
Tool 1 — Retrieval

Two LangChain tools for semantic search over the FAISS vector store:

  search_review_chunks_global(query, top_k)
      Full-corpus search. Use when no specific business is mentioned.

  search_review_chunks_by_business(business_id, query, top_k)
      Pre-filtered search restricted to one business's chunks.
      Avoids the global Top-K + post-filter anti-pattern for businesses
      with few reviews.

Both tools return a list of result dicts so that pipelines and the Agent
can decide how to format the output downstream.
"""

from functools import lru_cache
from typing import Optional

import faiss
import numpy as np
from langchain.tools import tool
from sentence_transformers import SentenceTransformer

from yelp_rag_agent.config import (
    VECTORSTORE_INDEX,
    VECTORSTORE_META,
    EMBED_MODEL,
    DEFAULT_TOP_K,
)

import pickle

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first call, reused afterwards
# ---------------------------------------------------------------------------

_store: Optional[dict] = None
_index: Optional[faiss.IndexFlatIP] = None
_embed_model: Optional[SentenceTransformer] = None


def _load_store() -> tuple[dict, faiss.IndexFlatIP, SentenceTransformer]:
    global _store, _index, _embed_model

    if _store is None:
        print("[retrieval_tool] Loading vector store …")
        with open(VECTORSTORE_META, "rb") as f:
            _store = pickle.load(f)
        _index = faiss.read_index(str(VECTORSTORE_INDEX))
        _embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"[retrieval_tool] Ready — {_index.ntotal:,} chunks, "
              f"{len(_store['business_to_indices']):,} businesses")

    return _store, _index, _embed_model


def _encode_query(query: str) -> np.ndarray:
    """Return a (1, 384) float32 normalised query vector."""
    _, _, model = _load_store()
    return model.encode([query], normalize_embeddings=True).astype("float32")


def _format_results(indices: list[int], scores: list[float],
                    store: dict) -> list[dict]:
    results = []
    for idx, score in zip(indices, scores):
        chunk = store["chunks"][idx]
        results.append({
            "chunk_idx"  : int(idx),
            "review_id"  : chunk["review_id"],
            "business_id": chunk["business_id"],
            "stars"      : float(chunk["stars"]),
            "chunk_text" : chunk["chunk_text"],
            "similarity" : round(float(score), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Tool 1a: global search
# ---------------------------------------------------------------------------

@tool
def search_review_chunks_global(query: str,
                                 top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search the full Yelp review corpus for chunks semantically similar to
    the query. Use this when no specific business_id is mentioned.

    Args:
        query:  Natural language search query.
        top_k:  Number of results to return (default 8).

    Returns:
        List of dicts with keys:
            chunk_idx, review_id, business_id, stars, chunk_text, similarity
    """
    store, index, _ = _load_store()
    q_vec = _encode_query(query)
    scores, idxs = index.search(q_vec, top_k)
    return _format_results(idxs[0].tolist(), scores[0].tolist(), store)


# ---------------------------------------------------------------------------
# Tool 1b: business-filtered search
# ---------------------------------------------------------------------------

@tool
def search_review_chunks_by_business(business_id: str,
                                      query: str,
                                      top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search reviews for a specific business for chunks semantically similar
    to the query. Use this when a business_id is available.

    Pre-filtering approach: loads only the target business's embeddings
    and computes cosine similarity on that subset, avoiding dilution for
    businesses with few reviews.

    Args:
        business_id:  Yelp business ID string.
        query:        Natural language search query.
        top_k:        Number of results to return (default 8).

    Returns:
        List of dicts with keys:
            chunk_idx, review_id, business_id, stars, chunk_text, similarity
        Returns an empty list if business_id is not found.
    """
    store, _, _ = _load_store()

    biz_indices = store["business_to_indices"].get(business_id)
    if not biz_indices:
        return []

    q_vec  = _encode_query(query)                    # (1, 384)
    subset = store["embeddings"][biz_indices]         # (M, 384)
    sims   = (subset @ q_vec.T).squeeze()             # (M,)

    if sims.ndim == 0:
        sims = np.array([float(sims)])

    k        = min(top_k, len(biz_indices))
    top_pos  = np.argsort(sims)[::-1][:k]
    top_idxs = [biz_indices[p] for p in top_pos]
    top_sims = [float(sims[p]) for p in top_pos]

    return _format_results(top_idxs, top_sims, store)
