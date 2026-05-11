"""
Stage 4 - Step 1: Build Chunk-Level Vector Store

Reads 50k Yelp reviews, splits long reviews into sentence-aware chunks,
encodes each chunk with all-MiniLM-L6-v2, and builds a FAISS index for
semantic retrieval.

Design decisions (from stage4_plan.md):
  - chunk-level indexing instead of full-review indexing, to avoid losing
    information from the second half of long reviews.
  - Pre-filtering by business_id: a business_to_indices mapping is stored
    alongside the embeddings so retrieval_tool.py can search only within
    a target business's chunks, without global Top-K + post-filter.

Output files:
  s4_agent/vectorstore/review_chunks.index   FAISS index (global search)
  s4_agent/vectorstore/review_chunks.pkl     chunk metadata + business map
                                             + raw embeddings (subset search)

Usage:
    python s4_agent/vectorstore/build_vectorstore.py
"""

import re
import gc
import pickle
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "yelp_reviews_sampled_50k.csv"
OUT_DIR      = PROJECT_ROOT / "s4_agent" / "vectorstore"
INDEX_PATH   = OUT_DIR / "review_chunks.index"
META_PATH    = OUT_DIR / "review_chunks.pkl"

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
# Reviews with fewer words than this threshold are kept as a single chunk.
# Longer reviews are split into sentence-aware chunks capped at this size.
CHUNK_MAX_WORDS = 180
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH     = 512   # sentences per encoding batch


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries using simple regex."""
    # Split after . ! ? followed by whitespace, keeping delimiter at end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def chunk_review(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """
    Return a list of text chunks for a single review.

    Short reviews  (word count <= max_words): returned as-is (1 chunk).
    Long reviews   (word count >  max_words): split into sentence-aligned
                   chunks, each capped at max_words.  No overlap is used
                   because adjacent chunks share the same review_id and
                   business_id in the metadata, so context is recoverable.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences   = _split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_wc  = 0

    for sent in sentences:
        sent_wc = len(sent.split())

        # If a single sentence exceeds the cap, force-split it by words
        if sent_wc > max_words:
            # flush current buffer first
            if current:
                chunks.append(" ".join(current))
                current, current_wc = [], 0
            # hard-split the long sentence
            sub_words = sent.split()
            for start in range(0, len(sub_words), max_words):
                chunks.append(" ".join(sub_words[start : start + max_words]))
            continue

        # Normal case: add sentence to current chunk if it fits
        if current_wc + sent_wc > max_words:
            chunks.append(" ".join(current))
            current, current_wc = [sent], sent_wc
        else:
            current.append(sent)
            current_wc += sent_wc

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Build chunk records from the CSV
# ---------------------------------------------------------------------------

def build_chunk_records(csv_path: Path) -> list[dict]:
    """
    Read the review CSV and return a flat list of chunk dicts.

    Each dict:
        chunk_idx   : int  — position in the flat list (= row in FAISS index)
        chunk_id    : str  — "{review_id}_{chunk_position}"
        review_id   : str
        business_id : str
        stars       : float
        chunk_text  : str  — text used for embedding
        full_text   : str  — original full review text
    """
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)

    records: list[dict] = []
    chunk_idx = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Chunking"):
        chunks = chunk_review(row.text, max_words=CHUNK_MAX_WORDS)
        for pos, chunk_text in enumerate(chunks):
            records.append(
                {
                    "chunk_idx"   : chunk_idx,
                    "chunk_id"    : f"{row.review_id}_{pos}",
                    "review_id"   : row.review_id,
                    "business_id" : row.business_id,
                    "stars"       : float(row.stars),
                    "chunk_text"  : chunk_text,
                    "full_text"   : row.text,
                }
            )
            chunk_idx += 1

    print(f"  {len(df):,} reviews  ->  {len(records):,} chunks "
          f"(avg {len(records)/len(df):.2f} chunks/review)")
    return records


# ---------------------------------------------------------------------------
# Encode with SentenceTransformer
# ---------------------------------------------------------------------------

def encode_chunks(
    records: list[dict],
    model_name: str = EMBED_MODEL,
    batch_size: int  = EMBED_BATCH,
) -> np.ndarray:
    """
    Encode chunk_text for every record.
    Returns float32 ndarray of shape (N, embedding_dim).
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [r["chunk_text"] for r in records]
    print(f"Encoding {len(texts):,} chunks (batch_size={batch_size}) ...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise -> inner product == cosine sim
    )
    print(f"Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    return embeddings.astype("float32")


# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact Inner-Product index.
    Because embeddings are L2-normalised, IP == cosine similarity.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Build business -> chunk indices mapping
# ---------------------------------------------------------------------------

def build_business_map(records: list[dict]) -> dict[str, list[int]]:
    """
    Return {business_id: [chunk_idx, ...]} for pre-filtered retrieval.
    This avoids global Top-K + post-filter, which would miss relevant
    chunks when the target business has few reviews.
    """
    mapping: dict[str, list[int]] = {}
    for r in records:
        mapping.setdefault(r["business_id"], []).append(r["chunk_idx"])
    print(f"Business map: {len(mapping):,} unique business IDs")
    return mapping


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    index: faiss.IndexFlatIP,
    records: list[dict],
    embeddings: np.ndarray,
    business_map: dict,
    index_path: Path,
    meta_path: Path,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving FAISS index -> {index_path}")
    faiss.write_index(index, str(index_path))

    payload = {
        "chunks"            : records,      # list of dicts
        "business_to_indices": business_map, # {biz_id: [chunk_idx]}
        "embeddings"        : embeddings,   # np.ndarray (N, 384) float32
    }
    print(f"Saving metadata + embeddings -> {meta_path}")
    with open(meta_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Report file sizes
    idx_mb  = index_path.stat().st_size / 1024 / 1024
    meta_mb = meta_path.stat().st_size  / 1024 / 1024
    print(f"  review_chunks.index : {idx_mb:.1f} MB")
    print(f"  review_chunks.pkl   : {meta_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Smoke test: verify retrieval works before declaring success
# ---------------------------------------------------------------------------

def smoke_test(index_path: Path, meta_path: Path) -> None:
    print("\n--- Smoke test ---")
    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    loaded_index = faiss.read_index(str(index_path))
    model        = SentenceTransformer(EMBED_MODEL)

    queries = [
        ("rude staff terrible service", None),
        ("slow service long wait time", None),
        ("amazing food delicious",       None),
    ]

    # Also pick a random business that has reviews
    sample_biz = list(data["business_to_indices"].keys())[0]
    queries.append(("good food", sample_biz))

    for query_text, biz_id in queries:
        q_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")

        if biz_id is None:
            # Global search via FAISS
            scores, idxs = loaded_index.search(q_vec, 3)
            label = "global"
        else:
            # Pre-filtered search via numpy subset
            biz_indices   = data["business_to_indices"][biz_id]
            subset_embeds = data["embeddings"][biz_indices]          # (M, 384)
            sims          = (subset_embeds @ q_vec.T).squeeze()      # (M,)
            top3_pos      = np.argsort(sims)[::-1][:3]
            idxs          = [[biz_indices[p] for p in top3_pos]]
            scores        = [[sims[p] for p in top3_pos]]
            label         = f"biz={biz_id[:12]}..."

        top_chunk = data["chunks"][idxs[0][0]]
        print(f"\n  Query  : '{query_text}'  [{label}]")
        print(f"  Score  : {scores[0][0]:.4f}")
        print(f"  Stars  : {top_chunk['stars']}")
        print(f"  Result : {top_chunk['chunk_text'][:120]}...")

    print("\nSmoke test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Stage 4 - Step 1: Build Chunk-Level Vector Store")
    print(f"Source : {DATA_PATH}")
    print(f"Output : {OUT_DIR}")
    print("=" * 60)

    # 1. Chunk
    records = build_chunk_records(DATA_PATH)

    # 2. Embed
    embeddings = encode_chunks(records)

    # 3. FAISS index
    index = build_faiss_index(embeddings)

    # 4. Business map
    business_map = build_business_map(records)

    # 5. Save
    save_artefacts(index, records, embeddings, business_map, INDEX_PATH, META_PATH)

    # 6. Free memory before smoke test
    del index, embeddings
    gc.collect()

    # 7. Smoke test (reloads from disk)
    smoke_test(INDEX_PATH, META_PATH)

    print("\n" + "=" * 60)
    print("Step 1 complete.")
    print(f"  {INDEX_PATH}")
    print(f"  {META_PATH}")
    print("\nNext: implement s4_agent/tools/retrieval_tool.py")


if __name__ == "__main__":
    main()
