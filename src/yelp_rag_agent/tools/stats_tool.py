"""
Tool 2 — Business Stats

Returns a structured statistical profile of one Yelp business based on
the 50k review dataset. Pure pandas — no LLM, no embeddings.

Output format:
{
    "business_id"       : "...",
    "review_count"      : 145,
    "avg_stars"         : 3.12,
    "star_distribution" : {"1": 20, "2": 18, "3": 35, "4": 40, "5": 32}
}
"""

from typing import Optional

import pandas as pd
from langchain.tools import tool

from yelp_rag_agent.config import DATA_PATH

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_df: Optional[pd.DataFrame] = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        print("[stats_tool] Loading review dataset …")
        _df = pd.read_csv(DATA_PATH, usecols=["review_id", "business_id", "stars"])
        _df["stars"] = _df["stars"].astype(float)
        print(f"[stats_tool] Loaded {len(_df):,} reviews, "
              f"{_df['business_id'].nunique():,} unique businesses")
    return _df


# ---------------------------------------------------------------------------
# Tool 2
# ---------------------------------------------------------------------------

@tool
def get_business_stats(business_id: str) -> dict:
    """
    Return a statistical profile of a Yelp business: review count,
    average star rating, and star distribution (1–5).

    Args:
        business_id:  Yelp business ID string.

    Returns:
        Dict with keys: business_id, review_count, avg_stars,
        star_distribution.
        Returns review_count=0 if the business is not found.
    """
    df     = _load_df()
    biz_df = df[df["business_id"] == business_id]

    if biz_df.empty:
        return {
            "business_id"      : business_id,
            "review_count"     : 0,
            "avg_stars"        : None,
            "star_distribution": {},
        }

    dist = (
        biz_df["stars"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    star_dist = {str(s): int(dist.get(s, 0)) for s in range(1, 6)}

    return {
        "business_id"      : business_id,
        "review_count"     : int(len(biz_df)),
        "avg_stars"        : round(float(biz_df["stars"].mean()), 2),
        "star_distribution": star_dist,
    }
