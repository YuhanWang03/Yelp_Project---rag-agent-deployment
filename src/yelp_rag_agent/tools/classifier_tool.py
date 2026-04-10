"""
Tool 3 — Review Classifier

Loads the RoBERTa 5-class model and runs inference on a single review text.

Output format:
{
    "predicted_stars"    : 4,
    "confidence"         : 0.87,
    "score_distribution" : {
        "1_star": 0.01, "2_star": 0.03, "3_star": 0.09,
        "4_star": 0.87, "5_star": 0.00
    }
}
"""

from typing import Optional

import torch
import torch.nn.functional as F
from langchain.tools import tool
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from yelp_rag_agent.config import CLASSIFIER_DIR

_MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None
_device: Optional[str] = None


def _load_model():
    global _tokenizer, _model, _device

    if _model is None:
        if not CLASSIFIER_DIR.exists():
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_DIR}. "
                "Run artifacts/step0_train_and_save.py first."
            )
        print(f"[classifier_tool] Loading RoBERTa from {CLASSIFIER_DIR} …")
        _tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
        _model     = AutoModelForSequenceClassification.from_pretrained(
            str(CLASSIFIER_DIR)
        )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model  = _model.to(_device)
        _model.eval()
        print(f"[classifier_tool] Model ready on {_device.upper()}")

    return _tokenizer, _model, _device


# ---------------------------------------------------------------------------
# Tool 3
# ---------------------------------------------------------------------------

@tool
def classify_review(text: str) -> dict:
    """
    Classify a Yelp review text and return a predicted star rating (1–5)
    using the fine-tuned RoBERTa model.

    Use this tool when you need to:
      - Verify the sentiment of a retrieved review chunk
      - Analyse a new review not in the dataset

    Args:
        text:  Raw review text (truncated to 512 tokens if needed).

    Returns:
        Dict with keys: predicted_stars (int 1-5), confidence (float),
        score_distribution (dict mapping label -> probability).
    """
    tokenizer, model, device = _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs           = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
    predicted_idx   = int(torch.argmax(logits, dim=-1).item())
    predicted_stars = predicted_idx + 1

    return {
        "predicted_stars"   : predicted_stars,
        "confidence"        : round(probs[predicted_idx], 4),
        "score_distribution": {
            f"{i+1}_star": round(probs[i], 4) for i in range(len(probs))
        },
    }
