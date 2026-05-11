"""
One-shot upload script for Hugging Face Hub deployment.

Creates / refreshes:
  - Dataset repo  : <user>/yelp-rag-data         (vectorstore + CSV + business JSON)
  - Model repo    : <user>/yelp-roberta-5class   (fine-tuned RoBERTa)

The HF Space itself is created via huggingface_hub later in a separate step.

Token is read from .hf_token in the project root (gitignored).

Usage:
    python scripts/upload_to_hf.py --user YUHAN03
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_token() -> str:
    token_path = PROJECT_ROOT / ".hf_token"
    if not token_path.exists():
        sys.exit(f"ERROR: {token_path} not found. Create it with your HF token.")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token.startswith("hf_"):
        sys.exit("ERROR: .hf_token does not look like a valid HF token (must start 'hf_').")
    return token


def upload_dataset(user: str, token: str) -> None:
    repo_id = f"{user}/yelp-rag-data"
    print(f"\n=== Dataset repo: {repo_id} ===")

    create_repo(repo_id=repo_id, repo_type="dataset", token=token,
                private=False, exist_ok=True)
    print(f"  Repo ready (created or already existed).")

    api = HfApi(token=token)

    files = [
        (PROJECT_ROOT / "vectorstore" / "review_chunks.index",
         "vectorstore/review_chunks.index"),
        (PROJECT_ROOT / "vectorstore" / "review_chunks.pkl",
         "vectorstore/review_chunks.pkl"),
        (PROJECT_ROOT / "data" / "processed" / "yelp_reviews_sampled_50k.csv",
         "yelp_reviews_sampled_50k.csv"),
        (PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_business.json",
         "yelp_academic_dataset_business.json"),
    ]

    for local, remote in files:
        if not local.exists():
            print(f"  SKIP (missing): {local}")
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        print(f"  Uploading {remote} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="dataset",
        )

    readme = (
        "---\nlicense: apache-2.0\ntask_categories:\n- text-retrieval\n"
        "language:\n- en\n---\n\n"
        "# Yelp RAG Agent Data\n\n"
        "Vector store and source data for the Yelp Business Intelligence "
        "Agent demo.\n\n"
        "- `vectorstore/review_chunks.index` — FAISS index (60,823 chunks)\n"
        "- `vectorstore/review_chunks.pkl` — chunk metadata + embeddings + "
        "business→indices map\n"
        "- `yelp_reviews_sampled_50k.csv` — 50k sampled Yelp reviews\n"
        "- `yelp_academic_dataset_business.json` — Yelp business metadata\n\n"
        "Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  README uploaded.")
    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


def upload_model(user: str, token: str) -> None:
    repo_id = f"{user}/yelp-roberta-5class"
    print(f"\n=== Model repo: {repo_id} ===")

    create_repo(repo_id=repo_id, repo_type="model", token=token,
                private=False, exist_ok=True)
    print(f"  Repo ready.")

    local_dir = PROJECT_ROOT / "artifacts" / "roberta_5class_best"
    if not local_dir.exists():
        sys.exit(f"ERROR: {local_dir} not found.")

    total_mb = sum(p.stat().st_size for p in local_dir.iterdir() if p.is_file()) / 1024 / 1024
    print(f"  Uploading folder ({total_mb:.1f} MB total) ...")

    upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload fine-tuned RoBERTa 5-class classifier",
    )

    api = HfApi(token=token)
    readme = (
        "---\nlicense: apache-2.0\nlanguage:\n- en\npipeline_tag: text-classification\n"
        "tags:\n- yelp\n- sentiment-analysis\n- roberta\n---\n\n"
        "# Yelp 5-class Star Rating Classifier (RoBERTa-base)\n\n"
        "Fine-tuned `roberta-base` for predicting Yelp star ratings (1–5) from "
        "review text. Trained on 50k sampled Yelp reviews.\n\n"
        "**Validation accuracy:** 0.6856 · **Macro-F1:** 0.6849\n\n"
        "Used by the Yelp Business Intelligence Agent for the optional "
        "`classify_review` tool.\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  README uploaded.")
    print(f"  Done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Your HF username")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    token = read_token()
    print(f"Token loaded from .hf_token ({token[:6]}...{token[-4:]})")

    if not args.skip_dataset:
        upload_dataset(args.user, token)
    if not args.skip_model:
        upload_model(args.user, token)

    print("\nAll uploads complete.")
    print(f"\nNext step: create the Space at")
    print(f"  https://huggingface.co/spaces/{args.user}/yelp-rag-agent")


if __name__ == "__main__":
    main()
