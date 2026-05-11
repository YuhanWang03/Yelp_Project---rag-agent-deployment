"""
Stage 4 - Step 0: Train and Save the Best RoBERTa 5-class Model

This script replicates the best Stage 2 experiment (roberta-base, 5_class,
run_006: Accuracy=0.6856, Macro_F1=0.6849) and saves the trained model
to s4_agent/artifacts/roberta_5class_best/ for use as a callable tool in
the Stage 4 agent pipeline.

The existing s2_bert_scripts are reused for data loading and tokenization.
The only addition is the model.save_pretrained() call at the end.

Usage:
    python s4_agent/step0_train_and_save.py
"""

import sys
import os
import gc
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ---------------------------------------------------------------------------
# Make s2_bert_scripts importable from this script's location
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "s2_bert_scripts"))

from data_loader import load_and_preprocess_data, tokenize_data
from utils import get_compute_metrics_fn

# ---------------------------------------------------------------------------
# Best configuration (roberta-base, 5_class, run_006)
# ---------------------------------------------------------------------------
CONFIG = {
    "task_type": "5_class",
    "use_regression": False,
    "model_name": "roberta-base",
    "max_length": 512,
    "learning_rate": 1e-5,
    "num_epochs": 4,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "seed": 42,
}

# Output directory: where the final model will be saved for Stage 4 use
SAVE_DIR = PROJECT_ROOT / "s4_agent" / "artifacts" / "roberta_5class_best"

# Temporary checkpoint directory (cleaned up after saving)
CKPT_DIR = PROJECT_ROOT / "s4_agent" / "artifacts" / "_roberta_tmp_ckpt"


def main() -> None:
    set_seed(CONFIG["seed"])

    print("=" * 60)
    print("Stage 4 - Step 0: Train and Save RoBERTa 5-class")
    print(f"Model   : {CONFIG['model_name']}")
    print(f"Config  : LR={CONFIG['learning_rate']}, Epochs={CONFIG['num_epochs']}, "
          f"BatchSize={CONFIG['batch_size']}, MaxLen={CONFIG['max_length']}")
    print(f"Save to : {SAVE_DIR}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
    val_path   = PROJECT_ROOT / "data" / "processed" / "val_data.csv"

    train_dataset, val_dataset, num_labels = load_and_preprocess_data(
        train_path=train_path,
        val_path=val_path,
        task_type=CONFIG["task_type"],
        use_regression=CONFIG["use_regression"],
    )
    print(f"Data loaded — train: {len(train_dataset)}, val: {len(val_dataset)}, "
          f"num_labels: {num_labels}")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenized_train, tokenized_val = tokenize_data(
        train_dataset, val_dataset, tokenizer, CONFIG["max_length"]
    )

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print(f"Loading {CONFIG['model_name']} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=num_labels,
    )

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        num_train_epochs=CONFIG["num_epochs"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        fp16=CONFIG["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    compute_metrics = get_compute_metrics_fn(CONFIG)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("Starting training ...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------
    print("Running final evaluation on validation set ...")
    eval_results = trainer.evaluate()
    print(f"Final val accuracy : {eval_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Final val macro_f1 : {eval_results.get('eval_macro_f1', 'N/A'):.4f}")

    # ------------------------------------------------------------------
    # 7. Save model + tokenizer to the Stage 4 artifacts directory
    #    (This is the step that was commented out in train.py)
    # ------------------------------------------------------------------
    print(f"\nSaving model and tokenizer to:\n  {SAVE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))

    # Save a label mapping so the classifier tool can decode predictions
    label_map = {
        "id2label": {str(i): f"{i+1}_star" for i in range(num_labels)},
        "label2id": {f"{i+1}_star": i for i in range(num_labels)},
        "num_labels": num_labels,
        "task_type": CONFIG["task_type"],
        "model_name": CONFIG["model_name"],
        "val_accuracy": eval_results.get("eval_accuracy", None),
        "val_macro_f1": eval_results.get("eval_macro_f1", None),
    }
    with open(SAVE_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("Model saved successfully.")
    print(f"Files in {SAVE_DIR}:")
    for p in sorted(SAVE_DIR.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:40s}  {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # 8. Clean up temporary checkpoint directory
    # ------------------------------------------------------------------
    import shutil
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
        print(f"\nTemporary checkpoint dir removed: {CKPT_DIR.name}")

    # ------------------------------------------------------------------
    # 9. Free memory
    # ------------------------------------------------------------------
    del model, trainer, tokenized_train, tokenized_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nStep 0 complete. Model is ready at:")
    print(f"  {SAVE_DIR}")
    print("\nNext: run s4_agent/test_classifier_load.py to verify the saved model.")


if __name__ == "__main__":
    main()
