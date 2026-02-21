import json
import os
from typing import Any, Dict
from zenml import step, ArtifactConfig
from typing import Annotated
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from scriptguard.utils.logger import logger


def aggregate_chunk_predictions(chunks: list[dict]) -> tuple[int, float]:
    best_label = chunks[0]["predicted_label"]
    best_confidence = chunks[0]["confidence"]
    for chunk in chunks[1:]:
        if chunk["confidence"] > best_confidence:
            best_confidence = chunk["confidence"]
            best_label = chunk["predicted_label"]
    return best_label, best_confidence


@step
def evaluate_codebert(
    test_dataset: Dataset,
    model_path: str,
    config: Dict[str, Any],
    scaler_path: str,
) -> Annotated[Dict[str, Any], ArtifactConfig(name="metrics")]:
    codebert_cfg = config.get("codebert", {})
    batch_size: int = codebert_cfg.get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model for evaluation from {model_path}")

    # Detect model type (fused vs legacy)
    inf_cfg_path = os.path.join(model_path, "inference_config.json")
    inf_cfg = {}
    if os.path.exists(inf_cfg_path):
        with open(inf_cfg_path) as f:
            inf_cfg = json.load(f)

    is_fused = inf_cfg.get("model_type") == "fused"

    if is_fused:
        from scriptguard.models.fused_classifier import load_fused_model, FusedDataCollator
        model, tokenizer = load_fused_model(model_path, device)
        collator = FusedDataCollator(tokenizer=tokenizer)
        eval_columns = ["input_ids", "attention_mask", "feature_vector"]
        logger.info("Evaluation: using fused model path")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        eval_columns = ["input_ids", "attention_mask"]
        logger.info("Evaluation: using legacy model path")

    # Select only the columns the collator expects
    available = [c for c in eval_columns if c in test_dataset.column_names]
    eval_ds = test_dataset.select_columns(available)

    script_ids: list[int] = test_dataset["script_id"]
    true_labels: list[int] = test_dataset["label"]

    chunk_results: dict[int, list[dict]] = {}
    chunk_true: dict[int, int] = {}

    all_preds: list[int] = []
    all_confs: list[float] = []

    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confidences.cpu().tolist())

    for i, (pred, conf) in enumerate(zip(all_preds, all_confs)):
        sid = script_ids[i]
        true_lbl = true_labels[i]
        if sid not in chunk_results:
            chunk_results[sid] = []
            chunk_true[sid] = true_lbl
        chunk_results[sid].append({"predicted_label": pred, "confidence": conf})

    final_preds: list[int] = []
    final_true: list[int] = []

    for sid, chunks in chunk_results.items():
        pred_label, _ = aggregate_chunk_predictions(chunks)
        final_preds.append(pred_label)
        final_true.append(chunk_true[sid])

    accuracy = accuracy_score(final_true, final_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        final_true, final_preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(final_true, final_preds, labels=[0, 1]).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "malicious_recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "num_scripts_evaluated": len(final_true),
        "num_chunks_evaluated": len(all_preds),
    }

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info(f"  Accuracy:          {accuracy:.4f}")
    logger.info(f"  Precision:         {precision:.4f}")
    logger.info(f"  Malicious Recall:  {recall:.4f}")
    logger.info(f"  F1:                {f1:.4f}")
    logger.info(f"  Confusion Matrix:  {cm}")
    logger.info(f"  Scripts evaluated: {len(final_true)}")
    logger.info("=" * 50)

    return metrics
