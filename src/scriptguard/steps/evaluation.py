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
    roc_auc_score,
    matthews_corrcoef,
)
from scriptguard.utils.logger import logger


def aggregate_chunk_predictions(
    chunks: list[dict],
    decision_threshold: float = 0.5,
) -> tuple[int, float]:
    """
    Aggregate chunk-level predictions to a single script-level prediction.

    Strategy: use the maximum malicious_prob across all chunks (consistent with
    ScriptGuardClassifier.classify which also uses best_malicious_prob).

    Returns:
        (predicted_label: int, best_malicious_prob: float)
    """
    best_malicious_prob = max(c["malicious_prob"] for c in chunks)
    label = 1 if best_malicious_prob >= decision_threshold else 0
    return label, best_malicious_prob


@step
def evaluate_codebert(
    test_dataset: Dataset,
    model_path: str,
    config: Dict[str, Any],
    scaler_path: str,
) -> Annotated[Dict[str, Any], ArtifactConfig(name="metrics")]:
    codebert_cfg = config.get("codebert", {})
    batch_size: int = codebert_cfg.get("batch_size", 16)
    decision_threshold: float = codebert_cfg.get("decision_threshold", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model for evaluation from {model_path}")

    inf_cfg_path = os.path.join(model_path, "inference_config.json")
    inf_cfg = {}
    if os.path.exists(inf_cfg_path):
        with open(inf_cfg_path) as f:
            inf_cfg = json.load(f)
        decision_threshold = inf_cfg.get("decision_threshold", decision_threshold)

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

    available = [c for c in eval_columns if c in test_dataset.column_names]
    eval_ds = test_dataset.select_columns(available)

    script_ids: list[int] = test_dataset["script_id"]
    true_labels: list[int] = test_dataset["label"]

    # Collect malicious_prob per chunk (consistent with inference)
    all_malicious_probs: list[float] = []

    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            malicious_probs = probs[:, 1]  # class 1 = malicious
            all_malicious_probs.extend(malicious_probs.cpu().tolist())

    # Group chunks by script_id
    chunk_results: dict[int, list[dict]] = {}
    chunk_true: dict[int, int] = {}

    for i, malicious_prob in enumerate(all_malicious_probs):
        sid = script_ids[i]
        true_lbl = true_labels[i]
        if sid not in chunk_results:
            chunk_results[sid] = []
            chunk_true[sid] = true_lbl
        chunk_results[sid].append({"malicious_prob": malicious_prob})

    # Aggregate to script level
    final_preds: list[int] = []
    final_true: list[int] = []
    final_scores: list[float] = []  # malicious_prob for ROC-AUC

    for sid, chunks in chunk_results.items():
        pred_label, best_prob = aggregate_chunk_predictions(chunks, decision_threshold)
        final_preds.append(pred_label)
        final_true.append(chunk_true[sid])
        final_scores.append(best_prob)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(final_true, final_preds, labels=[0, 1]).ravel()
    cm = [[int(tn), int(fp)], [int(fn), int(tp)]]

    accuracy = accuracy_score(final_true, final_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        final_true, final_preds, average="binary", pos_label=1, zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(final_true, final_preds)

    # ROC-AUC (requires both classes present)
    try:
        roc_auc = roc_auc_score(final_true, final_scores)
    except ValueError:
        roc_auc = float("nan")
        logger.warning("ROC-AUC skipped: only one class present in test set")

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "malicious_recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "fpr": float(fpr),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "num_scripts_evaluated": len(final_true),
        "num_chunks_evaluated": len(all_malicious_probs),
        "decision_threshold": decision_threshold,
    }

    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS (script-level)")
    logger.info(f"  Threshold:         {decision_threshold:.2f}")
    logger.info(f"  Accuracy:          {accuracy:.4f}")
    logger.info(f"  Precision:         {precision:.4f}")
    logger.info(f"  Malicious Recall:  {recall:.4f}  (sensitivity)")
    logger.info(f"  Specificity:       {specificity:.4f} (benign recall)")
    logger.info(f"  FPR:               {fpr:.4f}  ← kluczowe dla security")
    logger.info(f"  F1:                {f1:.4f}")
    logger.info(f"  MCC:               {mcc:.4f}")
    logger.info(f"  ROC-AUC:           {roc_auc:.4f}")
    logger.info(f"  Confusion Matrix:  TN={tn} FP={fp} FN={fn} TP={tp}")
    logger.info(f"  Scripts evaluated: {len(final_true)}")
    logger.info("=" * 55)

    return metrics
