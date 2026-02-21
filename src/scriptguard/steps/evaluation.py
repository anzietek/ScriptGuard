import json
import os
from typing import Any, Dict, List
from zenml import step, ArtifactConfig
from typing import Annotated
import numpy as np
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

_THRESHOLDS = [round(t, 2) for t in np.arange(0.40, 0.71, 0.05)]


def _metrics_at_threshold(
    true_labels: list[int],
    max_probs: list[float],
    threshold: float,
) -> dict[str, Any]:
    preds = [1 if p >= threshold else 0 for p in max_probs]
    tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="binary", pos_label=1, zero_division=0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(true_labels, preds))
    accuracy = float(accuracy_score(true_labels, preds))
    return {
        "threshold": threshold,
        "preds": preds,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "mcc": mcc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


@step
def evaluate_codebert(
    test_dataset: Dataset,
    test_data: List[Dict[str, Any]],
    model_path: str,
    config: Dict[str, Any],
    scaler_path: str,
) -> Annotated[Dict[str, Any], ArtifactConfig(name="metrics")]:
    codebert_cfg = config.get("codebert", {})
    batch_size: int = codebert_cfg.get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model for evaluation from {model_path}")

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

    available = [c for c in eval_columns if c in test_dataset.column_names]
    eval_ds = test_dataset.select_columns(available)

    chunk_script_ids: list[int] = test_dataset["script_id"]
    chunk_true_labels: list[int] = test_dataset["label"]

    # ------------------------------------------------------------------
    # 1. Inference — collect malicious_prob per chunk (single forward pass)
    # ------------------------------------------------------------------
    all_malicious_probs: list[float] = []
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            all_malicious_probs.extend(probs[:, 1].cpu().tolist())

    # ------------------------------------------------------------------
    # 2. Aggregate chunks → scripts (max malicious_prob, threshold-free)
    # ------------------------------------------------------------------
    script_max_prob: dict[int, float] = {}
    script_true: dict[int, int] = {}

    for i, mal_prob in enumerate(all_malicious_probs):
        sid = chunk_script_ids[i]
        if sid not in script_max_prob or mal_prob > script_max_prob[sid]:
            script_max_prob[sid] = mal_prob
        script_true[sid] = chunk_true_labels[i]

    sids: list[int] = list(script_max_prob.keys())
    true_labels_script: list[int] = [script_true[s] for s in sids]
    max_probs_script: list[float] = [script_max_prob[s] for s in sids]

    # ROC-AUC is threshold-independent — compute once
    try:
        roc_auc = float(roc_auc_score(true_labels_script, max_probs_script))
    except ValueError:
        roc_auc = float("nan")
        logger.warning("ROC-AUC skipped: only one class present in test set")

    # ------------------------------------------------------------------
    # 3. Sweep thresholds 0.40 → 0.70, step 0.05
    # ------------------------------------------------------------------
    results: list[dict] = []
    for thr in _THRESHOLDS:
        results.append(_metrics_at_threshold(true_labels_script, max_probs_script, thr))

    # Log threshold table
    logger.info("=" * 80)
    logger.info("THRESHOLD SWEEP (script-level, max malicious_prob aggregation)")
    logger.info(f"{'Thr':>5}  {'Acc':>6}  {'Prec':>6}  {'Recall':>6}  {'Spec':>6}  "
                f"{'FPR':>6}  {'F1':>6}  {'MCC':>7}  {'TP':>4}  {'FP':>4}  {'TN':>4}  {'FN':>4}")
    logger.info("-" * 80)
    for r in results:
        logger.info(
            f"{r['threshold']:>5.2f}  {r['accuracy']:>6.4f}  {r['precision']:>6.4f}  "
            f"{r['recall']:>6.4f}  {r['specificity']:>6.4f}  {r['fpr']:>6.4f}  "
            f"{r['f1']:>6.4f}  {r['mcc']:>7.4f}  "
            f"{r['tp']:>4}  {r['fp']:>4}  {r['tn']:>4}  {r['fn']:>4}"
        )
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 4. Best threshold by MCC (most robust for imbalanced binary classification)
    # ------------------------------------------------------------------
    best = max(results, key=lambda r: r["mcc"])
    best_thr = best["threshold"]
    best_preds = best["preds"]

    # FP / FN IDs at best threshold
    fp_db_ids: list[Any] = []
    fn_db_ids: list[Any] = []
    for sid, pred, true_lbl in zip(sids, best_preds, true_labels_script):
        db_id = test_data[sid].get("id") if sid < len(test_data) else None
        if pred == 1 and true_lbl == 0:
            fp_db_ids.append(db_id)
        elif pred == 0 and true_lbl == 1:
            fn_db_ids.append(db_id)

    logger.info(f"False Positives @ {best_thr:.2f} ({len(fp_db_ids)} benign → predicted malicious): {fp_db_ids}")
    logger.info(f"False Negatives @ {best_thr:.2f} ({len(fn_db_ids)} malicious → predicted benign): {fn_db_ids}")

    # ------------------------------------------------------------------
    # 5. Best metrics summary
    # ------------------------------------------------------------------
    cm = [[best["tn"], best["fp"]], [best["fn"], best["tp"]]]

    logger.info("=" * 55)
    logger.info(f"BEST THRESHOLD: {best_thr:.2f}  (optimised by MCC)")
    logger.info(f"  Accuracy:          {best['accuracy']:.4f}")
    logger.info(f"  Precision:         {best['precision']:.4f}")
    logger.info(f"  Malicious Recall:  {best['recall']:.4f}  (sensitivity)")
    logger.info(f"  Specificity:       {best['specificity']:.4f} (benign recall)")
    logger.info(f"  FPR:               {best['fpr']:.4f}  ← kluczowe dla security")
    logger.info(f"  F1:                {best['f1']:.4f}")
    logger.info(f"  MCC:               {best['mcc']:.4f}")
    logger.info(f"  ROC-AUC:           {roc_auc:.4f}")
    logger.info(f"  Confusion Matrix:  TN={best['tn']} FP={best['fp']} FN={best['fn']} TP={best['tp']}")
    logger.info(f"  Scripts evaluated: {len(true_labels_script)}")
    logger.info(f"  Chunks evaluated:  {len(all_malicious_probs)}")
    logger.info("=" * 55)

    metrics: Dict[str, Any] = {
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "malicious_recall": best["recall"],
        "specificity": best["specificity"],
        "f1": best["f1"],
        "fpr": best["fpr"],
        "mcc": best["mcc"],
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tp": best["tp"], "fp": best["fp"], "tn": best["tn"], "fn": best["fn"],
        "best_decision_threshold": best_thr,
        "num_scripts_evaluated": len(true_labels_script),
        "num_chunks_evaluated": len(all_malicious_probs),
        "threshold_sweep": [
            {k: v for k, v in r.items() if k != "preds"}
            for r in results
        ],
    }

    return metrics
