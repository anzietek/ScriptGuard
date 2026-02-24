"""
Fused CodeBERT Classifier for ScriptGuard — Gated Fusion v2.0.

Architecture:
    feature_vector [B×F] → log1p → Linear(F→128) → BN → GELU → Dropout(0.2)
                                  → Linear(128→256) → BN → GELU → Dropout(0.2)
                                  → v_proj [B×256]
                                                           ↘
    input_ids → GraphCodeBERT → [CLS] [B×768]              cat([e,v]) [B×1024]
                                              ↘            → Linear(1024→256)
                                               alpha [B×256] = sigmoid(...)
                                               z = alpha ⊙ v_proj + (1-alpha) ⊙ e[:,:256]
                                               → Linear(256→128) → GELU → Dropout(0.3) → Linear(128→2)

This module also provides FusedDataCollator, FusedWeightedTrainer,
save_fused_model, and load_fused_model.
"""

import json
import os
import shutil
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from scriptguard.models.codebert_classifier import WeightedTrainer
from scriptguard.utils.logger import logger


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusedCodeBERTClassifier(nn.Module):
    """
    GraphCodeBERT [CLS] embedding fused with a hand-crafted feature branch.

    Args:
        model_name:         HuggingFace model identifier (e.g. "microsoft/graphcodebert-base")
        num_labels:         Number of output classes (2 for binary classification)
        feature_dim:        Dimensionality of the hand-crafted feature vector (28)
        mlp_hidden_dim:     Hidden units in the feature MLP branch (128)
        fusion_hidden_dim:  Hidden units in the fusion head (256)
        dropout_rate:       Dropout probability applied in the feature projection branch (0.2)
        cls_head_dropout:   Dropout probability applied in the classification head (0.3)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        feature_dim: int = 28,
        mlp_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        cls_head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout_rate = dropout_rate
        self.cls_head_dropout = cls_head_dropout

        # BERT backbone — raw encoder, not classification head
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden: int = self.bert.config.hidden_size  # 768 for base models

        # Feature projection: feature_dim → mlp_hidden_dim → fusion_hidden_dim
        # dropout_rate (0.2) applied after each BN+GELU block
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Gated fusion: Linear(bert_hidden + fusion_hidden_dim → fusion_hidden_dim)
        # concat([e_trans, v_proj]) ∈ R^(768+256=1024) → alpha ∈ R^256
        self.gate = nn.Linear(bert_hidden + fusion_hidden_dim, fusion_hidden_dim)

        # Classification head: fusion_hidden_dim → mlp_hidden_dim → num_labels
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_head_dropout),
            nn.Linear(mlp_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_vector: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        """
        Args:
            input_ids:      [B, seq_len] Long tensor
            attention_mask: [B, seq_len] Long tensor
            feature_vector: [B, feature_dim] Float tensor
            labels:         [B] Long tensor (unused in forward; consumed by WeightedTrainer)

        Returns:
            SimpleNamespace with attribute `logits` of shape [B, num_labels].
        """
        # BERT path → [CLS] representation
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        e_trans: torch.Tensor = bert_out.last_hidden_state[:, 0, :]  # [B, 768]

        # Feature projection branch
        # Cast to BERT dtype for fp16/bf16 autocast consistency, then apply log1p
        v_feat: torch.Tensor = feature_vector.to(dtype=e_trans.dtype)
        # Signed log1p: sign(x)·log1p(|x|) — numerically stable on Z-scored inputs
        # (plain log1p(x) returns NaN for x < -1; signed form handles all reals)
        v_feat_scaled: torch.Tensor = torch.sign(v_feat) * torch.log1p(v_feat.abs())
        v_proj: torch.Tensor = self.feat_proj(v_feat_scaled)        # [B, 256]

        # Gated fusion
        concat: torch.Tensor = torch.cat([e_trans, v_proj], dim=-1) # [B, 1024]
        alpha: torch.Tensor = torch.sigmoid(self.gate(concat))       # [B, 256]
        z: torch.Tensor = (
            alpha * v_proj + (1.0 - alpha) * e_trans[:, : self.fusion_hidden_dim]
        )                                                             # [B, 256]

        # Classification head
        logits: torch.Tensor = self.cls_head(z)                      # [B, 2]

        return SimpleNamespace(logits=logits, alpha=alpha)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class FusedDataCollator:
    """
    Data collator that handles the extra `feature_vector`, `script_id`, and
    `chunk_index` columns added by the extract_features step.

    It pops those columns before delegating to DataCollatorWithPadding so that
    the base collator only sees HuggingFace-compatible token fields.
    `feature_vector` is re-attached as a Float32 tensor.
    """

    tokenizer: Any  # AutoTokenizer

    def __post_init__(self) -> None:
        # Cache once — avoids re-constructing on every batch call
        self._base_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __call__(self, features: list[dict]) -> dict:
        feature_vecs = [f.pop("feature_vector") for f in features]
        for f in features:
            f.pop("script_id", None)
            f.pop("chunk_index", None)

        batch = self._base_collator(features)
        batch["feature_vector"] = torch.tensor(feature_vecs, dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FusedWeightedTrainer(WeightedTrainer):
    """
    Extends WeightedTrainer (Focal Loss + class weights) to support the custom
    FusedCodeBERTClassifier which is not a HuggingFace PreTrainedModel.

    Key overrides:
      - evaluate()        → script-level aggregation (max malicious_prob), consistent with inference
      - save_model()      → st_save_file state dict to checkpoint/model.safetensors
      - _load_best_model() → st_load_file state dict from checkpoint/model.safetensors
    """

    def __init__(self, *args: Any, decision_threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_threshold = decision_threshold

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Script-level evaluation — aggregates chunk predictions by max malicious_prob,
        identical to ScriptGuardClassifier.classify(). This ensures that the metric
        used for early stopping (eval_recall) and checkpoint selection matches the
        true script-level performance, not chunk-level performance.
        """
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Read script_ids and labels from the raw dataset BEFORE the collator
        # strips them. DataLoader preserves insertion order for eval (no shuffle).
        script_ids: list[int] = eval_ds["script_id"]
        true_labels_flat: list[int] = eval_ds["label"]

        dataloader = self.get_eval_dataloader(eval_ds)

        all_malicious_probs: list[float] = []
        alpha_means: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                batch.pop("labels", None)  # not needed for forward pass
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_malicious_probs.extend(probs[:, 1].cpu().tolist())
                if hasattr(outputs, "alpha"):
                    alpha_means.append(outputs.alpha.mean().item())

        # Aggregate chunks → scripts using max malicious_prob (same as inference)
        chunk_mal_probs: dict[int, list[float]] = {}
        chunk_true: dict[int, int] = {}
        for i, mal_prob in enumerate(all_malicious_probs):
            sid = script_ids[i]
            chunk_mal_probs.setdefault(sid, []).append(mal_prob)
            chunk_true[sid] = true_labels_flat[i]

        final_preds: list[int] = []
        final_true: list[int] = []
        for sid, probs_list in chunk_mal_probs.items():
            best_prob = max(probs_list)
            final_preds.append(1 if best_prob >= self.decision_threshold else 0)
            final_true.append(chunk_true[sid])

        labels_arr = np.array(final_true)
        preds_arr = np.array(final_preds)

        tn = int(np.sum((preds_arr == 0) & (labels_arr == 0)))
        fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))
        tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = float(matthews_corrcoef(final_true, final_preds))

        metrics: dict[str, float] = {
            f"{metric_key_prefix}_accuracy":    float(accuracy_score(final_true, final_preds)),
            f"{metric_key_prefix}_f1":          float(f1_score(final_true, final_preds, average="binary", zero_division=0)),
            f"{metric_key_prefix}_precision":   float(precision_score(final_true, final_preds, average="binary", zero_division=0)),
            f"{metric_key_prefix}_recall":      float(recall_score(final_true, final_preds, average="binary", zero_division=0)),
            f"{metric_key_prefix}_fpr":         fpr,
            f"{metric_key_prefix}_specificity": specificity,
            f"{metric_key_prefix}_mcc":         mcc,
        }

        alpha_str = f"  gate_alpha={sum(alpha_means)/len(alpha_means):.4f}" if alpha_means else ""
        logger.info(
            f"[script-level val] recall={metrics[f'{metric_key_prefix}_recall']:.4f}  "
            f"fpr={fpr:.4f}  f1={metrics[f'{metric_key_prefix}_f1']:.4f}  "
            f"mcc={mcc:.4f}  scripts={len(final_true)}{alpha_str}"
        )

        # Log metrics so they appear in training state/history.
        # Do NOT call on_evaluate here — the Trainer's _inner_training_loop calls it
        # after evaluate() returns.  Calling it here too would double-fire
        # EarlyStoppingCallback and halve the effective patience.
        self.log(metrics)

        return metrics

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        out = output_dir or self.args.output_dir
        os.makedirs(out, exist_ok=True)
        st_save_file(self.model.state_dict(), os.path.join(out, "model.safetensors"))
        logger.info(f"FusedWeightedTrainer: saved model.safetensors to {out}")

    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint:
            state = st_load_file(
                os.path.join(self.state.best_model_checkpoint, "model.safetensors"),
                device=str(self.args.device),
            )
            self.model.load_state_dict(state)
            self.model.to(self.args.device)
            logger.info(f"FusedWeightedTrainer: loaded best model from {self.state.best_model_checkpoint}")


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_fused_model(
    model: FusedCodeBERTClassifier,
    tokenizer: Any,
    output_dir: str,
    inference_cfg: dict,
    scaler_path: str,
) -> None:
    """
    Persist a trained FusedCodeBERTClassifier to `output_dir`.

    Saved files:
        tokenizer files          — for AutoTokenizer.from_pretrained compat
        model.safetensors        — state dict in safetensors format
        fused_model_config.json  — architecture params
        inference_config.json    — runtime config (includes "model_type": "fused")
        feature_scaler.joblib    — copied from `scaler_path`
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)

    st_save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))

    model_cfg = {
        "model_name": model.model_name,
        "num_labels": model.num_labels,
        "feature_dim": model.feature_dim,
        "mlp_hidden_dim": model.mlp_hidden_dim,
        "fusion_hidden_dim": model.fusion_hidden_dim,
        "dropout_rate": model.dropout_rate,
        "cls_head_dropout": model.cls_head_dropout,
    }
    with open(os.path.join(output_dir, "fused_model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    inference_cfg["model_type"] = "fused"
    with open(os.path.join(output_dir, "inference_config.json"), "w") as f:
        json.dump(inference_cfg, f, indent=2)

    dest_scaler = os.path.join(output_dir, "feature_scaler.joblib")
    if os.path.abspath(scaler_path) != os.path.abspath(dest_scaler):
        shutil.copy(scaler_path, dest_scaler)

    logger.info(f"Saved fused model to {output_dir}")


def load_fused_model(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> tuple[FusedCodeBERTClassifier, Any]:
    """
    Load a FusedCodeBERTClassifier from `model_dir`.

    Returns:
        (model, tokenizer) — model is in eval mode on `device`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, "fused_model_config.json")) as f:
        cfg = json.load(f)

    model = FusedCodeBERTClassifier(
        model_name=cfg["model_name"],
        num_labels=cfg.get("num_labels", 2),
        feature_dim=cfg.get("feature_dim", 28),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 128),
        fusion_hidden_dim=cfg.get("fusion_hidden_dim", 256),
        dropout_rate=cfg.get("dropout_rate", 0.2),
        cls_head_dropout=cfg.get("cls_head_dropout", 0.3),
    )

    state = st_load_file(
        os.path.join(model_dir, "model.safetensors"),
        device=str(device),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    logger.info(f"Loaded fused model from {model_dir} on {device}")
    return model, tokenizer
