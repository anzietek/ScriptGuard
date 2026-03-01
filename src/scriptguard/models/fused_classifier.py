"""
Fused CodeBERT Classifier for ScriptGuard.

Supports two fusion architectures, selected by config.fusion_architecture:

  "concat":
    feature_vector [B×F] → MLP(F→H_mlp) [GELU+Dropout]
    fused = cat([CLS, feat_proj]) [B × (768+H_mlp)]
    logits = fusion_head(fused)

  "gated":
    feature_vector [B×F] → signed-log1p → projection(F→H2) [BN+GELU blocks]
    cls_proj = bert_projection(CLS) [B×H2]  (Linear+GELU+Dropout, no hard slicing)
    alpha = sigmoid(gate(cat([cls_proj, feat_proj])))   [B×H2]
    z = alpha * feat_proj + (1-alpha) * cls_proj
    logits = fusion_head(z)
    (also returns alpha for gate monitoring)

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
    GraphCodeBERT [CLS] fused with hand-crafted features.

    Fusion architecture is controlled by config.fusion_architecture:
      "concat": features → MLP → cat([CLS, feat]) → fusion_head → logits
      "gated":  features → projection → gated_combine(cls_proj, feat_proj) → fusion_head → logits
                also returns alpha for gate monitoring

    Args:
        model_name:  HuggingFace model identifier (e.g. "microsoft/graphcodebert-base")
        num_labels:  Number of output classes (default 2)
        config:      SimpleNamespace with architecture parameters:
                       both arch:  feature_dim
                       concat:     concat_mlp_hidden, concat_fusion_hidden, concat_dropout
                       gated:      gated_proj_hidden_1, gated_proj_hidden_2, gated_proj_dropout,
                                   gated_fusion_hidden, gated_fusion_dropout
                     Defaults to concat with feature_dim=27 if None.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        config: Optional[SimpleNamespace] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        if config is None:
            logger.warning(
                "FusedCodeBERTClassifier: no config provided; defaulting to concat "
                "architecture with feature_dim=27, concat_mlp_hidden=128, "
                "concat_fusion_hidden=256, concat_dropout=0.3"
            )
            config = SimpleNamespace(
                fusion_architecture="concat",
                feature_dim=27,
                concat_mlp_hidden=128,
                concat_fusion_hidden=256,
                concat_dropout=0.3,
            )
        self.config = config

        # BERT backbone — raw encoder, no classification head
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden: int = self.bert.config.hidden_size  # 768 for base models

        arch = config.fusion_architecture

        if arch == "concat":
            # feature → MLP → concat([CLS, feat]) → fusion_head
            self.mlp = nn.Sequential(
                nn.Linear(config.feature_dim, config.concat_mlp_hidden),
                nn.GELU(),
                nn.Dropout(config.concat_dropout),
            )
            concat_dim = bert_hidden + config.concat_mlp_hidden
            self.fusion_head = nn.Sequential(
                nn.Linear(concat_dim, config.concat_fusion_hidden),
                nn.GELU(),
                nn.Dropout(config.concat_dropout),
                nn.Linear(config.concat_fusion_hidden, num_labels),
            )

        elif arch == "gated":
            # feature → signed-log1p → projection [BN blocks] → gated combine
            self.projection = nn.Sequential(
                nn.Linear(config.feature_dim, config.gated_proj_hidden_1),
                nn.BatchNorm1d(config.gated_proj_hidden_1),
                nn.GELU(),
                nn.Dropout(config.gated_proj_dropout),
                nn.Linear(config.gated_proj_hidden_1, config.gated_proj_hidden_2),
                nn.BatchNorm1d(config.gated_proj_hidden_2),
                nn.GELU(),
                nn.Dropout(config.gated_proj_dropout),
            )
            # Project BERT CLS (768) → H2 so it is compatible with feat_proj for gating.
            # A learned linear projection preserves all semantic information; the old
            # approach of hard-slicing CLS[:, :H2] discarded 512 of 768 dimensions.
            self.bert_projection = nn.Sequential(
                nn.Linear(bert_hidden, config.gated_proj_hidden_2),
                nn.GELU(),
                nn.Dropout(config.gated_proj_dropout),
            )
            # Gate input is cat([cls_proj, feat_proj]) — both are H2-dimensional.
            gate_input_dim = config.gated_proj_hidden_2 * 2
            self.gate = nn.Linear(gate_input_dim, config.gated_proj_hidden_2)
            self.fusion_head = nn.Sequential(
                nn.Linear(config.gated_proj_hidden_2, config.gated_fusion_hidden),
                nn.GELU(),
                nn.Dropout(config.gated_fusion_dropout),
                nn.Linear(config.gated_fusion_hidden, num_labels),
            )

        else:
            raise ValueError(
                f"FusedCodeBERTClassifier: unknown fusion_architecture={arch!r}. "
                "Expected 'concat' or 'gated'."
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
            feature_vector: [B, feature_dim] Float tensor (StandardScaler-normalised)
            labels:         [B] Long tensor — not used here; consumed by WeightedTrainer

        Returns:
            SimpleNamespace with `logits` [B, num_labels].
            Gated architecture additionally returns `alpha` [B, gated_proj_hidden_2].
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls: torch.Tensor = bert_out.last_hidden_state[:, 0, :]  # [B, 768]

        # Cast feature_vector to BERT dtype for fp16/bf16 autocast consistency
        features: torch.Tensor = feature_vector.to(dtype=cls.dtype)

        arch = self.config.fusion_architecture

        if arch == "concat":
            feat_proj: torch.Tensor = self.mlp(features)              # [B, concat_mlp_hidden]
            fused: torch.Tensor = torch.cat([cls, feat_proj], dim=-1)  # [B, 768+H_mlp]
            expected_dim = cls.shape[-1] + self.config.concat_mlp_hidden
            if fused.shape[-1] != expected_dim:
                logger.warning(
                    f"Concat dim mismatch: got {fused.shape[-1]}, expected {expected_dim}"
                )
            logits: torch.Tensor = self.fusion_head(fused)             # [B, num_labels]
            return SimpleNamespace(logits=logits)

        else:  # gated
            # Signed log1p: sign(x)·log1p(|x|) — handles Z-scored values < -1 without NaN
            feat_scaled: torch.Tensor = torch.sign(features) * torch.log1p(features.abs())
            feat_proj: torch.Tensor = self.projection(feat_scaled)      # [B, H2]
            cls_proj: torch.Tensor = self.bert_projection(cls)          # [B, H2]
            concat: torch.Tensor = torch.cat([cls_proj, feat_proj], dim=-1)  # [B, H2*2]
            alpha: torch.Tensor = torch.sigmoid(self.gate(concat))      # [B, H2]
            z: torch.Tensor = alpha * feat_proj + (1.0 - alpha) * cls_proj  # [B, H2]
            logits = self.fusion_head(z)                                # [B, num_labels]
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
      - create_optimizer() → differential LRs: lr_backbone for BERT, lr_fusion_head for the rest
      - compute_loss()     → gate alpha logging per 50 steps (malicious vs benign split)
      - evaluate()         → script-level aggregation + optional zero-API ghost cluster recall
      - save_model()       → st_save_file state dict to checkpoint/model.safetensors
      - _load_best_model() → st_load_file state dict from checkpoint/model.safetensors
    """

    def __init__(
        self,
        *args: Any,
        decision_threshold: float = 0.5,
        lr_backbone: float = 1e-5,
        lr_fusion_head: float = 1e-4,
        zero_api_zscore: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decision_threshold = decision_threshold
        self._lr_backbone = lr_backbone
        self._lr_fusion_head = lr_fusion_head
        # Z-scored threshold for the zero-API ghost cluster (malware_api_score == 0).
        # Computed externally as (0 - scaler.mean_[21]) / scaler.scale_[21].
        self._zero_api_zscore = zero_api_zscore

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Differential learning rates:
          - BERT backbone parameters: lr_backbone (default 1e-5)
          - Projection + gate + classification head: lr_fusion_head (default 1e-4)

        All parameters — including frozen backbone params — are registered from the
        start so that unfreezing mid-training (BackboneUnfreezeCallback) immediately
        picks up the correct LR without optimizer reconstruction.
        """
        model = self.model
        backbone_params = list(model.bert.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        fusion_params = [p for p in model.parameters() if id(p) not in backbone_ids]

        optimizer_grouped_parameters = [
            {
                "params": backbone_params,
                "lr": self._lr_backbone,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": fusion_params,
                "lr": self._lr_fusion_head,
                "weight_decay": self.args.weight_decay,
            },
        ]
        try:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        except RuntimeError as e:
            logger.warning(f"FusedWeightedTrainer.create_optimizer failed: {e}; falling back to default optimizer")
            return super().create_optimizer()
        logger.info(
            f"FusedWeightedTrainer: differential LRs — backbone={self._lr_backbone:.2e}  "
            f"fusion_head={self._lr_fusion_head:.2e}"
        )
        return self.optimizer

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Overrides WeightedTrainer.compute_loss to add per-class gate alpha logging
        every 50 training steps.  Logs mean alpha separately for malicious and benign
        samples so gate collapse or over-reliance on features is immediately visible.
        """
        # Peek at labels before super() pops them from the inputs dict.
        labels_tensor: Optional[torch.Tensor] = inputs.get("labels")

        # Delegate loss + forward to parent; always request outputs for alpha access.
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # Gate monitoring every 50 steps during training (state is None during eval).
        step: int = self.state.global_step if self.state is not None else -1
        if step >= 0 and step % 50 == 0 and hasattr(outputs, "alpha") and labels_tensor is not None:
            try:
                with torch.no_grad():
                    alpha_mean = outputs.alpha.mean(dim=-1)  # [B] — mean over fusion dim
                    mal_mask = labels_tensor == 1
                    ben_mask = labels_tensor == 0
                    alpha_mal = alpha_mean[mal_mask].mean().item() if mal_mask.any() else float("nan")
                    alpha_ben = alpha_mean[ben_mask].mean().item() if ben_mask.any() else float("nan")
                logger.info(
                    f"[step {step}] gate alpha — malicious: {alpha_mal:.4f}  benign: {alpha_ben:.4f}"
                )
            except RuntimeError as e:
                logger.warning(f"Gate monitoring failed at step {step}: {e}")

        return (loss, outputs) if return_outputs else loss

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
        final_sids: list[int] = []
        for sid, probs_list in chunk_mal_probs.items():
            best_prob = max(probs_list)
            final_preds.append(1 if best_prob >= self.decision_threshold else 0)
            final_true.append(chunk_true[sid])
            final_sids.append(sid)

        # Zero-API ghost cluster recall — scripts whose malware_api_score == 0.
        # These are the hardest FNs; we track them separately per epoch.
        if self._zero_api_zscore is not None:
            try:
                all_fvecs: list[list[float]] = eval_ds["feature_vector"]
                # A script is "zero-API" when ALL its chunks have API score ≤ threshold.
                # feature_vector index 21 = malware_api_score (Z-scored).
                _thresh = self._zero_api_zscore + 1e-3
                zero_api_by_sid: dict[int, bool] = {}
                for i, fvec in enumerate(all_fvecs):
                    sid = script_ids[i]
                    if sid not in zero_api_by_sid:
                        zero_api_by_sid[sid] = True
                    zero_api_by_sid[sid] = zero_api_by_sid[sid] and (fvec[21] <= _thresh)

                za_preds = [p for sid, p in zip(final_sids, final_preds) if zero_api_by_sid.get(sid, False)]
                za_true  = [t for sid, t in zip(final_sids, final_true)  if zero_api_by_sid.get(sid, False)]
                if za_preds:
                    za_recall = float(recall_score(za_true, za_preds, average="binary", zero_division=0))
                    n_mal_za = sum(za_true)
                    logger.info(
                        f"[zero-API ghost cluster] n={len(za_preds)}  malicious={n_mal_za}  recall={za_recall:.4f}"
                    )
            except RuntimeError as e:
                logger.warning(f"Zero-API ghost cluster logging failed: {e}")

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

    # Serialize model_name + num_labels + all fusion config attrs from SimpleNamespace
    model_cfg = {
        "model_name": model.model_name,
        "num_labels": model.num_labels,
        **vars(model.config),
    }
    with open(os.path.join(output_dir, "fused_model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    inference_cfg["model_type"] = "fused"
    with open(os.path.join(output_dir, "inference_config.json"), "w") as f:
        json.dump(inference_cfg, f, indent=2)

    dest_scaler = os.path.join(output_dir, "feature_scaler.joblib")
    if os.path.abspath(scaler_path) != os.path.abspath(dest_scaler):
        shutil.copy(scaler_path, dest_scaler)

    logger.info(f"Saved fused model ({model.config.fusion_architecture}) to {output_dir}")


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

    model_name: str = cfg.pop("model_name")
    num_labels: int = cfg.pop("num_labels", 2)

    # Load weights first so we can detect old architecture from key names
    state = st_load_file(
        os.path.join(model_dir, "model.safetensors"),
        device=str(device),
    )

    if "fusion_architecture" not in cfg:
        # Detect architecture from state-dict keys (old checkpoints didn't store this)
        _is_old_gated = any(
            k.startswith("feat_proj.") or k.startswith("cls_head.")
            for k in state
        )
        if _is_old_gated:
            logger.warning(
                "fused_model_config.json missing 'fusion_architecture'; detected gated "
                "architecture from checkpoint keys (feat_proj/cls_head). "
                "Mapping old keys → new layout."
            )
            cfg["fusion_architecture"] = "gated"
            # Map old config keys → new gated_* names
            if "mlp_hidden_dim" in cfg:
                cfg["gated_proj_hidden_1"] = cfg.pop("mlp_hidden_dim")
            if "fusion_hidden_dim" in cfg:
                cfg["gated_proj_hidden_2"] = cfg.pop("fusion_hidden_dim")
            if "dropout_rate" in cfg:
                cfg["gated_proj_dropout"] = cfg.pop("dropout_rate")
            if "cls_head_dropout" in cfg:
                cfg["gated_fusion_dropout"] = cfg.pop("cls_head_dropout")
            # Derive gated_fusion_hidden from cls_head.0.weight shape [hidden, proj_hidden_2]
            if "cls_head.0.weight" in state:
                cfg.setdefault("gated_fusion_hidden", int(state["cls_head.0.weight"].shape[0]))
        else:
            logger.warning(
                "fused_model_config.json missing 'fusion_architecture'; defaulting to 'concat'. "
                "This checkpoint was saved with an older version of FusedCodeBERTClassifier."
            )
            cfg["fusion_architecture"] = "concat"
            if "mlp_hidden_dim" in cfg and "concat_mlp_hidden" not in cfg:
                cfg["concat_mlp_hidden"] = cfg.pop("mlp_hidden_dim")
            if "fusion_hidden_dim" in cfg and "concat_fusion_hidden" not in cfg:
                cfg["concat_fusion_hidden"] = cfg.pop("fusion_hidden_dim")
            if "dropout_rate" in cfg and "concat_dropout" not in cfg:
                cfg["concat_dropout"] = cfg.pop("dropout_rate")
            for _stale in ("cls_head_dropout",):
                cfg.pop(_stale, None)

    fusion_config = SimpleNamespace(**cfg)

    model = FusedCodeBERTClassifier(
        model_name=model_name,
        num_labels=num_labels,
        config=fusion_config,
    )

    # Rename old gated state-dict keys to current names before loading
    _KEY_PREFIX_RENAMES = {"feat_proj": "projection", "cls_head": "fusion_head"}
    remapped_state = {}
    for k, v in state.items():
        new_k = k
        for old_prefix, new_prefix in _KEY_PREFIX_RENAMES.items():
            if k.startswith(old_prefix + "."):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped_state[new_k] = v

    model.load_state_dict(remapped_state)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loaded fused model ({fusion_config.fusion_architecture}) from {model_dir} on {device}")
    return model, tokenizer
