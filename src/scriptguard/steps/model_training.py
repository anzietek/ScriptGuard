import json
import os
from typing import Any, Dict
from zenml import step, ArtifactConfig
from typing import Annotated
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from datasets import Dataset
from scriptguard.features.extractor import FeatureExtractor
from scriptguard.models.fused_classifier import (
    FusedCodeBERTClassifier,
    FusedDataCollator,
    FusedWeightedTrainer,
    save_fused_model,
)
from scriptguard.exceptions import ModelTrainingError
from scriptguard.utils.logger import logger


class BackboneUnfreezeCallback(TrainerCallback):
    # Unfreezes the BERT backbone at the start of epoch `freeze_epochs` (0-indexed).
    # Epoch 0 = first epoch (backbone frozen), epoch freeze_epochs = unfreeze point.

    def __init__(self, freeze_epochs: int) -> None:
        self.freeze_epochs = freeze_epochs
        self._unfrozen: bool = False

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._unfrozen:
            return
        model: Any = kwargs.get("model")
        if model is None or not hasattr(model, "bert"):
            return
        current_epoch: int = int(state.epoch) if state.epoch is not None else 0
        if current_epoch >= self.freeze_epochs:
            for param in model.bert.parameters():
                param.requires_grad = True
            self._unfrozen = True
            logger.info(
                f"BackboneUnfreezeCallback: unfroze BERT backbone at epoch {current_epoch} "
                f"(freeze_epochs={self.freeze_epochs})"
            )


def _get_class_weights(labels: list[int], num_labels: int = 2) -> torch.Tensor:
    n_samples = len(labels)
    class_counts = [labels.count(i) for i in range(num_labels)]
    weights = [
        n_samples / (num_labels * c) if c > 0 else 1.0
        for c in class_counts
    ]
    logger.info(f"Class counts: {class_counts}, weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


@step
def train_codebert(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Dict[str, Any],
    scaler_path: str,
) -> Annotated[str, ArtifactConfig(name="model_path")]:
    codebert_cfg = config.get("codebert", {})
    training_cfg = config.get("training", {})

    model_name: str = codebert_cfg.get("model_name", "microsoft/codebert-base")
    output_dir: str = codebert_cfg.get("output_dir", "/workspace/models/codebert")
    early_stopping_patience: int = training_cfg.get("early_stopping_patience", 3)
    max_tokens: int = codebert_cfg.get("max_tokens", 512)
    chunk_overlap: int = codebert_cfg.get("chunk_overlap", 50)
    decision_threshold: float = codebert_cfg.get("decision_threshold", 0.5)

    feature_dim: int = FeatureExtractor.FEATURE_DIM  # single source of truth
    mlp_hidden_dim: int = codebert_cfg.get("mlp_hidden_dim", 128)
    fusion_hidden_dim: int = codebert_cfg.get("fusion_hidden_dim", 256)
    dropout_rate: float = codebert_cfg.get("dropout_rate", 0.3)
    num_labels: int = codebert_cfg.get("num_labels", 2)
    logging_steps: int = codebert_cfg.get("logging_steps", 50)
    save_total_limit: int = codebert_cfg.get("save_total_limit", 2)
    metric_for_best_model: str = codebert_cfg.get("metric_for_best_model", "eval_recall")

    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Building FusedCodeBERTClassifier: {model_name}, feature_dim={feature_dim}")
        model = FusedCodeBERTClassifier(
            model_name=model_name,
            num_labels=num_labels,
            feature_dim=feature_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Backbone freeze: keep BERT frozen for the first N epochs so the fusion
        # head and projection learn a stable basis before full fine-tuning begins.
        backbone_freeze_epochs: int = codebert_cfg.get("backbone_freeze_epochs", 1)
        if backbone_freeze_epochs > 0:
            for param in model.bert.parameters():
                param.requires_grad = False
            logger.info(
                f"BERT backbone frozen for first {backbone_freeze_epochs} epoch(s); "
                "will unfreeze via BackboneUnfreezeCallback"
            )

        labels: list[int] = train_dataset["label"]
        class_weights = _get_class_weights(labels)

        focal_gamma: float = codebert_cfg.get("focal_gamma", 2.0)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=codebert_cfg.get("num_epochs", 5),
            per_device_train_batch_size=codebert_cfg.get("batch_size", 16),
            per_device_eval_batch_size=codebert_cfg.get("batch_size", 16),
            learning_rate=codebert_cfg.get("learning_rate", 2e-5),
            weight_decay=codebert_cfg.get("weight_decay", 0.01),
            warmup_steps=codebert_cfg.get("warmup_steps", 200),
            lr_scheduler_type=codebert_cfg.get("lr_scheduler_type", "cosine"),
            max_grad_norm=codebert_cfg.get("max_grad_norm", 1.0),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            bf16=False,
            save_safetensors=False,  # Custom nn.Module, not PreTrainedModel
            report_to=codebert_cfg.get("report_to", "none"),
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
        )

        collator = FusedDataCollator(tokenizer=tokenizer)

        trainer = FusedWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            decision_threshold=decision_threshold,
            callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            *(
                [BackboneUnfreezeCallback(freeze_epochs=backbone_freeze_epochs)]
                if backbone_freeze_epochs > 0
                else []
            ),
        ],
        )

        logger.info(f"Starting fused CodeBERT fine-tuning → {output_dir}")
        trainer.train()

        inference_config = {
            "max_tokens": max_tokens,
            "chunk_overlap": chunk_overlap,
            "decision_threshold": decision_threshold,
        }

        logger.info(f"Saving fused model to {output_dir}")
        save_fused_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            inference_cfg=inference_config,
            scaler_path=scaler_path,
        )

        logger.info(f"Training complete. Fused model saved to {output_dir}")
        return output_dir

    except Exception as exc:
        raise ModelTrainingError(f"Fused training failed: {exc}") from exc
