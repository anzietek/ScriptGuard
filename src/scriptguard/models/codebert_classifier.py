from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset
from scriptguard.utils.logger import logger


def compute_metrics(pred: EvalPrediction) -> dict:
    # NOTE: HuggingFace Trainer operates at CHUNK level during training.
    # These metrics are chunk-level (not script-level) because the Trainer has
    # no access to script_id for aggregation. The final evaluate_codebert step
    # computes the authoritative script-level metrics on the test set.
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tp = int(np.sum((preds == 1) & (labels == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "fpr": fpr,
        "specificity": specificity,
        "mcc": mcc,
    }


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: Optional[torch.Tensor] = None, focal_gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype) if self.class_weights is not None else None
        loss_fct = FocalLoss(gamma=self.focal_gamma, weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CodeBERTClassifier:
    def __init__(self, model_name: str = "microsoft/codebert-base", num_labels: int = 2) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loading model with {num_labels} output labels")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            use_safetensors=True,
        )

    def get_class_weights(self, labels: list[int]) -> torch.Tensor:
        n_samples = len(labels)
        n_classes = self.num_labels
        class_counts = [labels.count(i) for i in range(n_classes)]
        weights = [
            n_samples / (n_classes * c) if c > 0 else 1.0
            for c in class_counts
        ]
        logger.info(f"Class counts: {class_counts}, weights: {[f'{w:.3f}' for w in weights]}")
        return torch.tensor(weights, dtype=torch.float32)

    def build_trainer(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str,
        config: dict,
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 3,
    ) -> WeightedTrainer:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get("num_epochs", 5),
            per_device_train_batch_size=config.get("batch_size", 16),
            per_device_eval_batch_size=config.get("batch_size", 16),
            learning_rate=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_steps=config.get("warmup_steps", 200),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_recall",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            bf16=False,
            save_safetensors=True,
            report_to="none",
            logging_steps=50,
            save_total_limit=2,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        focal_gamma = config.get("focal_gamma", 2.0)

        return WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
