import json
import os
from typing import Any, Dict
from zenml import step, ArtifactConfig
from typing import Annotated
from datasets import Dataset
from scriptguard.models.codebert_classifier import CodeBERTClassifier
from scriptguard.exceptions import ModelTrainingError
from scriptguard.utils.logger import logger


@step
def train_codebert(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Dict[str, Any],
) -> Annotated[str, ArtifactConfig(name="model_path")]:
    codebert_cfg = config.get("codebert", {})
    training_cfg = config.get("training", {})

    model_name: str = codebert_cfg.get("model_name", "microsoft/codebert-base")
    output_dir: str = codebert_cfg.get("output_dir", "/workspace/models/codebert")
    early_stopping_patience: int = training_cfg.get("early_stopping_patience", 3)
    max_tokens: int = codebert_cfg.get("max_tokens", 512)
    chunk_overlap: int = codebert_cfg.get("chunk_overlap", 50)

    os.makedirs(output_dir, exist_ok=True)

    try:
        classifier = CodeBERTClassifier(model_name=model_name)

        labels: list[int] = train_dataset["label"]
        class_weights = classifier.get_class_weights(labels)

        trainer = classifier.build_trainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            config=codebert_cfg,
            class_weights=class_weights,
            early_stopping_patience=early_stopping_patience,
        )

        logger.info(f"Starting CodeBERT fine-tuning → {output_dir}")
        trainer.train()

        logger.info(f"Saving model and tokenizer to {output_dir}")
        trainer.save_model(output_dir)
        classifier.tokenizer.save_pretrained(output_dir)

        decision_threshold: float = codebert_cfg.get("decision_threshold", 0.5)
        inference_config = {"max_tokens": max_tokens, "chunk_overlap": chunk_overlap, "decision_threshold": decision_threshold}
        with open(os.path.join(output_dir, "inference_config.json"), "w") as f:
            json.dump(inference_config, f)

        logger.info(f"Training complete. Model saved to {output_dir}")
        return output_dir

    except Exception as exc:
        raise ModelTrainingError(f"Training failed: {exc}") from exc
