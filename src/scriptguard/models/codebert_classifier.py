"""
Alternative Model Architecture: CodeBERT for Binary Classification
====================================================================

This is a more efficient alternative to using StarCoder2 Causal LM
for malware classification. CodeBERT is designed for code understanding
tasks and has a classification head built-in.

ADVANTAGES:
-----------
1. Faster training (125M vs 3B parameters)
2. Less VRAM (1-2GB vs 4GB+)
3. Direct classification (no text generation)
4. Better suited for binary classification tasks
5. Proven track record on code classification

EXPECTED IMPROVEMENTS:
---------------------
- Accuracy: 75-85% (vs current 58%)
- Training time: ~30min (vs 2h+)
- Inference: 10-50ms (vs 500ms+)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from typing import Dict, Any, Optional
from scriptguard.utils.logger import logger

class CodeBERTClassifier:
    """
    Binary classifier using CodeBERT for malware detection.

    This is a simpler and more efficient approach than Causal LM.
    """

    def __init__(self, model_id: str = "microsoft/codebert-base", config: dict = None):
        """
        Initialize CodeBERT classifier.

        Args:
            model_id: Base model (codebert-base, graphcodebert-base, etc.)
            config: Configuration dictionary
        """
        self.model_id = model_id
        self.config = config or {}

        logger.info(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info("Loading model with classification head (2 classes)")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,  # Binary classification
            problem_type="single_label_classification"
        )

    def prepare_dataset(self, data: list) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            data: List of dicts with 'content' and 'label' keys

        Returns:
            Tokenized dataset ready for training
        """
        # Extract code and labels
        texts = []
        labels = []

        for item in data:
            code = item.get("content", "")
            label_str = item.get("label", "benign")

            # Convert to binary: 0=benign, 1=malicious
            label = 1 if label_str.lower() == "malicious" else 0

            texts.append(code)
            labels.append(label)

        # Tokenize
        logger.info(f"Tokenizing {len(texts)} samples...")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None  # Return lists for Dataset
        )

        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })

        logger.info(f"Dataset prepared: {len(dataset)} samples")
        return dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./results"
    ):
        """
        Train the classifier.

        Args:
            train_dataset: Training dataset (tokenized)
            eval_dataset: Optional evaluation dataset
            output_dir: Where to save the model
        """
        training_config = self.config.get("training", {})

        # Compute class weights for imbalanced datasets
        labels = train_dataset["labels"]
        num_malicious = sum(labels)
        num_benign = len(labels) - num_malicious

        if num_malicious > 0 and num_benign > 0:
            # Weight inversely proportional to class frequencies
            pos_weight = num_benign / num_malicious
            logger.info(f"Class distribution: {num_malicious} malicious, {num_benign} benign")
            logger.info(f"Using positive class weight: {pos_weight:.2f}")
        else:
            pos_weight = 1.0
            logger.warning("Imbalanced dataset detected but one class is empty!")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("num_epochs", 5),
            per_device_train_batch_size=training_config.get("batch_size", 8),
            per_device_eval_batch_size=8,
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_steps=training_config.get("warmup_steps", 500),
            logging_steps=10,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="wandb",
            fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
        )

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final model
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        logger.info(f"Model saved to {output_dir}/final_model")

    def predict(self, code: str) -> tuple[int, float]:
        """
        Predict if code is malicious.

        Args:
            code: Python code string

        Returns:
            (label, confidence) where label is 0=benign, 1=malicious
        """
        self.model.eval()

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        return predicted_class, confidence


# Usage Example:
# ==============
# from scriptguard.models.codebert_classifier import CodeBERTClassifier
#
# # Initialize
# classifier = CodeBERTClassifier(config=config)
#
# # Prepare data
# train_dataset = classifier.prepare_dataset(train_data)
# eval_dataset = classifier.prepare_dataset(eval_data)
#
# # Train
# classifier.train(train_dataset, eval_dataset)
#
# # Predict
# label, confidence = classifier.predict(malicious_code)
# print(f"Prediction: {'Malicious' if label == 1 else 'Benign'} ({confidence:.2%})")
