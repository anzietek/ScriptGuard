from typing import Dict, Any
from zenml import step
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from scriptguard.utils.logger import logger
import numpy as np

@step
def evaluate_model(
    adapter_path: str,
    test_dataset: Dataset,
    base_model_id: str = "bigcode/starcoder2-3b"
) -> Dict[str, Any]:
    """
    Evaluates the fine-tuned model on test set.

    Computes:
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix
    - Per-class metrics
    - Sample predictions for inspection
    """
    logger.info(f"Evaluating model from: {adapter_path}")
    logger.info(f"Test set size: {len(test_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Prepare predictions
    y_true = []
    y_pred = []
    sample_results = []

    logger.info("Running inference on test set...")

    for i, sample in enumerate(test_dataset):
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(test_dataset)} samples")

        # Get true label
        true_label = sample.get("label", sample.get("is_malicious", 0))
        y_true.append(true_label)

        # Prepare prompt for classification
        code = sample.get("code", sample.get("content", ""))
        prompt = f"Analyze if this code is malicious (0=benign, 1=malicious):\n\n{code[:500]}\n\nLabel:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_text = generated_text.split("Label:")[-1].strip()

        # Parse prediction (look for 0 or 1)
        if "0" in predicted_text[:5]:
            predicted_label = 0
        elif "1" in predicted_text[:5]:
            predicted_label = 1
        else:
            # Default to benign if unclear
            predicted_label = 0
            logger.warning(f"Unclear prediction: {predicted_text[:50]}")

        y_pred.append(predicted_label)

        # Store sample result for inspection (first 10 samples)
        if i < 10:
            sample_results.append({
                "code_snippet": code[:200],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "raw_prediction": predicted_text[:100]
            })

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "sample_predictions": sample_results,
        "test_set_size": len(test_dataset)
    }

    # Log summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TN: {tn:4d}  FP: {fp:4d}")
    logger.info(f"FN: {fn:4d}  TP: {tp:4d}")
    logger.info("=" * 60)

    return metrics
