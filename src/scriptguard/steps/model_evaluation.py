from typing import Dict, Any
from zenml import step
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import format_inference_prompt, parse_classification_output
import numpy as np

@step
def evaluate_model(
    adapter_path: str,
    test_dataset: Dataset,
    base_model_id: str = "bigcode/starcoder2-3b",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluates the fine-tuned model on test set.

    Args:
        adapter_path: Path to trained LoRA adapter
        test_dataset: Test dataset (tokenized)
        base_model_id: Base model identifier
        config: Configuration dictionary from config.yaml

    Computes:
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix
    - Per-class metrics
    - Sample predictions for inspection
    """
    logger.info(f"Evaluating model from: {adapter_path}")
    logger.info(f"Test set size: {len(test_dataset)}")

    # Get evaluation config
    config = config or {}
    eval_config = config.get("training", {})
    max_new_tokens = eval_config.get("eval_max_new_tokens", 20)
    temperature = eval_config.get("eval_temperature", 0.1)
    max_code_length = eval_config.get("eval_max_code_length", 500)

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

        # Get true label - convert string labels to binary
        label_str = sample.get("label", sample.get("is_malicious", "benign"))
        if isinstance(label_str, str):
            true_label = 1 if label_str.lower() == "malicious" else 0
        else:
            true_label = int(label_str)
        y_true.append(true_label)

        # Get code from tokenized dataset (need to decode back)
        # The test_dataset is already tokenized, so we need to decode it
        if "input_ids" in sample:
            # Decode the tokenized input back to text
            input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            # Extract code portion from training format
            if "Code:\n" in input_text and "\n\nClassification:" in input_text:
                code = input_text.split("Code:\n")[1].split("\n\nClassification:")[0]
            else:
                code = input_text[:500]
        else:
            code = sample.get("code", sample.get("content", ""))

        # Use centralized prompt formatting (matches training format)
        prompt = format_inference_prompt(code=code, max_code_length=max_code_length)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Use centralized parsing
        predicted_label = parse_classification_output(generated_text)

        y_pred.append(predicted_label)

        # Store sample result for inspection (first 10 samples)
        if i < 10:
            sample_results.append({
                "code_snippet": code[:200],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "raw_prediction": generated_text.split("Classification:")[-1].strip()[:100]
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
