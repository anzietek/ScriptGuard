from typing import Dict, Any
from zenml import step
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import (
    format_inference_prompt,
    format_fewshot_prompt
)
from scriptguard.rag.code_similarity_store import CodeSimilarityStore


class StrictBinaryClassificationProcessor(LogitsProcessor):
    """
    Strict constrained generation: ONLY allow BENIGN or MALICIOUS tokens.
    Forces the model to output exactly one of these two words by setting
    all other token logits to -inf for the FIRST token only.
    After the first token, allows natural completion of the word.
    """
    def __init__(self, benign_token_id: int, malicious_token_id: int, prompt_length: int):
        """
        Args:
            benign_token_id: Token ID for " BENIGN" first token (with leading space)
            malicious_token_id: Token ID for " MALICIOUS" first token (with leading space)
            prompt_length: Length of the prompt in tokens (to detect when we start generating)
        """
        self.benign_token_id = benign_token_id
        self.malicious_token_id = malicious_token_id
        self.prompt_length = prompt_length
        self.valid_tokens = {benign_token_id, malicious_token_id}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Set all token logits to -inf except for BENIGN and MALICIOUS (first token only).

        Args:
            input_ids: Already generated token IDs
            scores: Logits for next token prediction

        Returns:
            Modified scores with only valid classification tokens allowed (first token only)
        """
        # Only constrain the FIRST generated token
        current_length = input_ids.shape[1]
        if current_length > self.prompt_length:
            # After first token, allow natural completion
            return scores

        # For first token: Create a mask with -inf for all tokens except valid ones
        mask = torch.full_like(scores, float('-inf'))

        # Set valid tokens to original scores (or boost them slightly)
        mask[:, self.benign_token_id] = scores[:, self.benign_token_id] + 5.0
        mask[:, self.malicious_token_id] = scores[:, self.malicious_token_id] + 5.0

        return mask


@step
def evaluate_model(
    adapter_path: str,
    test_dataset: Dataset,
    base_model_id: str = "bigcode/starcoder2-3b",
    config: Dict[str, Any] = None,
    use_fewshot_rag: bool = True  # NEW: Enable Few-Shot RAG
) -> Dict[str, Any]:
    """
    Evaluates the fine-tuned model on test set.

    Args:
        adapter_path: Path to trained LoRA adapter
        test_dataset: Test dataset (tokenized)
        base_model_id: Base model identifier
        config: Configuration dictionary from config.yaml
        use_fewshot_rag: If True, use Few-Shot RAG with code similarity search

    Computes:
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix
    - Per-class metrics
    - Sample predictions for inspection
    """
    logger.info(f"Evaluating model from: {adapter_path}")
    logger.info(f"Test set size: {len(test_dataset)}")
    logger.info(f"Few-Shot RAG: {'ENABLED' if use_fewshot_rag else 'DISABLED'}")

    # Get evaluation config
    config = config or {}
    eval_config = config.get("training", {})
    max_new_tokens = eval_config.get("eval_max_new_tokens", 20)
    temperature = eval_config.get("eval_temperature", 0.1)
    max_code_length = eval_config.get("eval_max_code_length", 500)

    # Initialize Code Similarity Store for Few-Shot RAG
    code_store = None
    if use_fewshot_rag:
        try:
            logger.info("Initializing Code Similarity Store for Few-Shot RAG...")
            qdrant_config = config.get("qdrant", {})
            code_embedding_model = config.get("code_embedding", {}).get(
                "model",
                "microsoft/unixcoder-base"
            )

            code_store = CodeSimilarityStore(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333),
                collection_name="code_samples",
                embedding_model=code_embedding_model
            )

            # Check if collection has data
            info = code_store.get_collection_info()
            if info.get("total_samples", 0) == 0:
                logger.warning("Code similarity store is empty. Run vectorize_samples step first.")
                logger.warning("Falling back to standard inference without RAG.")
                code_store = None
            else:
                logger.info(f"✓ Code store ready: {info.get('total_samples', 0)} samples")

        except Exception as e:
            logger.error(f"Failed to initialize Code Similarity Store: {e}")
            logger.warning("Falling back to standard inference without RAG.")
            code_store = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with proper configuration for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if device == "cuda":
            # Load model on GPU without device_map to avoid PEFT adapter issues
            logger.info("Attempting to load model on GPU with float16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # NO device_map="auto" - causes issues with PEFT adapter loading
            )
            base_model = base_model.to(device)
            logger.info("✓ Model loaded on GPU with float16")
        else:
            raise RuntimeError("CUDA not available")
    except Exception as e:
        logger.warning(f"GPU loading failed: {e}")
        logger.info("Falling back to CPU (evaluation will be slower)...")
        # Final fallback: Load on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        base_model = base_model.to("cpu")
        device = "cpu"
        logger.info("✓ Model loaded on CPU")

    # Load LoRA adapter - must be after base model is loaded and placed on device
    logger.info(f"Loading PEFT adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Prepare predictions
    y_true = []
    y_pred = []
    sample_results = []

    logger.info("Running inference on test set...")

    for i, sample in enumerate(test_dataset):
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(test_dataset)} samples")

        # Get code from raw dataset - should have 'code' or 'content' field
        code = sample.get("code", sample.get("content", ""))
        
        if not code:
            logger.warning(f"Sample {i} has no code content, skipping")
            continue

        # Get true label - convert string labels to binary
        label_str = sample.get("label", sample.get("is_malicious", "benign"))
        if isinstance(label_str, str):
            true_label = 1 if label_str.lower() == "malicious" else 0
        else:
            true_label = int(label_str)

        # Generate prompt with Few-Shot RAG if available
        if code_store is not None:
            # Retrieve similar code examples from Qdrant
            try:
                similar_examples = code_store.search_similar_code(
                    query_code=code,
                    k=3,  # Retrieve top 3 similar examples
                    balance_labels=True,  # Ensure mix of malicious/benign
                    score_threshold=0.3
                )

                if similar_examples:
                    # Use Few-Shot prompt with context
                    prompt = format_fewshot_prompt(
                        target_code=code,
                        context_examples=similar_examples,
                        max_code_length=max_code_length,
                        max_context_length=300
                    )

                    if i < 3:  # Log first few prompts for debugging
                        logger.debug(f"Sample {i} using Few-Shot RAG with {len(similar_examples)} examples")
                else:
                    # No similar examples found, use standard prompt
                    prompt = format_inference_prompt(code=code, max_code_length=max_code_length)

            except Exception as e:
                logger.warning(f"RAG retrieval failed for sample {i}: {e}")
                prompt = format_inference_prompt(code=code, max_code_length=max_code_length)
        else:
            # Standard prompt without RAG
            prompt = format_inference_prompt(code=code, max_code_length=max_code_length)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # STRICT CONSTRAINED GENERATION: Force model to only output BENIGN or MALICIOUS
        # Get token IDs for the two valid words (without special tokens, with leading space)
        benign_tokens = tokenizer.encode(" BENIGN", add_special_tokens=False)
        malicious_tokens = tokenizer.encode(" MALICIOUS", add_special_tokens=False)

        # Debug token encoding on first sample
        if i == 0:
            logger.info(f"Token IDs for ' BENIGN': {benign_tokens}")
            logger.info(f"Token IDs for ' MALICIOUS': {malicious_tokens}")
            logger.info(f"Decoded ' BENIGN': {[tokenizer.decode([t]) for t in benign_tokens]}")
            logger.info(f"Decoded ' MALICIOUS': {[tokenizer.decode([t]) for t in malicious_tokens]}")

        # Get first token of each word (this is what model will generate first)
        benign_token_id = benign_tokens[0] if benign_tokens else None
        malicious_token_id = malicious_tokens[0] if malicious_tokens else None

        if benign_token_id is None or malicious_token_id is None:
            logger.error("Failed to get token IDs for BENIGN/MALICIOUS. Using fallback generation.")
            # Fallback: standard generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            # Create strict constraint processor
            prompt_length_tokens = inputs['input_ids'].shape[1]
            constraint_processor = StrictBinaryClassificationProcessor(
                benign_token_id=benign_token_id,
                malicious_token_id=malicious_token_id,
                prompt_length=prompt_length_tokens
            )

            if i < 2:  # Debug first few samples
                logger.debug(f"Using StrictBinaryClassificationProcessor: BENIGN={benign_token_id}, MALICIOUS={malicious_token_id}")

            # Generate with STRICT CONSTRAINTS using LogitsProcessor
            # Allow up to 5 tokens to complete the full word (MALICIOUS = 4 tokens, BENIGN = 3 tokens)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,  # Allow full word completion
                    do_sample=False,  # Deterministic greedy decoding
                    num_beams=1,  # Greedy search
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=[constraint_processor]  # Apply strict constraint
                )

        # Decode ONLY the newly generated tokens (not the entire sequence)
        prompt_length_tokens = inputs['input_ids'].shape[1]
        generated_token_ids = outputs[0][prompt_length_tokens:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        # Log first few predictions for debugging
        if i < 5:
            logger.info(f"Sample {i} prediction:")
            logger.info(f"  True label: {true_label}")
            logger.info(f"  Generated token(s): '{generated_text}'")
            logger.info(f"  Generated token IDs: {generated_token_ids.tolist()}")

        # Parse prediction: check if generated text contains MALICIOUS or BENIGN
        generated_upper = generated_text.upper()
        if "MALICIOUS" in generated_upper:
            predicted_label = 1
        elif "BENIGN" in generated_upper:
            predicted_label = 0
        else:
            # Fallback: check first character (M vs B)
            predicted_label = 1 if generated_upper.startswith("M") else 0
            logger.warning(f"Sample {i}: Unexpected output '{generated_text}', using fallback prediction")

        # Add to results lists (only after successful prediction)
        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Store sample result for inspection (first 10 samples)
        if len(sample_results) < 10:
            sample_results.append({
                "code_snippet": code[:200],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "raw_prediction": generated_text
            })

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate

    # Try to compute ROC AUC if we have both classes
    roc_auc = None
    avg_precision = None
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            avg_precision = average_precision_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")

    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(avg_precision) if avg_precision is not None else None,
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
    logger.info(f"Test Set Size: {len(test_dataset)}")
    logger.info(f"")
    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1 Score:      {f1:.4f}")
    logger.info(f"Specificity:   {specificity:.4f}")
    if roc_auc is not None:
        logger.info(f"ROC AUC:       {roc_auc:.4f}")
    if avg_precision is not None:
        logger.info(f"Avg Precision: {avg_precision:.4f}")
    logger.info(f"")
    logger.info(f"False Positive Rate: {fpr:.4f}")
    logger.info(f"False Negative Rate: {fnr:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TN: {tn:4d}  FP: {fp:4d}")
    logger.info(f"FN: {fn:4d}  TP: {tp:4d}")
    logger.info("=" * 60)

    return metrics
