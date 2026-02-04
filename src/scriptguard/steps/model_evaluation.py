from typing import Dict, Any
from zenml import step
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
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
    parse_classification_output,
    format_fewshot_prompt
)
from scriptguard.rag.code_similarity_store import CodeSimilarityStore


class ClassificationConstraintProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor that heavily biases towards MALICIOUS/BENIGN tokens.
    Forces the model to choose between only these two classification outputs.

    Strategy: Allow only first token of " MALICIOUS" or " BENIGN" (with leading space)
    since that's what comes after "Classification: "
    """
    def __init__(self, tokenizer, boost_factor: float = 10.0):
        """
        Args:
            tokenizer: The tokenizer to use for encoding target words
            boost_factor: How much to boost the target tokens (log-scale)
        """
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.prompt_length = None  # Will be set on first call

        # Get FIRST token of each word (with leading space, since prompt ends with ": ")
        malicious_with_space = tokenizer.encode(" MALICIOUS", add_special_tokens=False)
        benign_with_space = tokenizer.encode(" BENIGN", add_special_tokens=False)

        # Also try lowercase versions
        malicious_lower = tokenizer.encode(" malicious", add_special_tokens=False)
        benign_lower = tokenizer.encode(" benign", add_special_tokens=False)

        # Only allow FIRST token of each word
        self.valid_first_tokens = set()
        if malicious_with_space:
            self.valid_first_tokens.add(malicious_with_space[0])
        if benign_with_space:
            self.valid_first_tokens.add(benign_with_space[0])
        if malicious_lower:
            self.valid_first_tokens.add(malicious_lower[0])
        if benign_lower:
            self.valid_first_tokens.add(benign_lower[0])

        logger.debug(f"ClassificationConstraintProcessor initialized:")
        logger.debug(f"  Valid first tokens: {self.valid_first_tokens}")
        logger.debug(f"  ' MALICIOUS' tokens: {malicious_with_space}")
        logger.debug(f"  ' BENIGN' tokens: {benign_with_space}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Boost scores for MALICIOUS/BENIGN tokens, heavily penalize others.

        Args:
            input_ids: Already generated token IDs
            scores: Logits for next token prediction

        Returns:
            Modified scores with boosted classification tokens
        """
        # Initialize prompt length on first call
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]

        # Calculate how many tokens we've generated after the prompt
        num_generated = input_ids.shape[1] - self.prompt_length

        # Only constrain the FIRST generated token
        if num_generated > 0:
            # After first token, allow model to continue naturally
            return scores

        # For the first token: heavily constrain to valid first tokens
        vocab_size = scores.shape[-1]
        penalty = -1000.0  # Very strong penalty

        for token_id in range(vocab_size):
            if token_id not in self.valid_first_tokens:
                scores[:, token_id] += penalty
            else:
                # Boost valid classification tokens
                scores[:, token_id] += self.boost_factor

        return scores

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

        # CONSTRAINT GENERATION: Use custom LogitsProcessor
        # This forces the model to only generate MALICIOUS or BENIGN tokens
        constraint_processor = ClassificationConstraintProcessor(
            tokenizer=tokenizer,
            boost_factor=15.0  # Very strong bias towards classification tokens
        )

        logits_processor = LogitsProcessorList([constraint_processor])

        if i < 2:  # Debug first few samples
            logger.debug(f"Using ClassificationConstraintProcessor with boost_factor=15.0")

        # Generate prediction with STRICT CONSTRAINTS
        # Strategy: Force model to choose between MALICIOUS/BENIGN only
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,  # Allow only 1-3 tokens (one word)
                min_new_tokens=1,  # Force at least one token
                do_sample=False,  # Deterministic (greedy) generation
                num_beams=1,  # No beam search for speed
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor  # Apply constraint processor
            )

        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the newly generated part (after the prompt)
        prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_only = generated_text[prompt_length:].strip()

        # Log first few predictions for debugging
        if i < 5:
            logger.info(f"Sample {i} prediction:")
            logger.info(f"  True label: {true_label}")
            logger.info(f"  Generated text: '{generated_only}'")
            logger.info(f"  Full output: '{generated_text[-200:]}'")

        # Use centralized parsing
        try:
            predicted_label = parse_classification_output(generated_text)
        except Exception as e:
            logger.error(f"Failed to parse prediction for sample {i}: {e}")
            logger.error(f"  Generated text: {generated_only}")
            # Default to benign on parsing error (conservative)
            predicted_label = 0

        # Add to results lists (only after successful prediction)
        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Store sample result for inspection (first 10 samples)
        if len(sample_results) < 10:
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
