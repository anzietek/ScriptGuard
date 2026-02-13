import os
import platform

# Force disable torch.compile on Windows at module import time
if platform.system() == "Windows":
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from unsloth import FastLanguageModel, UnslothTrainer
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
from collections import Counter
import numpy as np
from scriptguard.utils.logger import logger

# Configure torch._dynamo on Windows
if platform.system() == "Windows":
    try:
        torch._dynamo.config.suppress_errors = True  # type: ignore
        torch._dynamo.config.disable = True  # type: ignore
        logger.info("Windows detected - torch.compile disabled in qlora_finetuner")
    except (AttributeError, ImportError):
        logger.warning("Could not disable torch._dynamo in qlora_finetuner")
        pass

def compute_class_weights(dataset: Dataset, method: str = "sqrt_inverse") -> dict:
    """
    Compute class weights for imbalanced datasets.

    Args:
        dataset: Dataset with 'text' field containing label information
        method: Weight computation method - "inverse_frequency" or "sqrt_inverse"

    Returns:
        Dictionary mapping label to weight tensor
    """
    # Extract labels from dataset
    # Labels are in the text field in format "Label: <label>\nCode: ..."
    labels = []
    for item in dataset:
        text = item.get("text", "")
        # Parse label from prompt format
        if "Label: malicious" in text:
            labels.append("malicious")
        elif "Label: benign" in text:
            labels.append("benign")
        else:
            # Fallback: try to get from other fields
            label = item.get("label", "unknown")
            labels.append(label)

    # Count occurrences
    label_counts = Counter(labels)
    total = sum(label_counts.values())

    logger.info(f"Class distribution for weighting: {dict(label_counts)}")

    # Compute weights
    weights = {}
    if method == "inverse_frequency":
        # w_i = N / n_i (inverse frequency)
        for label, count in label_counts.items():
            weights[label] = total / count
    elif method == "sqrt_inverse":
        # w_i = sqrt(N / n_i) (gentler weighting)
        for label, count in label_counts.items():
            weights[label] = np.sqrt(total / count)
    else:
        raise ValueError(f"Unknown weight method: {method}")

    # Normalize weights so they sum to number of classes
    num_classes = len(weights)
    weight_sum = sum(weights.values())
    normalized_weights = {label: (w / weight_sum) * num_classes for label, w in weights.items()}

    logger.info(f"Computed class weights ({method}): {normalized_weights}")
    return normalized_weights


class WeightedLossTrainer(UnslothTrainer):
    """Custom trainer with weighted loss for imbalanced datasets."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}

        # Convert weights to tensor for efficiency
        # Note: This is a simplified approach - for true per-sample weighting,
        # we'd need to parse labels from each sample during training
        if self.class_weights:
            logger.info(f"WeightedLossTrainer initialized with weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss for imbalanced datasets using sample-level weighting.

        This implementation applies weights at the sample level rather than token level,
        which is more appropriate for instruction-tuned models where we want to emphasize
        learning from underrepresented classes.

        Args:
            model: The model being trained
            inputs: Dictionary of input tensors (includes input_ids and labels)
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in the batch (for compatibility with transformers>=4.46)

        Returns:
            Weighted loss tensor, or tuple of (loss, outputs) if return_outputs=True
        """
        if not self.class_weights:
            # No weights configured, use default loss
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Compute standard loss first (always request outputs for proper unpacking)
        loss_output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # Always unpack since parent was called with return_outputs=True
        base_loss, outputs = loss_output

        # Decode input_ids to determine sample class
        # The prompt format is: "... classified as: MALICIOUS" or "... classified as: BENIGN"
        input_ids = inputs.get("input_ids")

        if input_ids is None or input_ids.shape[0] == 0:
            # Fallback: no weighting if we can't decode
            return (base_loss, outputs) if return_outputs else base_loss

        # Calculate per-sample weights
        sample_weights = []

        for i in range(input_ids.shape[0]):
            # Decode the text for this sample
            try:
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)

                # Determine class based on prompt content
                # Training prompts have format: "... classified as: MALICIOUS" or "... classified as: BENIGN"
                if "MALICIOUS" in text.upper():
                    weight = self.class_weights.get('malicious', 1.0)
                elif "BENIGN" in text.upper():
                    weight = self.class_weights.get('benign', 1.0)
                else:
                    # Unknown class, use neutral weight
                    weight = 1.0

                sample_weights.append(weight)

            except Exception as e:
                # Fallback on decode error
                logger.warning(f"Failed to decode sample {i} for class weighting: {e}")
                sample_weights.append(1.0)

        # Ensure base_loss is a tensor (defensive check)
        if isinstance(base_loss, tuple):
            base_loss = base_loss[0]

        # Convert to tensor and compute weighted loss
        weights_tensor = torch.tensor(sample_weights, dtype=base_loss.dtype, device=base_loss.device)

        # Apply sample weights: scale loss by average weight
        # This preserves gradient scale while emphasizing minority class
        avg_weight = weights_tensor.mean()
        weighted_loss = base_loss * avg_weight

        return (weighted_loss, outputs) if return_outputs else weighted_loss


class QLoRAFineTuner:
    def __init__(self, model_id: str = "bigcode/starcoder2-3b", config: dict = None):
        self.model_id = model_id
        self.config = config or {}
        self.model = None
        self.tokenizer = None

    def train(self, dataset: Dataset, eval_dataset: Dataset = None, output_dir: str = "./results"):
        logger.info(f"Tokenizing dataset with {len(dataset)} samples...")
        if eval_dataset:
            logger.info(f"Evaluation dataset with {len(eval_dataset)} samples will be used during training")

        training_config = self.config.get("training", {})
        max_length = training_config.get("tokenizer_max_length", 512)

        logger.info("Loading model with unsloth optimization...")

        # Unsloth may use flash attention by default (even if not explicitly enabled)
        # Flash Attention requires dropout=0.0, so we disable it unconditionally on Windows
        import platform
        is_windows = platform.system() == "Windows"

        use_flash_attn = training_config.get("use_flash_attention_2", False)

        # Prepare model config overrides
        model_kwargs = {}

        # Platform-specific attention implementation
        if is_windows:
            # Force eager attention on Windows to avoid flex_attention errors
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["attention_dropout"] = 0.0
            model_kwargs["residual_dropout"] = 0.0
            model_kwargs["embedding_dropout"] = 0.0
            logger.info("Windows detected - using eager attention (flex_attention disabled)")
        elif use_flash_attn:
            logger.info("Flash Attention 2 enabled - disabling all dropout layers")
            model_kwargs["attention_dropout"] = 0.0
            model_kwargs["residual_dropout"] = 0.0
            model_kwargs["embedding_dropout"] = 0.0
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=True,
            **model_kwargs
        )

        # Ensure tokenizer has pad_token (required for DataCollator)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

        logger.info("Adding LoRA adapters with unsloth...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=int(training_config.get("lora_r", 16)),
            target_modules=training_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_alpha=int(training_config.get("lora_alpha", 32)),
            lora_dropout=float(training_config.get("lora_dropout", 0.05)),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,  # No padding during tokenization - DataCollator will handle it
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing training dataset"
        )

        logger.info(f"Training tokenization complete. Sample count: {len(tokenized_dataset)}")

        tokenized_eval_dataset = None
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing evaluation dataset"
            )
            logger.info(f"Evaluation tokenization complete. Sample count: {len(tokenized_eval_dataset)}")

        has_cuda = torch.cuda.is_available()
        has_bf16 = has_cuda and torch.cuda.is_bf16_supported()

        use_fp16 = training_config.get("fp16", False)
        use_bf16 = training_config.get("bf16", True)

        if use_bf16 and not has_bf16:
            logger.warning("BF16 requested but not supported. Falling back to FP16.")
            use_bf16 = False
            use_fp16 = has_cuda

        if not has_cuda:
            logger.warning("CUDA not available. Training on CPU (this will be very slow).")
            use_bf16 = False
            use_fp16 = False

        logger.info(f"Training precision: BF16={use_bf16}, FP16={use_fp16}, Device={'cuda' if has_cuda else 'cpu'}")

        eval_strategy = training_config.get("evaluation_strategy", "no")
        if eval_strategy != "no" and tokenized_eval_dataset is None:
            logger.warning("evaluation_strategy is set but no eval_dataset provided. Setting to 'no'.")
            eval_strategy = "no"
        elif tokenized_eval_dataset is not None and eval_strategy == "no":
            logger.info("eval_dataset provided. Enabling evaluation strategy 'steps'.")
            eval_strategy = "steps"

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", training_config.get("batch_size", 4))),
            per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", training_config.get("batch_size", 4))),
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
            label_smoothing_factor=float(training_config.get("label_smoothing_factor", 0.0)),
            warmup_steps=int(training_config.get("warmup_steps", 100)),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
            num_train_epochs=int(training_config.get("num_epochs", 3)),
            fp16=use_fp16,
            bf16=use_bf16,
            optim=training_config.get("optim", "adamw_8bit"),
            logging_steps=int(training_config.get("logging_steps", 10)),
            eval_strategy=eval_strategy,
            eval_steps=int(training_config.get("eval_steps", 100)) if eval_strategy != "no" else None,
            save_strategy="steps",
            save_steps=int(training_config.get("save_steps", 500)),
            load_best_model_at_end=True if eval_strategy != "no" else False,
            metric_for_best_model="eval_loss" if eval_strategy != "no" else None,
            group_by_length=training_config.get("group_by_length", False),
            tf32=training_config.get("tf32", True),
            report_to=training_config.get("report_to", ["wandb"]),
            run_name=training_config.get("run_name", "scriptguard-training"),
            push_to_hub=False,
        )


        # Use dynamic padding via DataCollator
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None! Cannot create DataCollator.")

        logger.info(f"Creating DataCollator with pad_token: {self.tokenizer.pad_token}")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Setup callbacks (early stopping if enabled)
        callbacks = []
        if training_config.get("early_stopping", False) and eval_strategy != "no":
            early_stopping_patience = int(training_config.get("early_stopping_patience", 3))
            early_stopping_threshold = float(training_config.get("early_stopping_threshold", 0.0))
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            ))
            logger.info(f"Early stopping enabled: patience={early_stopping_patience}, threshold={early_stopping_threshold}")

        # Compute class weights if enabled
        class_weights = None
        use_class_weights = training_config.get("use_class_weights", False)
        if use_class_weights:
            method = training_config.get("class_weight_method", "sqrt_inverse")
            class_weights = compute_class_weights(dataset, method=method)
            logger.info(f"Using weighted loss with {method} method")

            # Use custom trainer with class weights
            trainer = WeightedLossTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                class_weights=class_weights,
            )
        else:
            # Use standard trainer
            trainer = UnslothTrainer(
                model=self.model,
                tokenizer=self.tokenizer,  # CRITICAL: Trainer needs tokenizer reference!
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )

        logger.info("Starting training with unsloth optimization...")
        trainer.train()
        
        self.model.save_pretrained(f"{output_dir}/final_adapter")
        self.tokenizer.save_pretrained(f"{output_dir}/final_adapter")
        logger.info(f"Training completed. Adapter saved to {output_dir}/final_adapter")
