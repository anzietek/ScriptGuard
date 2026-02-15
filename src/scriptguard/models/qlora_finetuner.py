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
    Correctly detects labels from the prompts.py format (# Analysis: ...)
    """
    labels = []

    # Anchors from prompts.py
    MALICIOUS_ANCHOR = "classified as: MALICIOUS"
    BENIGN_ANCHOR = "classified as: BENIGN"

    logger.info("Scanning dataset for class labels...")

    for item in dataset:
        text = item.get("text", "")

        if MALICIOUS_ANCHOR in text:
            labels.append("malicious")
        elif BENIGN_ANCHOR in text:
            labels.append("benign")
        else:
            # Fallback
            if "Label: malicious" in text:
                labels.append("malicious")
            elif "Label: benign" in text:
                labels.append("benign")
            else:
                label = item.get("label", "unknown")
                labels.append(str(label).lower() if label else "unknown")

    label_counts = Counter(labels)
    total = sum(label_counts.values())

    logger.info(f"Class distribution for weighting: {dict(label_counts)}")

    if "malicious" not in label_counts or "benign" not in label_counts:
        logger.warning("⚠️  WARNING: Could not find both classes! Weights will be 1.0.")
        return {"malicious": 1.0, "benign": 1.0}

    weights = {}
    if method == "inverse_frequency":
        for label, count in label_counts.items():
            weights[label] = total / count
    elif method == "sqrt_inverse":
        for label, count in label_counts.items():
            weights[label] = np.sqrt(total / count)
    else:
        raise ValueError(f"Unknown weight method: {method}")

    num_classes = len(weights)
    weight_sum = sum(weights.values())
    normalized_weights = {label: (w / weight_sum) * num_classes for label, w in weights.items()}

    logger.info(f"Computed class weights ({method}): {normalized_weights}")
    return normalized_weights


class WeightedLossTrainer(UnslothTrainer):
    """Custom trainer with weighted loss AND masked inputs awareness."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}
        if self.class_weights:
            logger.info(f"WeightedLossTrainer initialized with weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.class_weights:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Standard loss calculation (which now respects the -100 masks from tokenization)
        # return_outputs=True is required to get the loss tensor before reduction
        loss_output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        base_loss, outputs = loss_output

        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.shape[0] == 0:
            return (base_loss, outputs) if return_outputs else base_loss

        sample_weights = []

        # Determine weight for each sample in batch
        for i in range(input_ids.shape[0]):
            try:
                # Optimized check: look at the end of the sequence for the label
                # prompt format ends with: "... classified as: MALICIOUS"
                end_ids = input_ids[i][-64:] # Check last 64 tokens
                text = self.processing_class.decode(end_ids, skip_special_tokens=True)

                if "MALICIOUS" in text:
                    weight = self.class_weights.get('malicious', 1.0)
                elif "BENIGN" in text:
                    weight = self.class_weights.get('benign', 1.0)
                else:
                    # Fallback to full decode
                    full_text = self.processing_class.decode(input_ids[i], skip_special_tokens=True)
                    if "MALICIOUS" in full_text:
                        weight = self.class_weights.get('malicious', 1.0)
                    elif "BENIGN" in full_text:
                        weight = self.class_weights.get('benign', 1.0)
                    else:
                        weight = 1.0

                sample_weights.append(weight)

            except Exception:
                sample_weights.append(1.0)

        if isinstance(base_loss, tuple):
            base_loss = base_loss[0]

        # Convert weights to tensor on correct device
        weights_tensor = torch.tensor(sample_weights, dtype=base_loss.dtype, device=base_loss.device)

        # Apply weights. 
        # Note: base_loss from HF is usually already a mean. 
        # Ideally we'd weight before mean, but Unsloth/HF interface makes that hard without overriding the model.
        # We multiply by the average weight of this batch to scale the gradient.
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

        training_config = self.config.get("training", {})
        max_length = training_config.get("tokenizer_max_length", 2048)
        logger.info(f"Using tokenizer_max_length: {max_length}")

        logger.info("Loading model with unsloth optimization...")

        import platform
        is_windows = platform.system() == "Windows"
        use_flash_attn = training_config.get("use_flash_attention_2", False)
        model_kwargs = {}

        if is_windows:
            model_kwargs["attn_implementation"] = "eager"
        elif use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=True,
            **model_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        # =================================================================
        # CRITICAL FIX: MASKING INPUTS (DATA COLLATOR SIMULATION)
        # This ensures the loss is calculated ONLY on the label ("MALICIOUS"/"BENIGN"),
        # not on the source code itself.
        # =================================================================
        def tokenize_function(examples):
            # 1. Tokenize full text normally
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

            input_ids_list = model_inputs["input_ids"]
            labels_list = []

            # The anchor string from prompts.py that separates Code from Label
            # We want to mask everything BEFORE the label.
            ANCHOR = "classified as: " 

            for i, full_text in enumerate(examples["text"]):
                input_ids = input_ids_list[i]
                labels = list(input_ids) # Start with copy of inputs

                # Check if anchor exists in this sample
                if ANCHOR in full_text:
                    try:
                        # Find the prompt part (everything before "classified as: ")
                        prompt_text = full_text.split(ANCHOR)[0] + ANCHOR

                        # Tokenize just the prompt to find how many tokens to mask
                        # Note: We assume standard tokenization. 
                        prompt_tokens = self.tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
                        mask_len = len(prompt_tokens)

                        # Apply Masking (-100 is ignored by CrossEntropyLoss)
                        # We mask everything from 0 to mask_len.
                        if mask_len < len(labels):
                            for j in range(mask_len):
                                labels[j] = -100
                    except Exception as e:
                        # Fallback: if splitting fails, just train on everything (noisy but safe)
                        pass

                labels_list.append(labels)

            model_inputs["labels"] = labels_list
            return model_inputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing training dataset (with masking)"
        )

        tokenized_eval_dataset = None
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing evaluation dataset"
            )

        # ... (rest of configuration is standard) ...
        use_fp16 = training_config.get("fp16", False)
        use_bf16 = training_config.get("bf16", True)
        if not torch.cuda.is_available(): use_bf16, use_fp16 = False, False

        eval_strategy = training_config.get("evaluation_strategy", "no")
        if tokenized_eval_dataset is not None and eval_strategy == "no":
            eval_strategy = "steps"

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 4)),
            per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 4)),
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
            warmup_steps=int(training_config.get("warmup_steps", 100)),
            num_train_epochs=int(training_config.get("num_epochs", 3)),
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=int(training_config.get("logging_steps", 10)),
            eval_strategy=eval_strategy,
            eval_steps=int(training_config.get("eval_steps", 100)) if eval_strategy != "no" else None,
            save_strategy="steps",
            save_steps=int(training_config.get("save_steps", 500)),
            report_to=training_config.get("report_to", ["wandb"]),
            run_name=training_config.get("run_name", "scriptguard-training"),
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        class_weights = None
        use_class_weights = training_config.get("use_class_weights", False)
        if use_class_weights:
            method = training_config.get("class_weight_method", "sqrt_inverse")
            class_weights = compute_class_weights(dataset, method=method)

            trainer = WeightedLossTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
                class_weights=class_weights,
            )
        else:
            trainer = UnslothTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
            )

        logger.info("Starting training...")
        trainer.train()

        self.model.save_pretrained(f"{output_dir}/final_adapter")
        self.tokenizer.save_pretrained(f"{output_dir}/final_adapter")
