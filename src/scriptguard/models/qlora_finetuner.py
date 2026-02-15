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
    Compute class weights matching the prompts.py format.
    """
    labels = []
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
            label = item.get("label")
            if label:
                labels.append(str(label).lower())
            else:
                labels.append("unknown")

    label_counts = Counter(labels)
    total = sum(label_counts.values())
    logger.info(f"Class distribution: {dict(label_counts)}")

    if "malicious" not in label_counts or "benign" not in label_counts:
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

    logger.info(f"Computed weights: {normalized_weights}")
    return normalized_weights


class WeightedLossTrainer(UnslothTrainer):
    """
    Custom trainer that applies class weights AND supports masked inputs.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.class_weights:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        loss_output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        base_loss, outputs = loss_output
        input_ids = inputs.get("input_ids")

        if input_ids is None or input_ids.shape[0] == 0:
            return (base_loss, outputs) if return_outputs else base_loss

        sample_weights = []

        for i in range(input_ids.shape[0]):
            try:
                end_ids = input_ids[i][-64:]
                text = self.processing_class.decode(end_ids, skip_special_tokens=True)

                if "MALICIOUS" in text:
                    weight = self.class_weights.get('malicious', 1.0)
                elif "BENIGN" in text:
                    weight = self.class_weights.get('benign', 1.0)
                else:
                    weight = 1.0
                sample_weights.append(weight)
            except:
                sample_weights.append(1.0)

        if isinstance(base_loss, tuple):
            base_loss = base_loss[0]

        weights_tensor = torch.tensor(sample_weights, dtype=base_loss.dtype, device=base_loss.device)
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
        training_config = self.config.get("training", {})

        # =========================================================
        # CRITICAL FIX: AUTO-SPLIT DATASET
        # =========================================================
        if eval_dataset is None:
            split_size = float(training_config.get("test_split_size", 0.0))
            if split_size > 0:
                logger.info(f"No eval_dataset provided. Splitting train dataset automatically (test_size={split_size})...")
                # Shuffle before splitting to ensure random distribution
                dataset = dataset.shuffle(seed=42)
                split_data = dataset.train_test_split(test_size=split_size, seed=42)
                dataset = split_data["train"]
                eval_dataset = split_data["test"]
                logger.info(f"Split complete: Train={len(dataset)}, Eval={len(eval_dataset)}")
            else:
                logger.warning("No eval_dataset provided and test_split_size is 0. Evaluation will be SKIPPED!")

        max_length = training_config.get("tokenizer_max_length", 2048)

        logger.info("Loading model with unsloth...")
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
        )

        # =========================================================
        # MASKING LOGIC
        # =========================================================
        def tokenize_and_mask(examples):
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

            input_ids_list = model_inputs["input_ids"]
            labels_list = []
            ANCHOR = "# Analysis: The script above is classified as:"

            for i, full_text in enumerate(examples["text"]):
                input_ids = input_ids_list[i]
                labels = list(input_ids)

                if ANCHOR in full_text:
                    try:
                        parts = full_text.split(ANCHOR)
                        prompt_text = parts[0] + ANCHOR 
                        prompt_tokens = self.tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
                        mask_len = len(prompt_tokens)

                        limit = min(mask_len, len(labels))
                        for j in range(limit):
                            labels[j] = -100
                    except Exception:
                        pass

                labels_list.append(labels)

            model_inputs["labels"] = labels_list
            return model_inputs

        logger.info("Tokenizing and MASKING dataset...")
        tokenized_dataset = dataset.map(tokenize_and_mask, batched=True, desc="Processing Train Data")

        tokenized_eval_dataset = None
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(tokenize_and_mask, batched=True, desc="Processing Eval Data")

        # Config adjustments for Trainer
        eval_strategy = training_config.get("evaluation_strategy", "no")
        if tokenized_eval_dataset is not None and eval_strategy == "no":
            eval_strategy = "steps"

        # Ensure eval_steps is set if strategy is steps
        eval_steps = int(training_config.get("eval_steps", 50))
        if eval_strategy == "steps" and eval_steps <= 0:
            eval_steps = 10

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
            warmup_steps=int(training_config.get("warmup_steps", 100)),
            num_train_epochs=int(training_config.get("num_epochs", 3)),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            # EVAL CONFIGURATION
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            # ------------------
            report_to=["wandb"],
            run_name="scriptguard-training",
        )

        trainer_cls = UnslothTrainer
        class_weights = None

        if training_config.get("use_class_weights", False):
            class_weights = compute_class_weights(dataset)
            trainer_cls = WeightedLossTrainer

        trainer = trainer_cls(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            **({"class_weights": class_weights} if class_weights else {})
        )

        logger.info("Starting training with MASKED INPUTS...")
        trainer.train()

        self.model.save_pretrained(f"{output_dir}/final_adapter")
        self.tokenizer.save_pretrained(f"{output_dir}/final_adapter")
