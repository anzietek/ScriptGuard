import os
import platform

# Force disable torch.compile on Windows at module import time
if platform.system() == "Windows":
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from unsloth import FastLanguageModel, UnslothTrainer
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
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
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
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
            report_to="wandb",
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

        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # CRITICAL: Trainer needs tokenizer reference!
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training with unsloth optimization...")
        trainer.train()
        
        self.model.save_pretrained(f"{output_dir}/final_adapter")
        self.tokenizer.save_pretrained(f"{output_dir}/final_adapter")
        logger.info(f"Training completed. Adapter saved to {output_dir}/final_adapter")
