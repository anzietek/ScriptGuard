import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from scriptguard.utils.logger import logger

class QLoRAFineTuner:
    def __init__(self, model_id: str = "bigcode/starcoder2-3b", config: dict = None):
        self.model_id = model_id
        self.config = config or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, dataset: Dataset, output_dir: str = "./results"):
        logger.info(f"Tokenizing dataset with {len(dataset)} samples...")

        # Get config values with defaults
        training_config = self.config.get("training", {})
        max_length = training_config.get("tokenizer_max_length", 512)
        padding = training_config.get("tokenizer_padding", "max_length")
        truncation = training_config.get("tokenizer_truncation", True)

        # Determine if using dynamic padding
        use_dynamic_padding = padding == "dynamic"

        # Tokenize the dataset
        def tokenize_function(examples):
            # For dynamic padding, don't pad during tokenization
            # DataCollator will handle it during batching
            if use_dynamic_padding:
                result = self.tokenizer(
                    examples["text"],
                    truncation=truncation,
                    max_length=max_length,
                    padding=False,  # No padding here - collator does it
                )
            else:
                # Static padding during tokenization
                result = self.tokenizer(
                    examples["text"],
                    truncation=truncation,
                    max_length=max_length,
                    padding=padding,
                )
            # For causal LM, labels are the same as input_ids
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset"
        )

        logger.info(f"Tokenization complete. Sample count: {len(tokenized_dataset)}")
        logger.info(f"Dataset columns: {tokenized_dataset.column_names}")
        logger.info(f"Padding strategy: {'Dynamic (batch-level)' if use_dynamic_padding else 'Static (max_length)'}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        logger.info("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        model = prepare_model_for_kbit_training(model)

        # Get LoRA config from config.yaml
        lora_config = LoraConfig(
            r=training_config.get("lora_r", 16),
            lora_alpha=training_config.get("lora_alpha", 32),
            target_modules=training_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=training_config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA config applied. Trainable parameters: {model.print_trainable_parameters()}")

        # Get training hyperparameters from config
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.get("batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 2e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_steps=training_config.get("warmup_steps", 100),
            num_train_epochs=training_config.get("num_epochs", 3),
            fp16=training_config.get("fp16", False),
            bf16=training_config.get("bf16", True),
            optim=training_config.get("optim", "paged_adamw_8bit"),
            logging_steps=training_config.get("logging_steps", 10),
            eval_strategy=training_config.get("evaluation_strategy", "no"),
            eval_steps=training_config.get("eval_steps", 100),
            save_strategy="steps",
            save_steps=training_config.get("save_steps", 500),
            report_to="wandb",
            push_to_hub=False,
        )

        # Create data collator - handles dynamic padding if enabled
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if use_dynamic_padding else None  # Efficient for GPU
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        model.config.use_cache = False
        logger.info("Starting training...")
        trainer.train()
        
        # Save the adapter
        model.save_pretrained(f"{output_dir}/final_adapter")
        logger.info(f"Training completed. Adapter saved to {output_dir}/final_adapter")
