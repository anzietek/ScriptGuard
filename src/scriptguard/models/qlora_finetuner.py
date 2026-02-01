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
    def __init__(self, model_id: str = "bigcode/starcoder2-3b"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def train(self, dataset: Dataset, output_dir: str = "./results"):
        logger.info(f"Tokenizing dataset with {len(dataset)} samples...")

        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize with padding and truncation
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
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

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA config applied. Trainable parameters: {model.print_trainable_parameters()}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            max_steps=100,
            report_to="wandb",
            push_to_hub=False,
            save_strategy="steps",
            save_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        model.config.use_cache = False
        logger.info("Starting training...")
        trainer.train()
        
        # Save the adapter
        model.save_pretrained(f"{output_dir}/final_adapter")
        logger.info(f"Training completed. Adapter saved to {output_dir}/final_adapter")
