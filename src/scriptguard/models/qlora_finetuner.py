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
import logging

logger = logging.getLogger(__name__)

class QLoRAFineTuner:
    def __init__(self, model_id: str = "bigcode/starcoder2-3b"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def train(self, dataset: Dataset, output_dir: str = "./results"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # Depends on model architecture
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            max_steps=100, # Small for demonstration
            report_to="wandb",  # Weights & Biases for better LLM tracking
            push_to_hub=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        model.config.use_cache = False
        trainer.train()
        
        # Save the adapter
        model.save_pretrained(f"{output_dir}/final_adapter")
        logger.info(f"Training completed. Adapter saved to {output_dir}/final_adapter")
