from typing import Dict, Any
from zenml import step
from datasets import Dataset
from scriptguard.models.qlora_finetuner import QLoRAFineTuner
from scriptguard.utils.logger import logger

@step
def train_model(dataset: Dataset, model_id: str = "bigcode/starcoder2-3b", config: Dict[str, Any] = None) -> str:
    """
    Fine-tunes the base model using QLoRA.

    Args:
        dataset: Training dataset
        model_id: Base model identifier
        config: Configuration dictionary from config.yaml

    Returns:
        Path to trained adapter
    """
    logger.info(f"Starting QLoRA fine-tuning for model: {model_id}")

    finetuner = QLoRAFineTuner(model_id=model_id, config=config or {})
    output_dir = config.get("training", {}).get("output_dir", "./model_checkpoints") if config else "./model_checkpoints"
    finetuner.train(dataset, output_dir=output_dir)

    return f"{output_dir}/final_adapter"
