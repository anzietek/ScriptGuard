from typing import Dict, Any, Optional
from zenml import step
from datasets import Dataset
from scriptguard.models.qlora_finetuner import QLoRAFineTuner
from scriptguard.utils.logger import logger

@step
def train_model(
    dataset: Dataset,
    model_id: str = "bigcode/starcoder2-3b",
    config: Dict[str, Any] = None,
    eval_dataset: Optional[Dataset] = None
) -> str:
    """
    Fine-tunes the base model using QLoRA.

    Args:
        dataset: Training dataset
        model_id: Base model identifier
        config: Configuration dictionary from config.yaml
        eval_dataset: Optional evaluation dataset for monitoring during training

    Returns:
        Path to trained adapter
    """
    logger.info(f"Starting QLoRA fine-tuning for model: {model_id}")
    # --- FIX START: FORCE BALANCE (CORRECTED) ---
    # Check balance
    labels = dataset['label']

    # FIX: Preprocessor returns strings "malicious"/"benign", not integers 0/1.
    # Handling both cases for safety.
    malicious_indices = [
        i for i, label in enumerate(labels)
        if label == "malicious" or label == 1
    ]
    benign_indices = [
        i for i, label in enumerate(labels)
        if label == "benign" or label == 0
    ]

    logger.info(f"Indices found: {len(malicious_indices)} malicious, {len(benign_indices)} benign")

    label_counts = {"malicious": len(malicious_indices), "benign": len(benign_indices)}
    logger.info(f"Original Training Distribution: {label_counts}")

    # Calculate minor class count
    min_count = min(len(malicious_indices), len(benign_indices))
    major_count = max(len(malicious_indices), len(benign_indices))

    # If imbalance is severe (e.g. > 1.5 ratio), force undersampling
    if min_count > 0 and major_count > min_count * 1.5:
        logger.warning(f"Severe class imbalance detected. Forcing undersampling to {min_count} samples per class.")

        import random
        random.seed(42)

        # Randomly select min_count from each
        malicious_sample = random.sample(malicious_indices, min(len(malicious_indices), min_count))
        benign_sample = random.sample(benign_indices, min(len(benign_indices), min_count))

        balanced_indices = malicious_sample + benign_sample
        random.shuffle(balanced_indices)

        dataset = dataset.select(balanced_indices)
        logger.info(f"Balanced Dataset Size: {len(dataset)}")
    elif min_count == 0:
        logger.error("CRITICAL: One class has 0 samples! Check data validation/preprocessing logic.")
    if eval_dataset:
        logger.info(f"Evaluation dataset provided with {len(eval_dataset)} samples")
    else:
        logger.info("No evaluation dataset provided - training without validation")

    finetuner = QLoRAFineTuner(model_id=model_id, config=config or {})
    output_dir = config.get("training", {}).get("output_dir", "./model_checkpoints") if config else "./model_checkpoints"
    finetuner.train(dataset, eval_dataset=eval_dataset, output_dir=output_dir)

    return f"{output_dir}/final_adapter"
