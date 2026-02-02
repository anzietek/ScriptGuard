from typing import List, Dict, Any
from zenml import step
from datasets import Dataset
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import format_training_prompt

@step
def preprocess_data(data: List[Dict[str, Any]]) -> Dataset:
    """
    Converts raw script data into a Hugging Face Dataset format.
    Prepares data for causal language modeling (next token prediction).
    """
    logger.info(f"Preprocessing {len(data)} samples.")

    # Log label distribution BEFORE preprocessing
    label_counts = {}
    for item in data:
        label = item.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS (before preprocessing)")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(data)}")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(data)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    logger.info("=" * 60)

    # Format for causal LM: Include label in text for training
    formatted_data = []
    for item in data:
        label = item.get('label', 'unknown')
        content = item.get('content', '')

        # Use centralized prompt formatting
        text = format_training_prompt(code=content, label=label)

        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Created dataset with {len(dataset)} samples")

    return dataset
