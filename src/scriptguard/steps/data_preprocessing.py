from typing import List, Dict, Any
from zenml import step
from datasets import Dataset
from scriptguard.utils.logger import logger

@step
def preprocess_data(data: List[Dict[str, Any]]) -> Dataset:
    """
    Converts raw script data into a Hugging Face Dataset format.
    Prepares data for causal language modeling (next token prediction).
    """
    logger.info(f"Preprocessing {len(data)} samples.")

    # Format for causal LM: Include label in text for training
    formatted_data = []
    for item in data:
        label = item.get('label', 'unknown')
        content = item.get('content', '')

        # Format: Instruction + Code + Label
        text = (
            f"Analyze if this code is malicious.\n\n"
            f"Code:\n{content}\n\n"
            f"Classification: {label}"
        )

        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Created dataset with {len(dataset)} samples")

    return dataset
