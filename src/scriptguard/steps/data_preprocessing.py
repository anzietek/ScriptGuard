from typing import List, Dict, Any
from zenml import step
from datasets import Dataset
from scriptguard.utils.logger import logger

@step
def preprocess_data(data: List[Dict[str, Any]]) -> Dataset:
    """
    Converts raw script data into a Hugging Face Dataset format.
    """
    logger.info(f"Preprocessing {len(data)} samples.")
    
    # Simple formatting: "Script: ... Label: ..."
    formatted_data = []
    for item in data:
        text = f"Script:\n{item['content']}\n\nLabel: {item['label']}"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    return dataset
