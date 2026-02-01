from typing import Dict
from zenml import step
import logging

logger = logging.getLogger(__name__)

@step
def evaluate_model(adapter_path: str) -> Dict[str, float]:
    """
    Evaluates the fine-tuned model.
    In a real scenario, this would compute metrics like accuracy, F1-score on a test set.
    """
    logger.info(f"Evaluating model at: {adapter_path}")
    
    # Mocked metrics
    metrics = {
        "accuracy": 0.95,
        "f1_score": 0.94,
        "precision": 0.96,
        "recall": 0.92
    }
    
    return metrics
