"""
Centralized prompt templates for ScriptGuard.
Ensures consistency between training and evaluation.
"""

def format_training_prompt(code: str, label: str) -> str:
    """
    Format prompt for training with causal language modeling.

    Args:
        code: Source code to analyze
        label: Classification label ('benign' or 'malicious')

    Returns:
        Formatted training text
    """
    return (
        f"Analyze if this code is malicious.\n\n"
        f"Code:\n{code}\n\n"
        f"Classification: {label}"
    )


def format_inference_prompt(code: str, max_code_length: int = 500) -> str:
    """
    Format prompt for inference/evaluation.
    Must match the training format structure.

    Args:
        code: Source code to analyze
        max_code_length: Maximum characters of code to include

    Returns:
        Formatted inference prompt
    """
    truncated_code = code[:max_code_length]
    return (
        f"Analyze if this code is malicious.\n\n"
        f"Code:\n{truncated_code}\n\n"
        f"Classification:"
    )


def parse_classification_output(generated_text: str) -> int:
    """
    Parse model output to extract binary classification.

    Args:
        generated_text: Full generated text from model

    Returns:
        0 for benign, 1 for malicious
    """
    # Extract text after "Classification:"
    if "Classification:" in generated_text:
        prediction_text = generated_text.split("Classification:")[-1].strip()
    else:
        prediction_text = generated_text

    # Normalize to lowercase
    prediction_lower = prediction_text.lower()

    # Check for malicious indicators
    if any(word in prediction_lower[:20] for word in ['malicious', 'malware', '1']):
        return 1

    # Check for benign indicators
    if any(word in prediction_lower[:20] for word in ['benign', 'safe', 'clean', '0']):
        return 0

    # Default to benign if unclear
    return 0
