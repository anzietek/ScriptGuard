"""
Centralized prompt templates for ScriptGuard.
Ensures consistency between training and evaluation.
"""

from typing import List, Dict, Any
from scriptguard.utils.logger import logger


def format_training_prompt(code: str, label: str) -> str:
    """
    Format prompt for training with causal language modeling.
    Must match inference prompt structure exactly.

    Args:
        code: Source code to analyze
        label: Classification label ('benign' or 'malicious')

    Returns:
        Formatted training text
    """
    # Normalize label to uppercase for consistency
    label_normalized = label.upper() if label.lower() in ['benign', 'malicious'] else label

    return (
        f"Analyze if this code is malicious.\n\n"
        f"Code:\n{code}\n\n"
        f"Classification: {label_normalized}"  # Space after colon to match inference
    )


def format_inference_prompt(code: str, max_code_length: int = 500) -> str:
    """
    Format prompt for inference/evaluation.
    Must match the training format structure.
    Uses explicit constraints to force binary output.

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
        f"Classification: "  # Added space to anchor the response
    )


def parse_classification_output(generated_text: str) -> int:
    """
    Parse model output to extract binary classification.
    STRICT PARSING: Only accepts unambiguous MALICIOUS or BENIGN responses.

    Args:
        generated_text: Full generated text from model

    Returns:
        0 for benign, 1 for malicious
        Raises ValueError if response is ambiguous
    """
    # Extract text after "Classification:"
    if "Classification:" in generated_text:
        prediction_text = generated_text.split("Classification:")[-1].strip()
    else:
        prediction_text = generated_text.strip()

    # Get first line only (ignore any continuation)
    first_line = prediction_text.split('\n')[0].strip()

    # Normalize to lowercase for comparison
    prediction_lower = first_line.lower()

    # Count keyword occurrences
    has_malicious = any(word in prediction_lower for word in ['malicious', 'malware'])
    has_benign = any(word in prediction_lower for word in ['benign', 'safe', 'clean'])

    # STRICT VALIDATION: Both or neither -> ambiguous
    if has_malicious and has_benign:
        logger.warning(f"Ambiguous prediction contains both MALICIOUS and BENIGN: '{first_line}'")
        # Default to benign (conservative choice)
        return 0

    if not has_malicious and not has_benign:
        logger.warning(f"Unclear prediction, no valid keywords: '{first_line}'")
        # Check for numeric indicators as fallback
        if '1' in prediction_lower[:5]:
            return 1
        elif '0' in prediction_lower[:5]:
            return 0
        # Default to benign (conservative choice)
        return 0

    # Unambiguous response
    if has_malicious:
        return 1
    if has_benign:
        return 0

    # Should never reach here
    return 0


def format_fewshot_prompt(
    target_code: str,
    context_examples: List[Dict[str, Any]],
    max_code_length: int = 500,
    max_context_length: int = 300
) -> str:
    """
    Format Few-Shot prompt with similar code examples from RAG.

    This prompt includes retrieved similar code samples as context,
    enabling the model to make more informed classifications based on
    concrete examples rather than abstract CVE descriptions.

    Args:
        target_code: The code to analyze
        context_examples: List of similar code examples from Qdrant
            Each example should have:
            - code: str (code content)
            - label: str ("malicious" or "benign")
            - score: float (similarity score, optional)
        max_code_length: Maximum length for target code
        max_context_length: Maximum length for each context example

    Returns:
        Formatted Few-Shot prompt string
    """
    # Build context section
    context_lines = ["[CONTEXT]", "The following are examples of Python scripts and their security classification:", ""]

    for i, example in enumerate(context_examples, 1):
        code = example.get("code", "")
        label = example.get("label", "unknown").upper()
        score = example.get("score")

        # Truncate code
        truncated_code = code[:max_context_length]
        if len(code) > max_context_length:
            truncated_code += "..."

        # Add example
        score_info = f" (similarity: {score:.2f})" if score else ""
        context_lines.append(f"--- Example {i} ({label}){score_info} ---")
        context_lines.append(truncated_code)
        context_lines.append("")

    # Build task section
    truncated_target = target_code[:max_code_length]
    if len(target_code) > max_code_length:
        truncated_target += "..."

    task_lines = [
        "[TASK]",
        "Analyze the script below and determine if it is MALICIOUS or BENIGN.",
        "Consider the examples above as reference patterns.",
        "Answer with ONE WORD ONLY: either 'MALICIOUS' or 'BENIGN'.",
        "",
        "Script to analyze:",
        truncated_target,
        "",
        "Classification: "  # Added space to anchor the response
    ]

    # Combine all sections
    prompt = "\n".join(context_lines + task_lines)

    return prompt


def format_fewshot_prompt_balanced(
    target_code: str,
    malicious_examples: List[Dict[str, Any]],
    benign_examples: List[Dict[str, Any]],
    max_code_length: int = 500,
    max_context_length: int = 300
) -> str:
    """
    Format Few-Shot prompt with explicitly balanced examples.

    This ensures the model sees both malicious and benign examples,
    preventing bias towards one class.

    Args:
        target_code: The code to analyze
        malicious_examples: List of malicious code examples
        benign_examples: List of benign code examples
        max_code_length: Maximum length for target code
        max_context_length: Maximum length for each context example

    Returns:
        Formatted Few-Shot prompt string
    """
    # Combine and interleave examples (malicious, benign, malicious, benign, ...)
    context_examples = []

    max_len = max(len(malicious_examples), len(benign_examples))
    for i in range(max_len):
        if i < len(malicious_examples):
            context_examples.append(malicious_examples[i])
        if i < len(benign_examples):
            context_examples.append(benign_examples[i])

    # Use standard few-shot formatter
    return format_fewshot_prompt(
        target_code=target_code,
        context_examples=context_examples,
        max_code_length=max_code_length,
        max_context_length=max_context_length
    )

