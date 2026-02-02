"""
Centralized prompt templates for ScriptGuard.
Ensures consistency between training and evaluation.
"""

from typing import List, Dict, Any


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
        "",
        "Script to analyze:",
        truncated_target,
        "",
        "Classification:"
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

