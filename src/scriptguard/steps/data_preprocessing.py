from typing import List, Dict, Any
from zenml import step
from datasets import Dataset
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import format_training_prompt
from scriptguard.materializers.dataset_materializer import HuggingFaceDatasetMaterializer
from scriptguard.preprocessing.smart_truncator import smart_truncate, simple_truncate

@step(output_materializers=HuggingFaceDatasetMaterializer)
def preprocess_data(
    data: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
) -> Dataset:
    """
    Converts raw script data into a Hugging Face Dataset format.
    Prepares data for causal language modeling (next token prediction).
    """
    logger.info(f"Preprocessing {len(data)} samples.")

    config = config or {}
    training_cfg = config.get("training", {})

    # Training-time safety knobs
    skip_unknown_labels = training_cfg.get("skip_unknown_labels", True)
    # Avoid creating extremely long single examples that dominate batches/memory.
    # Uses tokenizer_max_length as a reasonable proxy; can be overridden.
    max_chars = training_cfg.get("preprocess_max_chars", training_cfg.get("tokenizer_max_length", 512) * 8)
    truncation_strategy = training_cfg.get("truncation_strategy", "simple")  # "simple" or "smart"

    # Log label distribution BEFORE preprocessing
    label_counts = {}
    for item in data:
        label = item.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS (before preprocessing)")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(data)}")
    for label, count in sorted(label_counts.items(), key=lambda x: str(x[0])):
        percentage = (count / len(data)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    logger.info("=" * 60)

    def _normalize_label(value: Any) -> str:
        # Accept common shapes:
        # - str: "benign"/"malicious"
        # - bool/int: 0/1 or True/False (assume 1=True=malicious)
        if isinstance(value, bool):
            return "malicious" if value else "benign"
        if isinstance(value, int):
            if value in (0, 1):
                return "malicious" if value == 1 else "benign"
            return "unknown"
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"benign", "malicious"}:
                return v
        return "unknown"

    def _extract_code(item: Dict[str, Any]) -> str:
        # Prefer content, then code; tolerate other common keys.
        for k in ("content", "code", "script", "text"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    # Format for causal LM: Include label in text for training
    formatted_data = []
    skipped_empty = 0
    skipped_unknown = 0
    truncated = 0

    for item in data:
        label = _normalize_label(item.get('label', 'unknown'))
        content = _extract_code(item)

        if not content:
            skipped_empty += 1
            continue

        if skip_unknown_labels and label == "unknown":
            skipped_unknown += 1
            continue

        if isinstance(max_chars, int) and max_chars > 0 and len(content) > max_chars:
            if truncation_strategy == "smart":
                content = smart_truncate(content, max_chars)
            else:
                content = simple_truncate(content, max_chars)
            truncated += 1

        # Use centralized prompt formatting
        text = format_training_prompt(code=content, label=label)

        # CRITICAL: Preserve metadata fields for vectorization
        # vectorize_samples needs: id, content, label, source, metadata
        formatted_data.append({
            "text": text,  # For training
            "id": item.get("id"),  # Database ID (None for synthetic)
            "content": content,  # Original code (for vectorization)
            "label": label,  # Normalized label
            "source": item.get("source", "unknown"),  # Data source
            "metadata": item.get("metadata", {})  # Additional metadata
        })

    dataset = Dataset.from_list(formatted_data)

    # Log post-filtering stats
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS (after preprocessing)")
    logger.info("=" * 60)
    logger.info(f"Created dataset with {len(dataset)} samples")
    logger.info(f"Skipped empty/no-code samples: {skipped_empty}")
    logger.info(f"Skipped unknown-label samples: {skipped_unknown} (skip_unknown_labels={skip_unknown_labels})")
    logger.info(f"Truncated long samples: {truncated} (preprocess_max_chars={max_chars})")
    logger.info("=" * 60)

    return dataset
