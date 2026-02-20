from typing import Any, Dict, List
from zenml import step, ArtifactConfig
from typing import Annotated
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from scriptguard.materializers.dataset_materializer import HuggingFaceDatasetMaterializer
from scriptguard.steps.advanced_augmentation import generate_polymorphic_variant
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.utils.logger import logger


def _label_to_int(label: str) -> int:
    return 1 if label == "malicious" else 0


@step(output_materializers=HuggingFaceDatasetMaterializer)
def augment_and_tokenize(
    train_tokens: Dataset,
    train_data: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Annotated[Dataset, ArtifactConfig(name="augmented_train_tokens")]:
    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        logger.info("Augmentation disabled; returning original train_tokens")
        return train_tokens

    variants_per_sample: int = aug_cfg.get("variants_per_sample", 2)
    codebert_cfg = config.get("codebert", {})
    model_name: str = codebert_cfg.get("model_name", "microsoft/codebert-base")
    max_tokens: int = codebert_cfg.get("max_tokens", 512)
    overlap: int = codebert_cfg.get("chunk_overlap", 50)

    malicious_samples = [s for s in train_data if s.get("label") == "malicious"]
    logger.info(
        f"Augmenting {len(malicious_samples)} malicious samples "
        f"× {variants_per_sample} variants = {len(malicious_samples) * variants_per_sample} new samples"
    )

    augmented_samples: list[dict] = []
    for sample in malicious_samples:
        for _ in range(variants_per_sample):
            variant = generate_polymorphic_variant(sample)
            if variant.get("content", "").strip():
                augmented_samples.append(variant)

    if not augmented_samples:
        logger.warning("No augmented samples produced; returning original train_tokens")
        return train_tokens

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    script_id_offset = len(train_data)
    aug_chunks: list[dict] = []
    skipped = 0
    for i, sample in enumerate(augmented_samples):
        content = sample.get("content", "")
        label_int = _label_to_int(sample.get("label", "malicious"))
        script_id = script_id_offset + i
        try:
            chunks = sliding_window_chunks(
                tokenizer=tokenizer,
                text=content,
                max_length=max_tokens,
                overlap=overlap,
                script_id=script_id,
                label=label_int,
            )
            aug_chunks.extend(chunks)
        except Exception:
            skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} augmented samples during tokenization")

    aug_dataset = Dataset.from_list(aug_chunks)
    combined = concatenate_datasets([train_tokens, aug_dataset])

    logger.info(
        f"Augmentation complete: {len(train_tokens)} original + {len(aug_dataset)} augmented "
        f"= {len(combined)} total train chunks"
    )

    return combined
