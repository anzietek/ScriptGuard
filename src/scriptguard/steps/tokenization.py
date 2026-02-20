from typing import Any, Dict, List, Tuple
from zenml import step, ArtifactConfig
from typing import Annotated
from transformers import AutoTokenizer
from datasets import Dataset
from scriptguard.materializers.dataset_materializer import HuggingFaceDatasetMaterializer
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.utils.logger import logger


def _label_to_int(label: str) -> int:
    return 1 if label == "malicious" else 0


def _tokenize_split(
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    overlap: int,
    script_id_offset: int = 0,
) -> Dataset:
    all_chunks: list[dict] = []
    skipped = 0
    for i, sample in enumerate(samples):
        content = sample.get("content", "") or ""
        label_int = _label_to_int(sample.get("label", "benign"))
        script_id = script_id_offset + i
        try:
            chunks = sliding_window_chunks(
                tokenizer=tokenizer,
                text=content,
                max_length=max_length,
                overlap=overlap,
                script_id=script_id,
                label=label_int,
            )
            all_chunks.extend(chunks)
        except Exception:
            skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} samples during tokenization")
    logger.info(f"Produced {len(all_chunks)} chunks from {len(samples) - skipped} samples")
    return Dataset.from_list(all_chunks)


@step(
    output_materializers={
        "train_tokens": HuggingFaceDatasetMaterializer,
        "val_tokens": HuggingFaceDatasetMaterializer,
        "test_tokens": HuggingFaceDatasetMaterializer,
    }
)
def tokenize_data(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[
    Annotated[Dataset, ArtifactConfig(name="train_tokens")],
    Annotated[Dataset, ArtifactConfig(name="val_tokens")],
    Annotated[Dataset, ArtifactConfig(name="test_tokens")],
]:
    codebert_cfg = config.get("codebert", {})
    model_name: str = codebert_cfg.get("model_name", "microsoft/codebert-base")
    max_tokens: int = codebert_cfg.get("max_tokens", 512)
    overlap: int = codebert_cfg.get("chunk_overlap", 50)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Tokenizing train ({len(train_data)} samples)...")
    train_tokens = _tokenize_split(train_data, tokenizer, max_tokens, overlap)

    logger.info(f"Tokenizing val ({len(val_data)} samples)...")
    val_tokens = _tokenize_split(val_data, tokenizer, max_tokens, overlap)

    logger.info(f"Tokenizing test ({len(test_data)} samples)...")
    test_tokens = _tokenize_split(test_data, tokenizer, max_tokens, overlap)

    logger.info(
        f"Tokenization complete — train: {len(train_tokens)}, "
        f"val: {len(val_tokens)}, test: {len(test_tokens)} chunks"
    )

    return train_tokens, val_tokens, test_tokens
