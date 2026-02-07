"""
Synchronization Process: PostgreSQL → Qdrant
Vectorizes verified code samples and uploads them to Qdrant for Few-Shot RAG.
"""

from typing import Dict, Any, Optional
from zenml import step

from scriptguard.database.dataset_manager import DatasetManager
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger


@step
def vectorize_samples(
    config: Dict[str, Any] = None,
    max_samples: Optional[int] = None,
    clear_existing: bool = False
) -> Dict[str, Any]:
    """
    Synchronize code samples from PostgreSQL to Qdrant vector database.

    This step:
    1. Retrieves all verified code samples from PostgreSQL
    2. Generates embeddings using a code-specific model
    3. Uploads vectors to Qdrant for similarity search during inference

    Args:
        config: Configuration dictionary from config.yaml
        max_samples: Optional limit on number of samples to sync (for testing). If 0, skip vectorization.
        clear_existing: If True, clear existing vectors before sync

    Returns:
        Dictionary with synchronization statistics
    """
    # Skip vectorization if max_samples is explicitly 0
    if max_samples is not None and max_samples == 0:
        logger.info("Vectorization skipped (max_samples_to_vectorize=0)")
        return {
            "status": "skipped",
            "samples_vectorized": 0,
            "malicious_count": 0,
            "benign_count": 0
        }

    logger.info("=" * 60)
    logger.info("VECTORIZING CODE SAMPLES (PostgreSQL → Qdrant)")
    logger.info("=" * 60)

    # Initialize managers
    db_manager = DatasetManager()

    # Get Qdrant config
    config = config or {}
    qdrant_config = config.get("qdrant", {})
    embedding_config = config.get("code_embedding", {})

    # Extract embedding configuration
    code_embedding_model = embedding_config.get("model", "microsoft/unixcoder-base")
    pooling_strategy = embedding_config.get("pooling_strategy", "mean_pooling")
    normalize = embedding_config.get("normalize", True)
    max_length = embedding_config.get("max_code_length", 512)
    enable_chunking = embedding_config.get("enable_chunking", True)
    chunk_overlap = embedding_config.get("chunk_overlap", 64)

    logger.info(f"Embedding configuration:")
    logger.info(f"  Model: {code_embedding_model}")
    logger.info(f"  Pooling: {pooling_strategy}")
    logger.info(f"  Normalize: {normalize}")
    logger.info(f"  Chunking: {enable_chunking}")
    logger.info(f"  Max length: {max_length}, Overlap: {chunk_overlap}")

    # Initialize Code Similarity Store with enhanced configuration
    code_store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        embedding_model=code_embedding_model,
        pooling_strategy=pooling_strategy,
        normalize=normalize,
        max_length=max_length,
        enable_chunking=enable_chunking,
        chunk_overlap=chunk_overlap
    )

    # Clear existing if requested
    if clear_existing:
        logger.info("Clearing existing code vectors...")
        code_store.clear_collection()

    # Get collection info before sync
    info_before = code_store.get_collection_info()
    logger.info(f"Collection before sync: {info_before.get('total_samples', 0)} samples")

    # Retrieve samples from PostgreSQL
    logger.info("Retrieving samples from PostgreSQL...")

    # Get malicious samples
    malicious_samples = db_manager.get_all_samples(
        label="malicious",
        limit=max_samples // 2 if max_samples else None
    )

    # Get benign samples
    benign_samples = db_manager.get_all_samples(
        label="benign",
        limit=max_samples // 2 if max_samples else None
    )

    all_samples = malicious_samples + benign_samples

    logger.info(f"Retrieved {len(all_samples)} samples from PostgreSQL")
    logger.info(f"  - Malicious: {len(malicious_samples)}")
    logger.info(f"  - Benign: {len(benign_samples)}")

    if not all_samples:
        logger.warning("No samples found in PostgreSQL. Run data ingestion first.")
        return {
            "status": "no_data",
            "samples_vectorized": 0,
            "malicious_count": 0,
            "benign_count": 0
        }

    # Prepare samples for vectorization
    samples_to_vectorize = []
    for sample in all_samples:
        samples_to_vectorize.append({
            "id": sample["id"],
            "content": sample["content"],
            "label": sample["label"],
            "source": sample["source"],
            "language": sample.get("metadata", {}).get("language", "python"),
            "metadata": sample.get("metadata", {})
        })

    # Vectorize and upload to Qdrant
    logger.info("Generating embeddings and uploading to Qdrant...")
    code_store.upsert_code_samples(samples_to_vectorize, batch_size=100)

    # Get collection info after sync
    info_after = code_store.get_collection_info()

    logger.info("=" * 60)
    logger.info("VECTORIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples vectorized: {info_after.get('total_samples', 0)}")
    logger.info(f"  - Malicious: {info_after.get('malicious_samples', 0)}")
    logger.info(f"  - Benign: {info_after.get('benign_samples', 0)}")
    logger.info(f"Embedding dimension: {info_after.get('embedding_dim', 'N/A')}")
    logger.info(f"Collection status: {info_after.get('status', 'N/A')}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "samples_vectorized": info_after.get('total_samples', 0),
        "malicious_count": info_after.get('malicious_samples', 0),
        "benign_count": info_after.get('benign_samples', 0),
        "embedding_dim": info_after.get('embedding_dim', 0),
        "collection_info": info_after
    }


if __name__ == "__main__":
    # Test vectorization
    import os
    import yaml

    # Load config from environment variable or default
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run vectorization (limit to 100 samples for testing)
    result = vectorize_samples(config=config, max_samples=100, clear_existing=True)

    print(f"\nVectorization result: {result}")
