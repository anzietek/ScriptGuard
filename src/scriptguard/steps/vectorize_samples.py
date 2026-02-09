"""
Synchronization Process: Training Data → Qdrant
Vectorizes verified code samples and uploads them to Qdrant for Few-Shot RAG.
"""

from typing import Dict, Any, Optional, List
from zenml import step
import os

from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger


@step
def vectorize_samples(
    data: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
    clear_existing: bool = False
) -> Dict[str, Any]:
    """
    Synchronize provided code samples to Qdrant vector database.

    This step:
    1. Takes the TRAINING dataset (list of dicts)
    2. Generates embeddings using a code-specific model
    3. Uploads vectors to Qdrant for similarity search during inference

    Args:
        data: List of training samples to vectorize
        config: Configuration dictionary from config.yaml
        clear_existing: If True, clear existing vectors before sync

    Returns:
        Dictionary with synchronization statistics
    """
    if not data:
        logger.warning("No data provided for vectorization.")
        return {
            "status": "skipped",
            "samples_vectorized": 0
        }

    # Check if vectorization is enabled via environment variable
    enable_vectorization = os.getenv("ENABLE_QDRANT_VECTORIZATION", "true").lower() == "true"
    
    if not enable_vectorization:
        logger.info("⚠️ Vectorization to Qdrant is DISABLED via ENABLE_QDRANT_VECTORIZATION env var.")
        logger.info("Skipping vectorization step.")
        return {
            "status": "skipped",
            "samples_vectorized": 0,
            "message": "Vectorization disabled by configuration"
        }

    logger.info("=" * 60)
    logger.info("VECTORIZING TRAINING SAMPLES (List → Qdrant)")
    logger.info("=" * 60)

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
        chunk_overlap=chunk_overlap,
        api_key=qdrant_config.get("api_key"),  # CRITICAL: Pass API key for connection consistency
        use_https=qdrant_config.get("use_https", False)
    )

    # Log connection information for diagnostics
    connection_info = code_store.get_connection_info()
    logger.info("=" * 60)
    logger.info("VECTORIZATION CONNECTION INFO")
    logger.info(f"  Type: {connection_info['connection_type']}")
    logger.info(f"  URL: {connection_info['connection_url']}")
    logger.info(f"  Collection: {connection_info['collection_name']}")
    logger.info(f"  API key: {'SET' if connection_info['has_api_key'] else 'NOT SET'}")
    logger.info("=" * 60)

    # Clear existing if requested
    if clear_existing:
        logger.info("Clearing existing code vectors...")
        code_store.clear_collection()

    # Get collection info before sync
    info_before = code_store.get_collection_info()
    logger.info(f"Collection before sync: {info_before.get('total_samples', 0)} samples")

    logger.info(f"Processing {len(data)} training samples...")

    # Prepare samples for vectorization
    samples_to_vectorize = []
    malicious_count = 0
    benign_count = 0

    for sample in data:
        # Separate concerns: point_id (for Qdrant) vs db_id (for PostgreSQL)
        db_id = sample.get("id")  # Real database ID (can be None for synthetic samples)

        # Generate point_id for Qdrant (always needed)
        if db_id is not None:
            # Use database ID as point ID
            point_id = db_id
        else:
            # Generate hash-based ID for synthetic/augmented samples
            import hashlib
            content_hash = hashlib.md5(sample.get("content", "").encode()).hexdigest()
            point_id = int(content_hash[:16], 16) % (2**63 - 1)

        label = sample.get("label", "unknown")
        if label == "malicious":
            malicious_count += 1
        elif label == "benign":
            benign_count += 1

        samples_to_vectorize.append({
            "id": point_id,  # Qdrant point ID
            "db_id": db_id,  # Real database ID (None if synthetic)
            "content": sample.get("content", ""),
            "label": label,
            "source": sample.get("source", "unknown"),
            "language": sample.get("metadata", {}).get("language", "python"),
            "metadata": sample.get("metadata", {})
        })

    logger.info(f"  - Malicious: {malicious_count}")
    logger.info(f"  - Benign: {benign_count}")

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
