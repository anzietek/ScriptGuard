#!/usr/bin/env python3
"""
Phase 1: Create Jina-v3 Test Collection

Sample 5,000 balanced samples from PostgreSQL and vectorize with Jina-v3.

Configuration:
- Model: jinaai/jina-embeddings-v3
- Dimension: 768 (UniXcoder compatible via Matryoshka)
- Task adapter: retrieval.v2
- Max length: 8192 tokens
- Collection: code_samples_jina_test

Success Criteria:
- 5,000 samples vectorized (2,500 benign + 2,500 malicious)
- All embeddings 768d with L2 norm ≈ 1.0
- Qdrant collection created and populated
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# Import ScriptGuard modules
from scriptguard.rag.embedding_service import EmbeddingService
from scriptguard.database import get_connection, return_connection
from scriptguard.utils.logger import logger


def create_jina_embedding_service():
    """Create Jina-v3 embedding service with 768d output."""
    logger.info("Initializing Jina-v3 embedding service...")

    # Create config for Jina-v3 with 768d dimension (UniXcoder compatible)
    config = {
        "code_embedding": {
            "model": "jinaai/jina-embeddings-v3",
            "jina_v3": {
                "enabled": True,
                "task_adapter": "retrieval.v2",
                "output_dimension": 768,  # Matryoshka: 768d for UniXcoder compatibility
                "max_length": 2048,  # Reduced from 8192 for faster CPU encoding
                "trust_remote_code": True
            }
        }
    }

    service = EmbeddingService(
        model_name="jinaai/jina-embeddings-v3",
        pooling_strategy="mean_pooling",
        normalize=True,
        max_length=2048,  # Reduced from 8192 for faster CPU encoding
        device="cpu",  # Use CPU to avoid OOM on 4GB GPU
        config=config
    )

    logger.info(f"✓ Jina-v3 service ready")
    logger.info(f"  Dimension: {service.embedding_dim}")
    logger.info(f"  Max length: 8192 tokens")
    logger.info(f"  Task adapter: retrieval.v2")

    return service


def create_test_collection(client, collection_name="code_samples_jina_test"):
    """Create temporary Jina-v3 collection."""
    logger.info(f"\nCreating temporary collection: {collection_name}")

    # Delete if exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"  Deleted existing collection")
    except:
        pass

    # Create collection with 768d vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,  # Jina-v3 Matryoshka dimension (UniXcoder compatible)
            distance=models.Distance.COSINE
        )
    )

    logger.info(f"✓ Collection created: {collection_name}")
    return collection_name


def fetch_training_samples(max_samples=5000):
    """Fetch balanced training samples from PostgreSQL."""
    logger.info(f"\nFetching {max_samples} training samples from database...")

    conn = get_connection()
    cursor = conn.cursor()

    # Fetch balanced samples
    samples_per_label = max_samples // 2

    # Fetch benign samples
    cursor.execute("""
        SELECT id, content, label, source
        FROM samples
        WHERE label = 'benign'
        AND content IS NOT NULL
        AND LENGTH(content) > 50
        ORDER BY RANDOM()
        LIMIT %s
    """, (samples_per_label,))

    benign_samples = cursor.fetchall()

    # Fetch malicious samples
    cursor.execute("""
        SELECT id, content, label, source
        FROM samples
        WHERE label = 'malicious'
        AND content IS NOT NULL
        AND LENGTH(content) > 50
        ORDER BY RANDOM()
        LIMIT %s
    """, (samples_per_label,))

    malicious_samples = cursor.fetchall()

    cursor.close()
    return_connection(conn)

    all_samples = benign_samples + malicious_samples

    logger.info(f"✓ Fetched {len(all_samples)} samples:")
    logger.info(f"  Benign:    {len(benign_samples)} ({len(benign_samples)/len(all_samples)*100:.1f}%)")
    logger.info(f"  Malicious: {len(malicious_samples)} ({len(malicious_samples)/len(all_samples)*100:.1f}%)")

    return all_samples


def vectorize_and_upload(embedding_service, client, collection_name, samples):
    """Vectorize samples and upload to Qdrant."""
    logger.info(f"\nVectorizing {len(samples)} samples with Jina-v3...")

    # Extract content and metadata
    codes = [sample['content'] for sample in samples]  # content column
    ids = [sample['id'] for sample in samples]    # id column
    labels = [sample['label'] for sample in samples]  # label column
    sources = [sample['source'] for sample in samples] # source column

    # Generate embeddings in batches (smaller batch for CPU)
    batch_size = 8  # Reduced from 32 to avoid memory issues with 8192 token context
    all_embeddings = []

    logger.info(f"Generating embeddings (batch_size={batch_size})...")
    total_batches = (len(codes) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(codes), batch_size), 1):
        batch_codes = codes[i:i+batch_size]
        batch_embeddings = embedding_service.encode(batch_codes, batch_size=batch_size, show_progress=False)
        all_embeddings.append(batch_embeddings)

        # Progress logging EVERY batch
        samples_done = min(i+batch_size, len(codes))
        percent = (samples_done / len(codes)) * 100
        logger.info(f"  Batch {batch_idx}/{total_batches}: {samples_done}/{len(codes)} samples ({percent:.1f}%)")

    embeddings = np.vstack(all_embeddings)
    logger.info(f"✓ Generated embeddings: {embeddings.shape}")

    # Verify embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"  L2 norm stats: mean={norms.mean():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

    # Upload to Qdrant
    logger.info(f"\nUploading to Qdrant collection '{collection_name}'...")

    points = []
    for idx, (sample_id, code, label, source, embedding) in enumerate(zip(ids, codes, labels, sources, embeddings)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "db_id": sample_id,
                    "content": code,  # Use 'content' to match database schema
                    "label": label,
                    "source": source,
                    "chunk_index": 0  # Treat as single document
                }
            )
        )

    # Upload in batches
    upload_batch_size = 100
    for i in range(0, len(points), upload_batch_size):
        batch = points[i:i+upload_batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f"  Uploaded {min(i+upload_batch_size, len(points))}/{len(points)} points")

    logger.info(f"✓ Upload complete: {len(points)} points")

    # Verify collection
    collection_info = client.get_collection(collection_name)
    logger.info(f"\nCollection verification:")
    logger.info(f"  Total points: {collection_info.points_count}")
    logger.info(f"  Vector dimension: {collection_info.config.params.vectors.size}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create Jina-v3 test collection")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to vectorize (default: 5000)")
    parser.add_argument("--collection", type=str, default="code_samples_jina_test", help="Collection name")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PHASE 1: CREATE JINA-V3 TEST COLLECTION")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Samples: {args.samples}")
    logger.info(f"  Collection: {args.collection}")
    logger.info(f"  Model: jinaai/jina-embeddings-v3")
    logger.info(f"  Dimension: 768 (Matryoshka)")
    logger.info(f"  Task adapter: retrieval.v2")
    logger.info(f"  Max length: 2048 tokens (CPU optimized)")

    # Initialize services
    embedding_service = create_jina_embedding_service()

    # Connect to Qdrant
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key if qdrant_api_key else None,
        https=False  # Local Qdrant uses HTTP
    )

    logger.info(f"\n✓ Connected to Qdrant: {qdrant_host}:{qdrant_port}")

    # Create collection
    collection_name = create_test_collection(client, args.collection)

    # Fetch training samples
    training_samples = fetch_training_samples(args.samples)

    # Vectorize and upload
    vectorize_and_upload(embedding_service, client, collection_name, training_samples)

    logger.info("\n" + "="*80)
    logger.info("PHASE 1 COMPLETE")
    logger.info("="*80)
    logger.info(f"\n✓ Test collection created: {args.collection}")
    logger.info(f"✓ Total samples: {len(training_samples)}")
    logger.info(f"\nNext step:")
    logger.info(f"  python scripts/test_rag_with_new_samples.py \\")
    logger.info(f"    --collection {args.collection} \\")
    logger.info(f"    --k 10 \\")
    logger.info(f"    --strategy majority_vote \\")
    logger.info(f"    --no-features")
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
