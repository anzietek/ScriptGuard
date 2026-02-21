#!/usr/bin/env python3
"""
Force re-index all samples with features.

This script bypasses ZenML pipeline and directly:
1. Loads samples from PostgreSQL
2. Extracts features for each sample
3. Re-vectorizes and uploads to Qdrant with features

Usage:
    python scripts/force_reindex_with_features.py --limit 100  # Test with 100 samples
    python scripts/force_reindex_with_features.py              # All samples
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from scriptguard.config_loader import load_raw_config
from scriptguard.database.postgres_manager import PostgresManager
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns,
    extract_string_features
)
from scriptguard.utils.logger import logger


def extract_features_for_sample(code: str) -> dict:
    """Extract all features for a code sample."""
    try:
        ast_features = extract_ast_features(code)
        entropy = calculate_entropy(code)
        api_patterns = extract_api_patterns(code)
        string_features = extract_string_features(code)

        features = {
            # Complexity metrics
            "complexity_score": ast_features.get("complexity_score", 0),
            "entropy": entropy,
            "code_length": len(code),
            "code_lines": code.count("\n") + 1,

            # Dangerous patterns
            "dangerous_api_calls": ast_features.get("dangerous_patterns", []),
            "suspicious_combinations": api_patterns.get("suspicious_combinations", []),

            # API usage flags
            "has_network_api": len(api_patterns.get("network_apis", [])) > 0,
            "has_file_api": len(api_patterns.get("file_apis", [])) > 0,
            "has_process_api": len(api_patterns.get("process_apis", [])) > 0,
            "has_crypto_api": len(api_patterns.get("crypto_apis", [])) > 0,

            # String patterns
            "has_urls": string_features.get("has_urls", False),
            "has_ips": string_features.get("has_ips", False),
            "has_base64": string_features.get("has_base64", False),
            "has_hex": string_features.get("has_hex", False),

            # Detailed arrays (for analysis, not filtering)
            "network_apis": api_patterns.get("network_apis", []),
            "file_apis": api_patterns.get("file_apis", []),
            "process_apis": api_patterns.get("process_apis", []),
            "crypto_apis": api_patterns.get("crypto_apis", []),
            "imports": ast_features.get("imports", []),
            "function_calls": ast_features.get("function_calls", []),
            "suspicious_strings": string_features.get("suspicious_strings", [])
        }

        return features

    except Exception as e:
        logger.warning(f"Failed to extract features: {e}")
        # Return empty features on error
        return {
            "complexity_score": 0,
            "entropy": 0.0,
            "code_length": len(code),
            "code_lines": code.count("\n") + 1,
            "dangerous_api_calls": [],
            "suspicious_combinations": [],
            "has_network_api": False,
            "has_file_api": False,
            "has_process_api": False,
            "has_crypto_api": False,
            "has_urls": False,
            "has_ips": False,
            "has_base64": False,
            "has_hex": False,
            "network_apis": [],
            "file_apis": [],
            "process_apis": [],
            "crypto_apis": [],
            "imports": [],
            "function_calls": [],
            "suspicious_strings": []
        }


def main():
    parser = argparse.ArgumentParser(description="Force re-index with features")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection first")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")

    args = parser.parse_args()

    config = load_raw_config("config.yaml")

    print("=" * 70)
    print("FORCE RE-INDEX WITH FEATURES")
    print("=" * 70)

    # Load samples from PostgreSQL
    print("\n1. Loading samples from PostgreSQL...")
    db = PostgresManager(config)

    with db.get_connection() as conn:
        query = """
            SELECT id, content, label, source
            FROM code_samples
            WHERE content IS NOT NULL
            AND LENGTH(content) > 50
        """

        if args.limit:
            query += f" LIMIT {args.limit}"

        result = conn.execute(query).fetchall()

    samples = [
        {
            "id": r[0],
            "content": r[1],
            "label": r[2],
            "source": r[3]
        }
        for r in result
    ]

    print(f"✓ Loaded {len(samples)} samples from PostgreSQL")

    # Extract features
    print("\n2. Extracting features for all samples...")
    samples_with_features = []
    failed_count = 0

    for i, sample in enumerate(samples, 1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")

        code = sample["content"]
        features = extract_features_for_sample(code)

        # Add features to sample
        sample_with_features = {
            **sample,
            "features": features
        }

        samples_with_features.append(sample_with_features)

        # Verify features were extracted
        if features.get("entropy", 0) == 0 and features.get("complexity_score", 0) == 0:
            failed_count += 1

    print(f"✓ Feature extraction complete")
    print(f"  Successful: {len(samples_with_features) - failed_count}")
    print(f"  Failed: {failed_count}")

    # Initialize CodeSimilarityStore
    print("\n3. Initializing Qdrant connection...")
    qdrant_config = config.get("qdrant", {})
    embedding_config = config.get("code_embedding", {})

    store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        embedding_model=embedding_config.get("model", "microsoft/unixcoder-base"),
        api_key=qdrant_config.get("api_key"),
        use_https=qdrant_config.get("use_https", False),
        enable_chunking=embedding_config.get("enable_chunking", True)
    )

    print(f"✓ Connected to Qdrant")

    # Clear collection if requested
    if args.clear:
        print("\n4. Clearing existing collection...")
        store.clear_collection()
        print("✓ Collection cleared")

    # Vectorize and upload
    print(f"\n5. Vectorizing {len(samples_with_features)} samples with features...")
    print(f"   Batch size: {args.batch_size}")

    store.upsert_code_samples(samples_with_features, batch_size=args.batch_size)

    print("✓ Re-indexing complete!")

    # Verify features were stored
    print("\n6. Verifying features in Qdrant...")
    from qdrant_client import QdrantClient

    client_kwargs = {
        "host": "localhost",
        "port": 6333,
        "https": False,
        "timeout": 60
    }

    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)

    results = client.scroll(collection_name="code_samples", limit=5)

    samples_with_features_count = 0
    for point in results[0]:
        if point.payload.get('features'):
            samples_with_features_count += 1

    print(f"✓ Verified: {samples_with_features_count}/5 samples have features")

    if samples_with_features_count == 0:
        print("\n❌ ERROR: Features were NOT stored in Qdrant!")
        print("   Check vectorize_samples.py and code_similarity_store.py")
        return 1
    elif samples_with_features_count < 5:
        print("\n⚠️  WARNING: Some samples missing features")
        return 1
    else:
        print("\n✅ SUCCESS: All sampled points have features!")
        print("\nNext steps:")
        print("  1. Run: python scripts/analyze_features.py")
        print("  2. Verify 100% coverage")
        return 0


if __name__ == "__main__":
    sys.exit(main())
