#!/usr/bin/env python3
"""
Simple test: Load samples from Qdrant, add features, re-upload.

This is a workaround since we don't have direct PostgreSQL access.
We'll:
1. Load existing samples from Qdrant (they have code but no features)
2. Extract features for each
3. Re-upload with features

Usage:
    python scripts/simple_reindex_test.py --limit 100
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from scriptguard.config_loader import load_raw_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns,
    extract_string_features
)


def extract_features_for_sample(code: str) -> dict:
    """Extract all features for a code sample."""
    try:
        ast_features = extract_ast_features(code)
        entropy = calculate_entropy(code)
        api_patterns = extract_api_patterns(code)
        string_features = extract_string_features(code)

        return {
            "complexity_score": ast_features.get("complexity_score", 0),
            "entropy": entropy,
            "code_length": len(code),
            "code_lines": code.count("\n") + 1,
            "dangerous_api_calls": ast_features.get("dangerous_patterns", []),
            "suspicious_combinations": api_patterns.get("suspicious_combinations", []),
            "has_network_api": len(api_patterns.get("network_apis", [])) > 0,
            "has_file_api": len(api_patterns.get("file_apis", [])) > 0,
            "has_process_api": len(api_patterns.get("process_apis", [])) > 0,
            "has_crypto_api": len(api_patterns.get("crypto_apis", [])) > 0,
            "has_urls": string_features.get("has_urls", False),
            "has_ips": string_features.get("has_ips", False),
            "has_base64": string_features.get("has_base64", False),
            "has_hex": string_features.get("has_hex", False),
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
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to process")
    args = parser.parse_args()

    print("=" * 70)
    print("SIMPLE RE-INDEX TEST WITH FEATURES")
    print("=" * 70)

    # Connect to Qdrant
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

    # 1. Load existing samples from Qdrant
    print(f"\n1. Loading {args.limit} samples from Qdrant...")

    samples = []
    offset = None

    while len(samples) < args.limit:
        results = client.scroll(
            collection_name="code_samples",
            limit=min(100, args.limit - len(samples)),
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

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

            if code and len(code) > 50:
                samples.append({
                    "id": point.id,
                    "code": code,
                    "label": payload.get("label", "unknown"),
                    "db_id": payload.get("db_id"),
                    "existing_features": payload.get("features")
                })

        offset = next_offset
        if offset is None:
            break

    print(f"✓ Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("❌ No samples found in Qdrant!")
        print("   Make sure you have data in the 'code_samples' collection")
        return 1

    # Check if they already have features
    with_features = sum(1 for s in samples if s["existing_features"])
    print(f"  Samples with features: {with_features}/{len(samples)}")

    # 2. Extract features
    print(f"\n2. Extracting features...")

    for i, sample in enumerate(samples, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(samples)}...")

        # Extract features
        features = extract_features_for_sample(sample["code"])
        sample["features"] = features

    print(f"✓ Features extracted for {len(samples)} samples")

    # Show example
    print(f"\n3. Example features:")
    example = samples[0]
    print(f"  Code preview: {example['code'][:60]}...")
    print(f"  Entropy: {example['features']['entropy']:.2f}")
    print(f"  Complexity: {example['features']['complexity_score']}")
    print(f"  Dangerous APIs: {example['features']['dangerous_api_calls'][:3]}")
    print(f"  Has network: {example['features']['has_network_api']}")

    # 4. Update points in Qdrant
    print(f"\n4. Updating points in Qdrant with features...")

    from qdrant_client import models

    updated_points = []
    for sample in samples:
        # Reconstruct payload with features
        payload = {
            "label": sample["label"],
            "db_id": sample["db_id"],
            "code_preview": sample["code"][:200],
            "features": sample["features"]  # NEW!
        }

        # We're updating existing points, so we use set_payload
        client.set_payload(
            collection_name="code_samples",
            payload={"features": sample["features"]},
            points=[sample["id"]]
        )

        updated_points.append(sample["id"])

        if len(updated_points) % 20 == 0:
            print(f"  Updated {len(updated_points)}/{len(samples)}...")

    print(f"✓ Updated {len(updated_points)} points with features")

    # 5. Verify
    print(f"\n5. Verifying...")

    verify_results = client.scroll(
        collection_name="code_samples",
        limit=5
    )

    verified_count = 0
    for point in verify_results[0]:
        if point.payload.get("features"):
            verified_count += 1
            features = point.payload["features"]
            print(f"  ✓ Point {point.id}: entropy={features.get('entropy', 0):.2f}")

    print(f"\n{'='*70}")
    if verified_count > 0:
        print(f"✅ SUCCESS: {verified_count}/5 verified samples have features!")
        print(f"\nNow run full update for ALL samples:")
        print(f"  python scripts/simple_reindex_test.py --limit 63000")
    else:
        print(f"❌ FAILURE: Features were not stored!")
    print("=" * 70)

    return 0 if verified_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
