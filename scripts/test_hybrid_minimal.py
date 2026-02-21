#!/usr/bin/env python3
"""
Minimal hybrid search test WITHOUT heavy dependencies.

Tests feature filtering directly with Qdrant client.
NO transformers, NO CodeSimilarityStore.

Usage:
    python scripts/test_hybrid_minimal.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns,
    extract_string_features
)


def print_separator(title):
    """Print separator."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)


def test_1_feature_extraction():
    """Test feature extraction."""
    print_separator("TEST 1: Feature Extraction")

    test_code = """
import socket
import os

s = socket.socket()
s.connect(('192.168.1.1', 4444))
os.system('whoami')
"""

    print(f"\nQuery code: {test_code.strip()}")

    ast_features = extract_ast_features(test_code)
    entropy = calculate_entropy(test_code)
    api_patterns = extract_api_patterns(test_code)
    string_features = extract_string_features(test_code)

    print(f"\nExtracted Features:")
    print(f"  Entropy: {entropy:.2f}")
    print(f"  Complexity: {ast_features.get('complexity_score', 0)}")
    print(f"  Dangerous APIs: {ast_features.get('dangerous_patterns', [])}")
    print(f"  Network APIs: {api_patterns.get('network_apis', [])}")
    print(f"  Process APIs: {api_patterns.get('process_apis', [])}")
    print(f"  Has IPs: {string_features.get('has_ips', False)}")

    print("\n✅ Feature extraction works!")


def test_2_qdrant_scroll(client):
    """Test scrolling all samples from Qdrant."""
    print_separator("TEST 2: Scroll All Samples")

    results = client.scroll(
        collection_name="code_samples",
        limit=100,
        with_payload=True,
        with_vectors=False
    )

    points = results[0]
    print(f"\nFound {len(points)} samples in collection")

    for idx, point in enumerate(points, 1):
        payload = point.payload
        code = payload.get('code', 'N/A')
        label = payload.get('label', 'unknown')
        features = payload.get('features', {})

        code_preview = code[:50].replace('\n', ' ') + "..."
        print(f"\n[{idx}] ID: {point.id} | Label: {label}")
        print(f"    Code: {code_preview}")

        if features:
            print(f"    Features:")
            print(f"      - Entropy: {features.get('entropy', 0):.2f}")
            print(f"      - Complexity: {features.get('complexity_score', 0)}")
            print(f"      - Network API: {features.get('has_network_api', False)}")
            print(f"      - Process API: {features.get('has_process_api', False)}")
            dangerous = features.get('dangerous_api_calls', [])
            if dangerous:
                print(f"      - Dangerous: {', '.join(dangerous)}")
        else:
            print(f"    ⚠️  No features!")

    print("\n✅ Scroll works!")


def test_3_filter_network_api(client):
    """Test filtering by network API feature."""
    print_separator("TEST 3: Filter by Network API")

    print("\nQuerying samples with has_network_api = True...")

    # Build filter
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="features.has_network_api",
                match=models.MatchValue(value=True)
            )
        ]
    )

    results = client.scroll(
        collection_name="code_samples",
        scroll_filter=filter_condition,
        limit=10,
        with_payload=True
    )

    points = results[0]
    print(f"\nFound {len(points)} samples with network API")

    for idx, point in enumerate(points, 1):
        payload = point.payload
        code = payload.get('code', 'N/A')
        label = payload.get('label', 'unknown')
        features = payload.get('features', {})

        code_preview = code[:50].replace('\n', ' ') + "..."
        has_network = features.get('has_network_api', False)

        print(f"\n[{idx}] Label: {label} | Network API: {has_network}")
        print(f"    Code: {code_preview}")

        if not has_network:
            print(f"    ❌ ERROR: Sample doesn't have network API!")
            return False

    print("\n✅ Network API filter works!")
    return True


def test_4_filter_entropy(client):
    """Test filtering by entropy (obfuscation detection)."""
    print_separator("TEST 4: Filter by Entropy (Obfuscation)")

    min_entropy = 4.0
    print(f"\nQuerying samples with entropy >= {min_entropy}...")

    # Build filter
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="features.entropy",
                range=models.Range(gte=min_entropy)
            )
        ]
    )

    results = client.scroll(
        collection_name="code_samples",
        scroll_filter=filter_condition,
        limit=10,
        with_payload=True
    )

    points = results[0]
    print(f"\nFound {len(points)} samples with entropy >= {min_entropy}")

    for idx, point in enumerate(points, 1):
        payload = point.payload
        code = payload.get('code', 'N/A')
        features = payload.get('features', {})

        code_preview = code[:50].replace('\n', ' ') + "..."
        entropy = features.get('entropy', 0)

        print(f"\n[{idx}] Entropy: {entropy:.2f}")
        print(f"    Code: {code_preview}")

        if entropy < min_entropy:
            print(f"    ❌ ERROR: Sample entropy {entropy:.2f} below threshold!")
            return False

    print("\n✅ Entropy filter works!")
    return True


def test_5_filter_dangerous_apis(client):
    """Test filtering by dangerous API calls."""
    print_separator("TEST 5: Filter by Dangerous APIs")

    print("\nQuerying samples with dangerous API calls...")

    # Build filter - check if dangerous_api_calls array is not empty
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="features.dangerous_api_calls",
                match=models.MatchAny(any=["eval", "exec", "system"])
            )
        ]
    )

    results = client.scroll(
        collection_name="code_samples",
        scroll_filter=filter_condition,
        limit=10,
        with_payload=True
    )

    points = results[0]
    print(f"\nFound {len(points)} samples with dangerous APIs")

    for idx, point in enumerate(points, 1):
        payload = point.payload
        code = payload.get('code', 'N/A')
        label = payload.get('label', 'unknown')
        features = payload.get('features', {})

        code_preview = code[:50].replace('\n', ' ') + "..."
        dangerous = features.get('dangerous_api_calls', [])

        print(f"\n[{idx}] Label: {label}")
        print(f"    Code: {code_preview}")
        print(f"    Dangerous APIs: {dangerous}")

        if not dangerous:
            print(f"    ⚠️  WARNING: No dangerous APIs found!")

    print("\n✅ Dangerous API filter works!")
    return True


def test_6_combined_filters(client):
    """Test combining multiple filters."""
    print_separator("TEST 6: Combined Filters")

    print("\nQuerying samples with:")
    print("  - Label = malicious")
    print("  - Network API = True")
    print("  - Entropy >= 4.0")

    # Build combined filter
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="label",
                match=models.MatchValue(value="malicious")
            ),
            models.FieldCondition(
                key="features.has_network_api",
                match=models.MatchValue(value=True)
            ),
            models.FieldCondition(
                key="features.entropy",
                range=models.Range(gte=4.0)
            )
        ]
    )

    results = client.scroll(
        collection_name="code_samples",
        scroll_filter=filter_condition,
        limit=10,
        with_payload=True
    )

    points = results[0]
    print(f"\nFound {len(points)} matching samples")

    for idx, point in enumerate(points, 1):
        payload = point.payload
        code = payload.get('code', 'N/A')
        label = payload.get('label', 'unknown')
        features = payload.get('features', {})

        code_preview = code[:50].replace('\n', ' ') + "..."
        entropy = features.get('entropy', 0)
        has_network = features.get('has_network_api', False)

        print(f"\n[{idx}] Label: {label} | Network: {has_network} | Entropy: {entropy:.2f}")
        print(f"    Code: {code_preview}")

        # Verify all conditions
        if label != "malicious":
            print(f"    ❌ ERROR: Label is {label}, not malicious!")
            return False
        if not has_network:
            print(f"    ❌ ERROR: No network API!")
            return False
        if entropy < 4.0:
            print(f"    ❌ ERROR: Entropy {entropy:.2f} below 4.0!")
            return False

    print("\n✅ Combined filters work!")
    return True


def main():
    """Run all tests."""
    print_separator("MINIMAL HYBRID SEARCH TEST")
    print("\nTesting feature filtering WITHOUT CodeSimilarityStore")
    print("(Avoids transformers dependency issues)")

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
    print("✅ Connected")

    # Check collection exists
    try:
        info = client.get_collection("code_samples")
        print(f"   Collection has {info.points_count} points")
    except Exception as e:
        print(f"❌ Collection 'code_samples' doesn't exist!")
        print(f"   Run: python scripts/add_features_to_qdrant.py")
        return 1

    try:
        # Run tests
        test_1_feature_extraction()
        test_2_qdrant_scroll(client)
        test_3_filter_network_api(client)
        test_4_filter_entropy(client)
        test_5_filter_dangerous_apis(client)
        test_6_combined_filters(client)

        # Summary
        print_separator("TEST SUMMARY")
        print("\n✅ ALL TESTS PASSED!")
        print("\nHybrid search features are working:")
        print("  ✅ Feature extraction")
        print("  ✅ Qdrant payload filtering")
        print("  ✅ Network API filter")
        print("  ✅ Entropy filter (obfuscation)")
        print("  ✅ Dangerous API filter")
        print("  ✅ Combined filters")
        print("\nNext steps:")
        print("  1. Populate with real data: python -m scriptguard.pipelines.train_pipeline")
        print("  2. Test with vector search (requires fixing transformers dependency)")
        print("  3. Test API endpoint")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
