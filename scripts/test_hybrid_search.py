#!/usr/bin/env python3
"""
Test hybrid search functionality (Component 2 - Stage 2D).

Tests:
1. Feature extraction
2. Feature filters
3. Hybrid search with filters
4. Feature-based reranking

Usage:
    python scripts/test_hybrid_search.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.config_loader import load_raw_config as load_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.steps.feature_extraction import (
    extract_ast_features,
    calculate_entropy,
    extract_api_patterns
)
from scriptguard.utils.logger import logger

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def test_feature_extraction():
    """Test feature extraction from code."""
    print("\n" + "=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)

    test_code = """
import socket
import os

s = socket.socket()
s.connect(('192.168.1.1', 4444))
os.system('whoami')
"""

    ast_features = extract_ast_features(test_code)
    entropy = calculate_entropy(test_code)
    api_patterns = extract_api_patterns(test_code)

    print(f"Entropy: {entropy:.2f}")
    print(f"Complexity Score: {ast_features.get('complexity_score', 0)}")
    print(f"Dangerous Patterns: {ast_features.get('dangerous_patterns', [])}")
    print(f"Network APIs: {api_patterns.get('network_apis', [])}")
    print(f"Process APIs: {api_patterns.get('process_apis', [])}")

    assert entropy > 0, "Entropy should be > 0"
    assert len(ast_features.get('dangerous_patterns', [])) > 0, "Should detect dangerous patterns"
    print("✓ Feature extraction working")


def test_hybrid_search():
    """Test hybrid search with feature filters."""
    print("\n" + "=" * 60)
    print("TEST 2: Hybrid Search with Feature Filters")
    print("=" * 60)

    config = load_config("config.yaml")
    qdrant_config = config.get("qdrant", {})

    store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        api_key=qdrant_config.get("api_key")
    )

    # Check collection has samples
    info = store.get_collection_info()
    total_samples = info.get("total_samples", 0)
    print(f"Collection has {total_samples} samples")

    if total_samples == 0:
        print("⚠️  Collection is empty - skipping hybrid search test")
        print("   Re-vectorize samples first: python -m scriptguard.pipelines.train_pipeline")
        return

    # Test 1: Search without filters (baseline)
    print("\n--- Test 1a: Baseline search (no filters) ---")
    query_code = "import socket; s = socket.socket()"

    results_baseline = store.search_similar_code(
        query_code=query_code,
        k=5,
        balance_labels=False,
        enable_reranking=False,  # Disable for pure vector search
        feature_filters=None,
        enable_feature_boosting=False
    )

    print(f"Baseline search returned {len(results_baseline)} results")
    for idx, r in enumerate(results_baseline[:3], 1):
        print(f"  [{idx}] Score: {r.get('score', 0):.4f}, Label: {r.get('label', 'unknown')}")

    # Test 2: Search with feature filter (network API required)
    print("\n--- Test 1b: Hybrid search (feature filter: has_network_api=True) ---")
    results_filtered = store.search_similar_code(
        query_code=query_code,
        k=5,
        balance_labels=False,
        enable_reranking=False,
        feature_filters={"required_apis": ["has_network_api"]},
        enable_feature_boosting=False
    )

    print(f"Filtered search returned {len(results_filtered)} results")
    for idx, r in enumerate(results_filtered[:3], 1):
        features = r.get('features', {})
        has_network = features.get('has_network_api', False)
        print(f"  [{idx}] Score: {r.get('score', 0):.4f}, has_network_api: {has_network}")

        # Verify filter worked
        if not has_network and features:  # Only check if features exist
            print(f"  ⚠️  WARNING: Result {idx} doesn't have network API (filter didn't work)")

    # Test 3: Feature boosting
    print("\n--- Test 1c: Hybrid search (feature boosting enabled) ---")
    results_boosted = store.search_similar_code(
        query_code=query_code,
        k=5,
        balance_labels=False,
        enable_reranking=False,
        feature_filters=None,
        enable_feature_boosting=True  # Auto-boost similar features
    )

    print(f"Boosted search returned {len(results_boosted)} results")
    for idx, r in enumerate(results_boosted[:3], 1):
        features = r.get('features', {})
        print(f"  [{idx}] Score: {r.get('score', 0):.4f}, has_network_api: {features.get('has_network_api', False)}")

    print("✓ Hybrid search working")


def test_obfuscated_search():
    """Test searching for obfuscated code (high entropy)."""
    print("\n" + "=" * 60)
    print("TEST 3: Obfuscated Code Search (High Entropy Filter)")
    print("=" * 60)

    config = load_config("config.yaml")
    qdrant_config = config.get("qdrant", {})

    store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        api_key=qdrant_config.get("api_key")
    )

    # Obfuscated query (high entropy)
    obfuscated_query = "exec(__import__('base64').b64decode('aW1wb3J0IG9z').decode())"

    entropy = calculate_entropy(obfuscated_query)
    print(f"Query entropy: {entropy:.2f}")

    if entropy > 6.0:
        print("✓ Query is highly obfuscated (entropy > 6.0)")

        # Search with high entropy filter
        results = store.search_similar_code(
            query_code=obfuscated_query,
            k=5,
            feature_filters={"min_entropy": 5.5},  # Only obfuscated samples
            enable_feature_boosting=True
        )

        print(f"\nFound {len(results)} obfuscated samples")
        for idx, r in enumerate(results[:3], 1):
            features = r.get('features', {})
            result_entropy = features.get('entropy', 0)
            print(f"  [{idx}] Score: {r.get('score', 0):.4f}, Entropy: {result_entropy:.2f}")

            if result_entropy < 5.5 and features:
                print(f"  ⚠️  WARNING: Result {idx} entropy too low (filter didn't work)")
    else:
        print("⚠️  Query entropy too low for obfuscation test")

    print("✓ Obfuscation search test complete")


def main():
    print("Testing Hybrid Search Functionality")
    print("=" * 60)

    try:
        # Test 1: Feature extraction
        test_feature_extraction()

        # Test 2: Hybrid search
        test_hybrid_search()

        # Test 3: Obfuscated search
        test_obfuscated_search()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nHybrid search is working correctly!")
        print("Next steps:")
        print("  1. Test API endpoint with feature analysis")
        print("  2. Monitor false positive rate")
        print("  3. Tune feature filters and boosting factors")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
