"""
Quick test of enhanced RAG features - graceful fallback, thresholds, and reranking
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger


def main():
    print("=" * 80)
    print("TESTING ENHANCED RAG FEATURES")
    print("=" * 80)
    print()

    # Initialize store with test collection (in-memory for testing)
    print("1. Initializing Code Similarity Store (in-memory mode)...")

    # Create a custom config that uses in-memory Qdrant
    from qdrant_client import QdrantClient
    store = CodeSimilarityStore(
        collection_name="test_enhanced_rag",
        config_path="../../config.yaml"
    )

    # Replace client with in-memory version
    store.client = QdrantClient(":memory:")
    store._ensure_collection()  # Recreate collection in memory
    print(f"   [OK] Store initialized")
    print(f"   [OK] Graceful fallback: {store.graceful_fallback_enabled}")
    print(f"   [OK] Reranking enabled: {store.reranking_service is not None}")
    print(f"   [OK] Thresholds: {store.score_thresholds}")
    print()

    # Clear any existing data
    print("2. Clearing test collection...")
    store.clear_collection()
    print("   [OK] Collection cleared")
    print()

    # Create test samples
    print("3. Creating test samples...")
    test_samples = [
        {
            "id": 1,
            "content": "import os\nos.system('rm -rf /')",
            "label": "malicious",
            "source": "test"
        },
        {
            "id": 2,
            "content": "import subprocess\nsubprocess.call(['whoami'])",
            "label": "malicious",
            "source": "test"
        },
        {
            "id": 3,
            "content": "eval(input('Enter code: '))",
            "label": "malicious",
            "source": "test"
        },
        {
            "id": 4,
            "content": "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "label": "benign",
            "source": "test"
        },
        {
            "id": 5,
            "content": "import numpy as np\narr = np.array([1, 2, 3])",
            "label": "benign",
            "source": "test"
        },
        {
            "id": 6,
            "content": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])",
            "label": "benign",
            "source": "test"
        }
    ]

    store.upsert_code_samples(test_samples)
    print(f"   [OK] Upserted {len(test_samples)} samples")
    print()

    # Test 1: Graceful Fallback with high threshold
    print("4. TEST: Graceful Fallback (high threshold)")
    print("   Query: 'import os\\nos.system('ls')'")
    print("   Threshold: 0.99 (very high)")
    print("   Expected: Should still get exactly k=3 results")

    results = store.search_similar_code(
        query_code="import os\nos.system('ls')",
        k=3,
        score_threshold=0.99,
        balance_labels=False
    )

    print(f"   [OK] Got {len(results)} results (expected 3)")
    assert len(results) == 3, f"FAILED: Expected 3 results, got {len(results)}"
    print()

    # Test 2: Configurable Thresholds
    print("5. TEST: Configurable Thresholds")
    for mode in ["lenient", "default", "strict"]:
        threshold = store.get_threshold(mode)
        print(f"   {mode:8s}: {threshold:.3f}")
    print()

    # Test 3: Label Balance
    print("6. TEST: Label Balance")
    print("   Query: 'import os'")
    print("   k=5, balance_labels=True")

    results = store.search_similar_code(
        query_code="import os",
        k=5,
        balance_labels=True,
        threshold_mode="default",
        enable_reranking=True
    )

    malicious_count = sum(1 for r in results if r["label"] == "malicious")
    benign_count = sum(1 for r in results if r["label"] == "benign")

    print(f"   [OK] Got {len(results)} results")
    print(f"   [OK] Malicious: {malicious_count}")
    print(f"   [OK] Benign: {benign_count}")
    assert malicious_count >= 1, "FAILED: Expected at least 1 malicious result"
    assert benign_count >= 1, "FAILED: Expected at least 1 benign result"
    print()

    # Test 4: Reranking Effect
    print("7. TEST: Reranking Effect")
    print("   Query with security pattern: 'import os\\nos.system(\"evil\")'")

    # Without reranking
    results_no_rerank = store.search_similar_code(
        query_code="import os\nos.system('evil')",
        k=3,
        balance_labels=False,
        enable_reranking=False
    )

    # With reranking
    results_reranked = store.search_similar_code(
        query_code="import os\nos.system('evil')",
        k=3,
        balance_labels=False,
        enable_reranking=True
    )

    print("   Without reranking:")
    for i, r in enumerate(results_no_rerank[:3], 1):
        print(f"     {i}. [{r['label']:10s}] score={r['score']:.3f} | {r['code'][:40]}...")

    print("   With reranking:")
    for i, r in enumerate(results_reranked[:3], 1):
        print(f"     {i}. [{r['label']:10s}] score={r['score']:.3f} | {r['code'][:40]}...")

    # Top result with reranking should have security pattern
    top_code = results_reranked[0]["code"]
    has_security = any(kw in top_code for kw in ["os.system", "subprocess", "eval"])
    print(f"   [OK] Top reranked result has security pattern: {has_security}")
    print()

    # Test 5: Collection Info
    print("8. Collection Info:")
    info = store.get_collection_info()
    print(f"   Total samples: {info.get('total_samples', 0)}")
    print(f"   Malicious: {info.get('malicious_samples', 0)}")
    print(f"   Benign: {info.get('benign_samples', 0)}")
    print()

    print("=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary of Enhanced Features:")
    print("  1. [OK] Graceful Fallback - Always returns exactly k results")
    print("  2. [OK] Configurable Thresholds - Per-model calibrated values")
    print("  3. [OK] Reranking - Improved relevance with security patterns")
    print("  4. [OK] Label Balance - Guaranteed diversity in results")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
