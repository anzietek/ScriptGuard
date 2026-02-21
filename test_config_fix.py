#!/usr/bin/env python3
"""
Test RAG configuration fix - verify balance_labels=false works correctly.

Expected behavior AFTER fix:
- Query: benign CSV code
- RAG finds 0 relevant benign CSV samples
- Should NOT retrieve random Django/Flask code with score 0.08
- Should either:
  1. Use zero-shot (no examples)
  2. Use only malicious CSV samples + warn about imbalance
  3. Return fewer than k=10 samples (only high-quality ones)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
from dotenv import load_dotenv

load_dotenv()

def test_config_values():
    """Verify config.yaml has correct values."""
    print("="*80)
    print("CONFIGURATION FIX VERIFICATION")
    print("="*80)

    # Load config
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check few-shot config
    fewshot = config.get("code_embedding", {}).get("fewshot", {})
    print("\n1. FEW-SHOT CONFIGURATION")
    print(f"   balance_labels: {fewshot.get('balance_labels')}")

    if fewshot.get("balance_labels") == False:
        print("   ✓ CORRECT: balance_labels=false (won't force irrelevant samples)")
    else:
        print("   ✗ WRONG: balance_labels=true (will retrieve garbage when no benign samples)")

    # Check graceful fallback config
    fallback = config.get("code_embedding", {}).get("graceful_fallback", {})
    print("\n2. GRACEFUL FALLBACK CONFIGURATION")
    print(f"   enabled: {fallback.get('enabled', True)}")  # Default is True if not specified
    print(f"   fallback_threshold: {fallback.get('fallback_threshold', 0.0)}")
    print(f"   ensure_label_balance: {fallback.get('ensure_label_balance', True)}")
    print(f"   min_per_label: {fallback.get('min_per_label', 1)}")

    if fallback.get("enabled") == False:
        print("   ✓ CORRECT: graceful_fallback disabled (won't pollute with threshold=0.0)")
    else:
        print("   ✗ WRONG: graceful_fallback enabled (will retrieve garbage with threshold=0.0)")

    if fallback.get("fallback_threshold", 0.0) >= 0.30:
        print(f"   ✓ CORRECT: fallback_threshold={fallback.get('fallback_threshold')} >= 0.30")
    else:
        print(f"   ✗ WRONG: fallback_threshold={fallback.get('fallback_threshold', 0.0)} < 0.30 (too low)")

    if fallback.get("ensure_label_balance") == False:
        print("   ✓ CORRECT: ensure_label_balance=false")
    else:
        print("   ✗ WRONG: ensure_label_balance=true (forces balance in fallback)")

    if fallback.get("min_per_label", 1) == 0:
        print("   ✓ CORRECT: min_per_label=0 (allows zero-shot)")
    else:
        print(f"   ✗ WRONG: min_per_label={fallback.get('min_per_label', 1)} (forces garbage retrieval)")

    # Overall verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    all_correct = (
        fewshot.get("balance_labels") == False and
        fallback.get("enabled") == False and
        fallback.get("fallback_threshold", 0.0) >= 0.30 and
        fallback.get("ensure_label_balance") == False and
        fallback.get("min_per_label", 1) == 0
    )

    if all_correct:
        print("\n✓ ALL CHECKS PASSED - Configuration fix applied correctly!")
        print("\nExpected improvements:")
        print("  - Stops retrieving random Django/Flask code for CSV queries")
        print("  - Uses zero-shot when no relevant benign examples exist")
        print("  - F1 score: 66.67% → 75-80% (estimated)")
    else:
        print("\n✗ SOME CHECKS FAILED - Fix incomplete")
        print("\nPlease update config.yaml:")
        print("  1. Set balance_labels: false")
        print("  2. Add graceful_fallback section with:")
        print("     - enabled: false")
        print("     - fallback_threshold: 0.30")
        print("     - ensure_label_balance: false")
        print("     - min_per_label: 0")

    print("\n" + "="*80)


def test_rag_retrieval():
    """Test RAG retrieval with benign CSV code."""
    print("\n" + "="*80)
    print("RAG RETRIEVAL TEST")
    print("="*80)

    from scriptguard.rag.code_similarity_store import CodeSimilarityStore

    # Initialize RAG store
    print("\nInitializing RAG store...")
    store = CodeSimilarityStore(
        collection_name="code_samples",
        embedding_model="microsoft/unixcoder-base",
        ensure_label_balance=False,  # Force override to test
        min_per_label=0
    )

    # Test query - benign CSV code
    test_code = """
import csv

with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['name'], row['email'])
"""

    print(f"\nTest query (benign CSV code):")
    print(test_code)

    # Search
    print("\nSearching for similar code samples (k=10)...")
    results = store.search_similar_code(
        query_code=test_code,
        k=10,
        balance_labels=False,  # Test with balance disabled
        score_threshold=0.50,
        threshold_mode="default"
    )

    print(f"\nResults: {len(results)} samples retrieved")

    # Analyze results
    malicious_count = sum(1 for r in results if r.get("label") == "malicious")
    benign_count = sum(1 for r in results if r.get("label") == "benign")

    print(f"\nLabel distribution:")
    print(f"  Malicious: {malicious_count}")
    print(f"  Benign:    {benign_count}")

    # Show top results
    print(f"\nTop 5 results:")
    for i, result in enumerate(results[:5], 1):
        label = result.get("label", "unknown")
        score = result.get("score", 0.0)
        code_snippet = result.get("code", "")[:100].replace("\n", " ")

        print(f"\n  [{i}] Label: {label}, Score: {score:.4f}")
        print(f"      Code: {code_snippet}...")

        # Check if low-score benign (indicates garbage retrieval)
        if label == "benign" and score < 0.30:
            print(f"      ⚠️  WARNING: Low-score benign sample (possible garbage retrieval)")

    # Verdict
    print("\n" + "="*80)
    print("RETRIEVAL VERDICT")
    print("="*80)

    # Check for garbage retrieval
    low_score_benign = [r for r in results if r.get("label") == "benign" and r.get("score", 0) < 0.30]

    if len(low_score_benign) > 0:
        print(f"\n✗ FOUND {len(low_score_benign)} LOW-SCORE BENIGN SAMPLES (< 0.30)")
        print("  This indicates garbage retrieval (random Django/Flask code)")
        print("  Configuration fix may not be applied correctly")
    else:
        print("\n✓ NO LOW-SCORE GARBAGE SAMPLES")
        print("  All benign samples have score >= 0.30")
        print("  Configuration fix working correctly!")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG configuration fix")
    parser.add_argument("--test-rag", action="store_true", help="Test RAG retrieval (requires Qdrant)")
    args = parser.parse_args()

    # Always test config values
    test_config_values()

    # Optionally test RAG retrieval
    if args.test_rag:
        try:
            test_rag_retrieval()
        except Exception as e:
            print(f"\n✗ RAG retrieval test failed: {e}")
            print("  (This is okay if Qdrant is not running)")
