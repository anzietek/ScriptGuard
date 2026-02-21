#!/usr/bin/env python3
"""
Test Production API on 99-Sample Diverse Test Set

Tests the ScriptGuard API endpoint with the expanded test set to validate
that config.yaml fixes (balance_labels=false, graceful_fallback=false)
are working correctly in production.

Expected: ~85% accuracy (vs test script's 66% using wrong defaults)
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import requests
from dotenv import load_dotenv

# Add parent directory to path to import level3_expansion
sys.path.insert(0, str(Path(__file__).parent))
from level3_expansion import LEVEL3_BENIGN_EXPANSION, LEVEL3_MALICIOUS_EXPANSION

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/analyze")
API_KEY = os.getenv("SCRIPTGUARD_API_KEY")

if not API_KEY:
    print("[!] ERROR: SCRIPTGUARD_API_KEY not found in environment")
    print("Please set it in .env file or export it")
    sys.exit(1)


def test_api_classification():
    """Test the API on all samples and calculate metrics"""

    # Prepare test samples
    test_samples = []
    for sample in LEVEL3_BENIGN_EXPANSION:
        test_samples.append({**sample, "label": "benign"})
    for sample in LEVEL3_MALICIOUS_EXPANSION:
        test_samples.append({**sample, "label": "malicious"})

    print("=" * 70)
    print(f"  Testing ScriptGuard API on {len(test_samples)} Diverse Samples")
    print("=" * 70 + "\n")

    # Track results
    results = []
    predictions = []
    actuals = []
    category_results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})

    # Test each sample
    for i, sample in enumerate(test_samples, 1):
        try:
            response = requests.post(
                API_URL,
                headers={"X-API-Key": API_KEY},
                json={"script_content": sample["code"], "include_rag": True},
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            prediction = "malicious" if result["is_malicious"] else "benign"
            confidence = result.get("confidence", 0.0)

        except Exception as e:
            print(f"  [{i:3}/{len(test_samples)}] [!] ERROR: {sample['category']:20s} - {str(e)[:50]}")
            continue

        actual = sample["label"]
        correct = prediction == actual

        # Track results
        results.append(correct)
        predictions.append(1 if prediction == "malicious" else 0)
        actuals.append(1 if actual == "malicious" else 0)

        # Track by category
        category = sample["category"]
        category_results[category]["total"] += 1
        category_results[category]["predictions"].append({
            "predicted": prediction,
            "actual": actual,
            "correct": correct,
            "confidence": confidence
        })
        if correct:
            category_results[category]["correct"] += 1

        # Display progress
        status = "[+]" if correct else "[-]"
        print(f"  [{i:3}/{len(test_samples)}] {status} {category:20s} Pred: {prediction:10s} Actual: {actual:10s} (conf: {confidence:.3f})")

    # Calculate overall metrics
    if not results:
        print("\n[!] No results to analyze!")
        return

    accuracy = sum(results) / len(results)

    # Calculate F1 score
    true_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
    false_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
    false_negatives = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print overall results
    print("\n" + "=" * 70)
    print("                     OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total Samples:      {len(test_samples)}")
    print(f"  Correct:            {sum(results)}")
    print(f"  Incorrect:          {len(results) - sum(results)}")
    print(f"  Accuracy:           {accuracy:.2%}")
    print(f"  Precision:          {precision:.2%}")
    print(f"  Recall:             {recall:.2%}")
    print(f"  F1 Score:           {f1_score:.2%}")

    # Print per-category breakdown
    print("\n" + "=" * 70)
    print("                   PER-CATEGORY BREAKDOWN")
    print("=" * 70)

    # Sort categories by accuracy (worst first)
    sorted_categories = sorted(
        category_results.items(),
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
    )

    for category, stats in sorted_categories:
        cat_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        # Determine label (look at actual labels)
        label = stats["predictions"][0]["actual"] if stats["predictions"] else "unknown"

        # Color code by performance
        if cat_accuracy >= 0.8:
            indicator = "[+]"
        elif cat_accuracy >= 0.5:
            indicator = "[~]"
        else:
            indicator = "[-]"

        print(f"  {indicator} {category:20s} ({label:10s}): {stats['correct']:2}/{stats['total']:2} = {cat_accuracy:6.1%}")

    # Highlight problem categories
    print("\n" + "=" * 70)
    print("                    PROBLEM CATEGORIES")
    print("=" * 70)

    problem_categories = [
        (cat, stats) for cat, stats in sorted_categories
        if (stats["correct"] / stats["total"]) < 0.5
    ]

    if problem_categories:
        for category, stats in problem_categories:
            cat_accuracy = stats["correct"] / stats["total"]
            label = stats["predictions"][0]["actual"]
            print(f"  [!] {category:20s} ({label:10s}): {cat_accuracy:.1%} accuracy")

            # Show first failed example
            for pred in stats["predictions"]:
                if not pred["correct"]:
                    print(f"      -> Predicted {pred['predicted']}, actually {pred['actual']} (conf: {pred['confidence']:.3f})")
                    break
    else:
        print("  [+] No categories with <50% accuracy!")

    # Comparison with expected results
    print("\n" + "=" * 70)
    print("                  COMPARISON WITH BASELINE")
    print("=" * 70)
    print(f"  Test Script Baseline (k=5, balance=true):  66.67% accuracy")
    print(f"  Production API (this test):                 {accuracy:.2%} accuracy")
    print(f"  Expected (k=10, balance=false):             ~85% accuracy")
    print()

    if accuracy >= 0.80:
        print("  [+] SUCCESS! Config fixes are working in production API!")
        print("  The problem is only in the test script using wrong defaults.")
    elif accuracy >= 0.70:
        print("  [~] PARTIAL IMPROVEMENT - Better than baseline but not optimal")
        print("  May need to restart API to pick up config.yaml changes")
    else:
        print("  [!] NO IMPROVEMENT - API still showing poor performance")
        print("  Investigate why config changes are not taking effect")

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "category_results": dict(category_results)
    }


if __name__ == "__main__":
    try:
        results = test_api_classification()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
