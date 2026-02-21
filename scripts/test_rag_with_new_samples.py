#!/usr/bin/env python3
"""
RAG Test with NEW samples (not in database).

Uses Level 3 expansion samples as test set to evaluate RAG on production collection.
These samples are NOT in the database, so it's a real test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from level3_expansion import LEVEL3_BENIGN_EXPANSION, LEVEL3_MALICIOUS_EXPANSION

def print_separator(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 'malicious' and p == 'malicious')
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 'benign' and p == 'benign')
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 'benign' and p == 'malicious')
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 'malicious' and p == 'benign')

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total
    }

def test_rag_with_expansion_samples(k=5, strategy='majority_vote', enable_features=True, collection_name='code_samples'):
    """
    Test RAG with Level 3 expansion samples.

    Args:
        k: Number of neighbors to retrieve
        strategy: 'majority_vote' or 'top1'
        enable_features: Enable feature boosting
        collection_name: Qdrant collection name to use
    """
    print_separator("RAG TEST WITH NEW SAMPLES (Level 3 Expansion)")

    print(f"\nConfiguration:")
    print(f"  Collection: {collection_name}")
    print(f"  K neighbors: {k}")
    print(f"  Strategy: {strategy}")
    print(f"  Feature boosting: {'ENABLED' if enable_features else 'DISABLED (embeddings only)'}")

    # Prepare test set
    test_samples = []

    # Add benign samples
    for sample in LEVEL3_BENIGN_EXPANSION:
        test_samples.append({
            'code': sample['code'],
            'label': 'benign',
            'category': sample['category'],
            'description': sample['description']
        })

    # Add malicious samples
    for sample in LEVEL3_MALICIOUS_EXPANSION:
        test_samples.append({
            'code': sample['code'],
            'label': 'malicious',
            'category': sample['category'],
            'description': sample['description']
        })

    print(f"\nTest set: {len(test_samples)} samples")
    print(f"  Benign: {len(LEVEL3_BENIGN_EXPANSION)}")
    print(f"  Malicious: {len(LEVEL3_MALICIOUS_EXPANSION)}")

    # Initialize RAG store
    print(f"\nInitializing RAG store...")
    store = CodeSimilarityStore(
        host="localhost",
        port=6333,
        collection_name=collection_name
    )
    print(f"[OK] Connected to collection: {collection_name}")

    # Evaluate
    print_separator(f"EVALUATION (k={k}, strategy={strategy})")

    predictions = []
    ground_truth = []
    errors = []
    low_confidence = []

    print(f"\nProcessing {len(test_samples)} queries...")

    for idx, sample in enumerate(test_samples):
        query_code = sample['code']
        true_label = sample['label']

        # Search similar code with FEATURE BOOSTING
        try:
            results = store.search_similar_code(
                query_code=query_code,
                k=k,
                filter_label=None,  # Get both labels
                balance_labels=False,  # CRITICAL: Disable forced label balancing
                enable_feature_boosting=enable_features  # Use features for hybrid search
            )
        except Exception as e:
            print(f"  [ERROR] Query {idx+1} ({sample['category']}) failed: {e}")
            predictions.append('benign')  # Default
            ground_truth.append(true_label)
            continue

        if not results:
            print(f"  [WARN] Query {idx+1} ({sample['category']}): No results")
            predictions.append('benign')
            ground_truth.append(true_label)
            continue

        # Extract labels and scores
        result_labels = [r.get('label', 'unknown') for r in results]
        result_scores = [r.get('score', 0.0) for r in results]

        # Prediction strategy
        if strategy == 'majority_vote':
            malicious_votes = sum(1 for l in result_labels if l == 'malicious')
            benign_votes = sum(1 for l in result_labels if l == 'benign')

            predicted_label = 'malicious' if malicious_votes > benign_votes else 'benign'
            confidence = max(malicious_votes, benign_votes) / len(result_labels)
        else:  # top1
            predicted_label = result_labels[0] if result_labels else 'benign'
            confidence = result_scores[0] if result_scores else 0.0

        predictions.append(predicted_label)
        ground_truth.append(true_label)

        # Track errors
        if predicted_label != true_label:
            errors.append({
                'category': sample['category'],
                'description': sample['description'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'top_scores': result_scores[:3],
                'top_labels': result_labels[:3],
                'code_preview': query_code[:200]
            })

        # Track low confidence predictions
        if confidence < 0.6:
            low_confidence.append({
                'category': sample['category'],
                'description': sample['description'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': predicted_label == true_label
            })

        # Progress
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(test_samples)}...")

    print(f"\n[OK] Evaluated {len(predictions)} queries")

    # Calculate metrics
    print_separator("RESULTS")

    metrics = calculate_metrics(ground_truth, predictions)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1']:.2%}")
    print(f"  FPR:       {metrics['fpr']:.2%}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']:4d} (malicious correctly identified)")
    print(f"  True Negatives:  {metrics['tn']:4d} (benign correctly identified)")
    print(f"  False Positives: {metrics['fp']:4d} (benign marked as malicious)")
    print(f"  False Negatives: {metrics['fn']:4d} (malicious marked as benign)")

    # Error analysis
    if errors:
        print_separator(f"ERROR ANALYSIS ({len(errors)}/{len(test_samples)} errors)")

        false_positives = [e for e in errors if e['true_label'] == 'benign']
        false_negatives = [e for e in errors if e['true_label'] == 'malicious']

        print(f"\nFalse Positives: {len(false_positives)} (benign -> malicious)")
        for i, err in enumerate(false_positives[:10], 1):
            print(f"\n  {i}. [{err['category']}] {err['description']}")
            print(f"     Confidence: {err['confidence']:.2%}")
            print(f"     Top-3 results: {list(zip(err['top_labels'], [f'{s:.3f}' for s in err['top_scores']]))}")
            print(f"     Code: {err['code_preview']}...")

        print(f"\nFalse Negatives: {len(false_negatives)} (malicious -> benign)")
        for i, err in enumerate(false_negatives[:10], 1):
            print(f"\n  {i}. [{err['category']}] {err['description']}")
            print(f"     Confidence: {err['confidence']:.2%}")
            print(f"     Top-3 results: {list(zip(err['top_labels'], [f'{s:.3f}' for s in err['top_scores']]))}")
            print(f"     Code: {err['code_preview']}...")

    # Low confidence analysis
    if low_confidence:
        print_separator(f"LOW CONFIDENCE PREDICTIONS ({len(low_confidence)})")

        print(f"\nPredictions with confidence < 60%:")
        for i, pred in enumerate(low_confidence[:10], 1):
            status = "[OK]" if pred['correct'] else "[FAIL]"
            print(f"\n  {i}. {status} [{pred['category']}] {pred['description']}")
            print(f"     True: {pred['true_label']}, Predicted: {pred['predicted_label']}, Confidence: {pred['confidence']:.2%}")

    # Category breakdown
    print_separator("CATEGORY BREAKDOWN")

    from collections import defaultdict
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})

    for sample, pred, truth in zip(test_samples, predictions, ground_truth):
        cat = sample['category']
        category_stats[cat]['total'] += 1
        if pred == truth:
            category_stats[cat]['correct'] += 1
        else:
            category_stats[cat]['errors'].append({
                'description': sample['description'],
                'true': truth,
                'pred': pred
            })

    print(f"\nAccuracy by category:")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        status = "[OK]" if accuracy >= 0.8 else "[FAIL]"
        print(f"  {status} {cat:30s}: {stats['correct']:2d}/{stats['total']:2d} ({accuracy:.1%})")

    # Summary
    print_separator("SUMMARY")

    print(f"\nRAG Performance on NEW Data:")
    print(f"  Test samples: {len(test_samples)} (NOT in database)")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    print(f"  Error rate: {len(errors)/len(test_samples):.2%}")

    if metrics['f1'] >= 0.90:
        print(f"\n[PASS] EXCELLENT: F1 >= 90% - RAG works very well!")
    elif metrics['f1'] >= 0.80:
        print(f"\n[OK] GOOD: F1 >= 80% - RAG works well")
    elif metrics['f1'] >= 0.70:
        print(f"\n[WARN]  MODERATE: F1 >= 70% - RAG needs improvement")
    else:
        print(f"\n[POOR] POOR: F1 < 70% - RAG needs significant improvement")

    print("\nThis test uses samples NOT in the database, so it's a real evaluation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG with new samples")
    parser.add_argument("--collection", type=str, default="code_samples",
                        help="Qdrant collection name (default: code_samples)")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors (default: 10, optimal)")
    parser.add_argument("--strategy", choices=['majority_vote', 'top1'], default='majority_vote',
                        help="Prediction strategy (default: majority_vote)")
    parser.add_argument("--no-features", action="store_true",
                        help="Disable feature boosting (test embeddings only)")

    args = parser.parse_args()

    enable_features = not args.no_features  # Default: True (features enabled)

    test_rag_with_expansion_samples(
        k=args.k,
        strategy=args.strategy,
        enable_features=enable_features,
        collection_name=args.collection
    )
