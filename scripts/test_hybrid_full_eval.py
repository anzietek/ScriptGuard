#!/usr/bin/env python3
"""
Full evaluation: Test all 40 samples with leave-one-out.

For each sample:
1. Use it as query
2. Search remaining 39 samples
3. Majority vote from top 3
4. Compare prediction vs ground truth

Metrics:
- Feature-only classification
- Vector-only search
- Hybrid (60% vector + 40% features)

Usage:
    python scripts/test_hybrid_full_eval.py [--use-existing]

    --use-existing: Use existing collection instead of recreating
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Import from the balanced test
sys.path.insert(0, str(Path(__file__).parent))
from test_hybrid_balanced import (
    BENIGN_SAMPLES,
    MALICIOUS_SAMPLES,
    extract_features_for_sample,
    calculate_feature_similarity
)


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)


def calculate_metrics(predictions, ground_truth):
    """Calculate classification metrics."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 'malicious' and g == 'malicious')
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 'malicious' and g == 'benign')
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 'benign' and g == 'benign')
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 'benign' and g == 'malicious')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'accuracy': accuracy
    }


def full_evaluation_vector_only(client, collection, tokenizer, model, device):
    """Full evaluation with vector-only search."""
    print_separator("FULL EVALUATION: Vector-Only Search")

    # Get all samples
    all_results = client.scroll(collection_name=collection, limit=100, with_payload=True)
    all_points = all_results[0]

    print(f"\nEvaluating {len(all_points)} samples (leave-one-out)...")

    predictions = []
    ground_truth = []
    errors = []

    for idx, query_point in enumerate(all_points):
        query_payload = query_point.payload
        query_code = query_payload['code']
        query_label = query_payload['label']
        query_id = query_point.id

        # Embed query
        with torch.no_grad():
            inputs = tokenizer([query_code], padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            query_vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        # Search (exclude self)
        results = client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=4  # Get top 4, exclude self
        ).points

        # Filter out self
        results = [r for r in results if r.id != query_id][:3]

        # Majority vote
        top_labels = [r.payload['label'] for r in results]
        prediction = max(set(top_labels), key=top_labels.count) if top_labels else 'benign'

        predictions.append(prediction)
        ground_truth.append(query_label)

        # Track errors
        if prediction != query_label:
            errors.append({
                'query': query_payload['description'],
                'expected': query_label,
                'predicted': prediction,
                'top_labels': top_labels
            })

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(all_points)}...")

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)

    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {metrics['accuracy']:<10.2%}")
    print(f"{'Precision':<20} {metrics['precision']:<10.2%}")
    print(f"{'Recall':<20} {metrics['recall']:<10.2%}")
    print(f"{'F1 Score':<20} {metrics['f1']:<10.2%}")
    print(f"{'False Positive Rate':<20} {metrics['fpr']:<10.2%}")

    # Show errors
    if errors:
        print(f"\n[ERROR] Errors ({len(errors)}):")
        for error in errors[:5]:  # Show first 5
            print(f"  - {error['query'][:50]}")
            print(f"    Expected: {error['expected']}, Predicted: {error['predicted']}")
            print(f"    Top 3: {error['top_labels']}")

    print("\n[OK] Vector-only evaluation complete!")
    return metrics, errors


def full_evaluation_hybrid(client, collection, tokenizer, model, device):
    """Full evaluation with hybrid search (vector + features)."""
    print_separator("FULL EVALUATION: Hybrid Search")

    # Get all samples
    all_results = client.scroll(collection_name=collection, limit=100, with_payload=True)
    all_points = all_results[0]

    print(f"\nEvaluating {len(all_points)} samples (leave-one-out)...")
    print("Strategy: Vector search (top 10) + feature reranking (60% vector + 40% features)")

    predictions = []
    ground_truth = []
    errors = []

    for idx, query_point in enumerate(all_points):
        query_payload = query_point.payload
        query_code = query_payload['code']
        query_label = query_payload['label']
        query_id = query_point.id

        # Extract query features
        query_features = query_payload.get('features', extract_features_for_sample(query_code))

        # Embed query
        with torch.no_grad():
            inputs = tokenizer([query_code], padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            query_vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        # Vector search (top 11, exclude self later)
        results = client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=11
        ).points

        # Filter out self
        results = [r for r in results if r.id != query_id][:10]

        # Feature-based reranking
        reranked = []
        for result in results:
            features = result.payload['features']
            vector_score = result.score
            feature_similarity = calculate_feature_similarity(query_features, features)

            # Hybrid score: 60% vector + 40% features
            hybrid_score = 0.6 * vector_score + 0.4 * feature_similarity

            reranked.append({
                'payload': result.payload,
                'hybrid_score': hybrid_score
            })

        reranked.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Majority vote from top 3
        top_labels = [r['payload']['label'] for r in reranked[:3]]
        prediction = max(set(top_labels), key=top_labels.count) if top_labels else 'benign'

        predictions.append(prediction)
        ground_truth.append(query_label)

        # Track errors
        if prediction != query_label:
            errors.append({
                'query': query_payload['description'],
                'expected': query_label,
                'predicted': prediction,
                'top_labels': top_labels
            })

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(all_points)}...")

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)

    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {metrics['accuracy']:<10.2%}")
    print(f"{'Precision':<20} {metrics['precision']:<10.2%}")
    print(f"{'Recall':<20} {metrics['recall']:<10.2%}")
    print(f"{'F1 Score':<20} {metrics['f1']:<10.2%}")
    print(f"{'False Positive Rate':<20} {metrics['fpr']:<10.2%}")

    # Show errors
    if errors:
        print(f"\n[ERROR] Errors ({len(errors)}):")
        for error in errors[:5]:
            print(f"  - {error['query'][:50]}")
            print(f"    Expected: {error['expected']}, Predicted: {error['predicted']}")
            print(f"    Top 3: {error['top_labels']}")

    print("\n[OK] Hybrid evaluation complete!")
    return metrics, errors


def main():
    """Run full evaluation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Full evaluation of UniXcoder hybrid search")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing collection instead of recreating")
    args = parser.parse_args()

    print_separator("FULL EVALUATION: 40 Samples (Leave-One-Out)")
    print("\nComparing 3 approaches:")
    print("  1. Feature-only (60% recall from previous test)")
    print("  2. Vector-only (UniXcoder semantic similarity)")
    print("  3. Hybrid (60% vector + 40% features)")

    if args.use_existing:
        print("\n[MODE] Using existing collection (no re-vectorization)")
    else:
        print("\n[MODE] Will create collection if needed")

    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client_kwargs = {"host": "localhost", "port": 6333, "https": False, "timeout": 60}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)
    print("\n[OK] Connected to Qdrant")

    # Check collection exists
    collection_name = "code_samples_balanced"
    collection_exists = False
    try:
        info = client.get_collection(collection_name)
        collection_exists = True
        print(f"[OK] Collection '{collection_name}' has {info.points_count} points")
    except Exception as e:
        if args.use_existing:
            print(f"[ERROR] Collection not found!")
            print(f"   Run without --use-existing to create it first")
            print(f"   Or run: python scripts/test_hybrid_balanced.py")
            return 1
        else:
            print(f"Collection '{collection_name}' not found, will create it...")
            # Import and run setup from test_hybrid_balanced
            sys.path.insert(0, str(Path(__file__).parent))
            from test_hybrid_balanced import setup_balanced_collection
            collection_name = setup_balanced_collection(client)
            print(f"[OK] Created collection '{collection_name}'")

    # Load UniXcoder
    print("\n[LOADING] Loading UniXcoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base", use_safetensors=True)
    model.to(device)
    model.eval()
    print(f"[OK] UniXcoder ready (device: {device})")

    try:
        # Evaluation 1: Vector-only
        print("\n")
        metrics_vector, errors_vector = full_evaluation_vector_only(
            client, collection_name, tokenizer, model, device
        )

        # Evaluation 2: Hybrid
        print("\n")
        metrics_hybrid, errors_hybrid = full_evaluation_hybrid(
            client, collection_name, tokenizer, model, device
        )

        # Final comparison
        print_separator("FINAL COMPARISON")

        # Feature-only baseline (from previous test)
        metrics_features = {
            'accuracy': 0.80,  # (12 TP + 20 TN) / 40
            'precision': 1.00,
            'recall': 0.60,
            'f1': 0.75,
            'fpr': 0.00
        }

        print(f"\n{'Approach':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
        print("-" * 85)
        print(f"{'Feature-only':<25} {metrics_features['accuracy']:<12.2%} {metrics_features['precision']:<12.2%} {metrics_features['recall']:<12.2%} {metrics_features['f1']:<12.2%} {metrics_features['fpr']:<12.2%}")
        print(f"{'Vector-only':<25} {metrics_vector['accuracy']:<12.2%} {metrics_vector['precision']:<12.2%} {metrics_vector['recall']:<12.2%} {metrics_vector['f1']:<12.2%} {metrics_vector['fpr']:<12.2%}")
        print(f"{'Hybrid (60% vec + 40% feat)':<25} {metrics_hybrid['accuracy']:<12.2%} {metrics_hybrid['precision']:<12.2%} {metrics_hybrid['recall']:<12.2%} {metrics_hybrid['f1']:<12.2%} {metrics_hybrid['fpr']:<12.2%}")

        # Improvement analysis
        print("\n" + "="*85)
        print("IMPROVEMENT ANALYSIS")
        print("="*85)

        recall_improvement_vector = ((metrics_vector['recall'] - metrics_features['recall']) / metrics_features['recall']) * 100
        recall_improvement_hybrid = ((metrics_hybrid['recall'] - metrics_features['recall']) / metrics_features['recall']) * 100

        print(f"\nRecall improvement:")
        print(f"  Vector-only vs Features: {recall_improvement_vector:+.1f}%")
        print(f"  Hybrid vs Features:      {recall_improvement_hybrid:+.1f}%")

        f1_improvement_vector = ((metrics_vector['f1'] - metrics_features['f1']) / metrics_features['f1']) * 100
        f1_improvement_hybrid = ((metrics_hybrid['f1'] - metrics_features['f1']) / metrics_features['f1']) * 100

        print(f"\nF1 Score improvement:")
        print(f"  Vector-only vs Features: {f1_improvement_vector:+.1f}%")
        print(f"  Hybrid vs Features:      {f1_improvement_hybrid:+.1f}%")

        # Best approach
        best_f1 = max(metrics_features['f1'], metrics_vector['f1'], metrics_hybrid['f1'])
        if best_f1 == metrics_hybrid['f1']:
            best = "Hybrid"
        elif best_f1 == metrics_vector['f1']:
            best = "Vector-only"
        else:
            best = "Feature-only"

        print(f"\n[BEST] Best approach: {best} (F1: {best_f1:.2%})")

        # Targets
        print("\n" + "="*85)
        print("TARGET VALIDATION")
        print("="*85)
        print(f"\n{'Metric':<25} {'Hybrid':<15} {'Target':<15} {'Status':<10}")
        print("-" * 70)
        print(f"{'Precision':<25} {metrics_hybrid['precision']:<15.2%} {'>= 85%':<15} {'[OK]' if metrics_hybrid['precision'] >= 0.85 else '[WARN] '}")
        print(f"{'Recall':<25} {metrics_hybrid['recall']:<15.2%} {'>= 90%':<15} {'[OK]' if metrics_hybrid['recall'] >= 0.90 else '[WARN] '}")
        print(f"{'F1 Score':<25} {metrics_hybrid['f1']:<15.2%} {'>= 87%':<15} {'[OK]' if metrics_hybrid['f1'] >= 0.87 else '[WARN] '}")
        print(f"{'False Positive Rate':<25} {metrics_hybrid['fpr']:<15.2%} {'<= 15%':<15} {'[OK]' if metrics_hybrid['fpr'] <= 0.15 else '[WARN] '}")

        print("\n✨ Full evaluation complete!")

        return 0

    except Exception as e:
        print(f"\n[ERROR] EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
