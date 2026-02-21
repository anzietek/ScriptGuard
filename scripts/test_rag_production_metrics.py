#!/usr/bin/env python3
"""
RAG Production Metrics Test - Leave-One-Out Evaluation

Tests RAG performance on REAL production data:
- Samples random test set from code_samples collection
- Leave-one-out evaluation (query not in results)
- Calculates precision, recall, F1, accuracy
- Shows detailed error analysis
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
import random
from collections import defaultdict
from typing import List, Dict, Any

from scriptguard.rag.code_similarity_store import CodeSimilarityStore

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

def sample_test_set(client, collection_name, n_samples=200, stratified=True):
    """
    Sample test set from production collection.

    Args:
        n_samples: Total samples to test (default 200)
        stratified: If True, balance malicious/benign (default True)
    """
    print_separator("SAMPLING TEST SET")

    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    print(f"Total points in collection: {total_points}")

    # Scroll and collect samples by label
    samples_by_label = defaultdict(list)
    offset = None

    print("\nScanning collection for samples with chunk_index=0 (parent docs)...")

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            offset=offset,
            scroll_filter={
                "must": [
                    {"key": "chunk_index", "match": {"value": 0}}
                ]
            }
        )

        points, next_offset = result

        if not points:
            break

        for point in points:
            label = point.payload.get("label", "unknown")
            if label in ["malicious", "benign"]:
                samples_by_label[label].append({
                    "id": point.id,
                    "db_id": point.payload.get("db_id"),
                    "label": label,
                    "source": point.payload.get("source", "unknown"),
                    "code_preview": point.payload.get("code_preview", ""),
                    "features": point.payload.get("features", {})
                })

        offset = next_offset
        if offset is None:
            break

    print(f"Found parent documents:")
    print(f"  Malicious: {len(samples_by_label['malicious'])}")
    print(f"  Benign: {len(samples_by_label['benign'])}")

    # Sample test set
    if stratified:
        # Balance: 50% malicious, 50% benign
        samples_per_label = n_samples // 2

        malicious_samples = random.sample(
            samples_by_label['malicious'],
            min(samples_per_label, len(samples_by_label['malicious']))
        )
        benign_samples = random.sample(
            samples_by_label['benign'],
            min(samples_per_label, len(samples_by_label['benign']))
        )

        test_set = malicious_samples + benign_samples
    else:
        # Random sample
        all_samples = samples_by_label['malicious'] + samples_by_label['benign']
        test_set = random.sample(all_samples, min(n_samples, len(all_samples)))

    random.shuffle(test_set)

    print(f"\n✓ Sampled {len(test_set)} test samples:")
    print(f"  Malicious: {sum(1 for s in test_set if s['label'] == 'malicious')}")
    print(f"  Benign: {sum(1 for s in test_set if s['label'] == 'benign')}")

    return test_set

def evaluate_rag(store: CodeSimilarityStore, client, collection_name, test_set, k=5, strategy='majority_vote'):
    """
    Evaluate RAG using leave-one-out.

    Args:
        store: CodeSimilarityStore instance for searching
        k: Number of neighbors to retrieve
        strategy: 'majority_vote' or 'top1'
    """
    print_separator(f"EVALUATING RAG (k={k}, strategy={strategy})")

    predictions = []
    ground_truth = []
    errors = []
    confidence_scores = []

    print(f"\nProcessing {len(test_set)} queries...")

    for idx, query_sample in enumerate(test_set):
        query_id = query_sample['id']
        query_db_id = query_sample['db_id']
        true_label = query_sample['label']

        # Get full code content for this sample
        # First get the point to extract full code
        try:
            point = client.retrieve(
                collection_name=collection_name,
                ids=[query_id],
                with_payload=True
            )[0]

            # Try to get full code from code_preview or parent_context
            query_code = point.payload.get("code_preview", "")

            # If code_preview is too short, try to reconstruct from chunks
            if len(query_code) < 100 and query_db_id:
                # Get all chunks for this document
                chunks_result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {"key": "db_id", "match": {"value": query_db_id}}
                        ]
                    },
                    limit=100,
                    with_payload=True
                )
                chunks = chunks_result[0]
                # Reconstruct code from chunks sorted by chunk_index
                sorted_chunks = sorted(chunks, key=lambda x: x.payload.get("chunk_index", 0))
                query_code = "\n".join([c.payload.get("code_preview", "") for c in sorted_chunks])

        except Exception as e:
            print(f"  [ERROR] Could not retrieve query code for {query_id}: {e}")
            query_code = query_sample['code_preview']  # Fallback to preview

        # Search for similar code using RAG
        try:
            results = store.search_similar_code(
                query_code=query_code,
                k=k * 3,  # Get more to filter out self
                filter_label=None  # Get both labels
            )
        except Exception as e:
            print(f"  [ERROR] Query {idx+1} failed: {e}")
            predictions.append('benign')  # Default
            ground_truth.append(true_label)
            continue

        # Filter out the query itself (by db_id)
        if query_db_id:
            results = [r for r in results if r.get('db_id') != query_db_id]

        # Take top k
        results = results[:k]

        if not results:
            print(f"  [WARN] Query {idx+1}: No results found after filtering")
            predictions.append('benign')  # Default to benign
            ground_truth.append(true_label)
            continue

        # Extract labels from results
        result_labels = [r.get("label", "unknown") for r in results]
        result_scores = [r.get("score", 0.0) for r in results]

        # Prediction strategy
        if strategy == 'majority_vote':
            # Count votes
            malicious_votes = sum(1 for l in result_labels if l == 'malicious')
            benign_votes = sum(1 for l in result_labels if l == 'benign')

            predicted_label = 'malicious' if malicious_votes > benign_votes else 'benign'
            confidence = max(malicious_votes, benign_votes) / len(result_labels)
        else:  # top1
            predicted_label = result_labels[0] if result_labels else 'benign'
            confidence = result_scores[0] if result_scores else 0.0

        predictions.append(predicted_label)
        ground_truth.append(true_label)
        confidence_scores.append(confidence)

        # Track errors
        if predicted_label != true_label:
            errors.append({
                'query_id': query_id,
                'query_db_id': query_db_id,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'source': query_sample['source'],
                'code_preview': query_sample['code_preview'][:100],
                'result_labels': result_labels,
                'result_scores': result_scores
            })

        # Progress
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(test_set)}...")

    print(f"\n✓ Evaluated {len(predictions)} queries")

    return predictions, ground_truth, errors, confidence_scores

def main():
    # Config
    TEST_SAMPLES = 200  # Number of samples to test
    K_NEIGHBORS = 5     # Number of neighbors to retrieve
    STRATEGY = 'majority_vote'  # 'majority_vote' or 'top1'

    print_separator("RAG PRODUCTION METRICS TEST")
    print(f"\nConfiguration:")
    print(f"  Test samples: {TEST_SAMPLES}")
    print(f"  K neighbors: {K_NEIGHBORS}")
    print(f"  Strategy: {STRATEGY}")

    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(
        host="localhost",
        port=6333,
        api_key=api_key,
        https=False
    )

    collection_name = "code_samples"

    # Initialize CodeSimilarityStore
    print("\nInitializing CodeSimilarityStore...")
    store = CodeSimilarityStore(
        host="localhost",
        port=6333,
        collection_name=collection_name,
        api_key=api_key
    )
    print("✓ RAG store initialized")

    # Sample test set
    test_set = sample_test_set(
        client,
        collection_name,
        n_samples=TEST_SAMPLES,
        stratified=True
    )

    # Evaluate
    predictions, ground_truth, errors, confidence_scores = evaluate_rag(
        store,
        client,
        collection_name,
        test_set,
        k=K_NEIGHBORS,
        strategy=STRATEGY
    )

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
    print(f"  True Positives:  {metrics['tp']:4d}")
    print(f"  True Negatives:  {metrics['tn']:4d}")
    print(f"  False Positives: {metrics['fp']:4d}")
    print(f"  False Negatives: {metrics['fn']:4d}")

    print(f"\nConfidence Statistics:")
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"  Average confidence: {avg_confidence:.2%}")
        print(f"  Min confidence: {min(confidence_scores):.2%}")
        print(f"  Max confidence: {max(confidence_scores):.2%}")

    # Error analysis
    if errors:
        print_separator(f"ERROR ANALYSIS ({len(errors)} errors)")

        # Group by error type
        false_positives = [e for e in errors if e['true_label'] == 'benign']
        false_negatives = [e for e in errors if e['true_label'] == 'malicious']

        print(f"\nFalse Positives: {len(false_positives)} (benign marked as malicious)")
        for i, err in enumerate(false_positives[:10], 1):
            print(f"\n  {i}. Source: {err['source']}")
            print(f"     Confidence: {err['confidence']:.2%}")
            print(f"     Retrieved labels: {err['result_labels']}")
            print(f"     Code: {err['code_preview']}...")

        if len(false_positives) > 10:
            print(f"\n  ... and {len(false_positives) - 10} more false positives")

        print(f"\nFalse Negatives: {len(false_negatives)} (malicious marked as benign)")
        for i, err in enumerate(false_negatives[:10], 1):
            print(f"\n  {i}. Source: {err['source']}")
            print(f"     Confidence: {err['confidence']:.2%}")
            print(f"     Retrieved labels: {err['result_labels']}")
            print(f"     Code: {err['code_preview']}...")

        if len(false_negatives) > 10:
            print(f"\n  ... and {len(false_negatives) - 10} more false negatives")

        # Source-based error analysis
        print(f"\nErrors by Source:")
        error_sources = defaultdict(int)
        for err in errors:
            error_sources[err['source']] += 1

        for source, count in sorted(error_sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source:30s}: {count:3d} errors")

    print_separator("SUMMARY")

    print(f"\nRAG Performance on Production Data:")
    print(f"  Test samples: {len(test_set)}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    print(f"  Error rate: {len(errors)/len(test_set):.2%}")

    if metrics['f1'] >= 0.90:
        print(f"\n✅ EXCELLENT: F1 >= 90%")
    elif metrics['f1'] >= 0.80:
        print(f"\n✓ GOOD: F1 >= 80%")
    elif metrics['f1'] >= 0.70:
        print(f"\n⚠️  MODERATE: F1 >= 70%")
    else:
        print(f"\n❌ POOR: F1 < 70%")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
