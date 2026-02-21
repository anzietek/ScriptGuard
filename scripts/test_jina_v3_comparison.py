#!/usr/bin/env python3
"""
Compare Jina-v3 vs UniXcoder for hybrid search.

Tests:
1. Re-vectorize with Jina-v3 (retrieval.v2 adapter)
2. Full evaluation (leave-one-out)
3. Compare with UniXcoder results

Model comparison:
- UniXcoder:  125M params, 768 dims, 512 tokens
- Jina-v3:    570M params, 1024 dims, 8192 tokens

Usage:
    python scripts/test_jina_v3_comparison.py [--use-existing]

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
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Import from balanced test
sys.path.insert(0, str(Path(__file__).parent))
from test_hybrid_balanced import (
    BENIGN_SAMPLES,
    MALICIOUS_SAMPLES,
    extract_features_for_sample,
    calculate_feature_similarity
)
from test_hybrid_full_eval import calculate_metrics


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)


def setup_jina_v3_collection(client: QdrantClient, use_existing: bool = False):
    """Create or verify Jina-v3 collection."""
    print_separator("SETUP: Jina-v3 Collection")

    collection_name = "code_samples_jina_v3"

    if use_existing:
        # Check if collection exists
        try:
            info = client.get_collection(collection_name)
            print(f"[OK] Using existing collection '{collection_name}'")
            print(f"     Points: {info.points_count}, Vectors: {info.config.params.vectors.size}d")

            # Load model and return
            print(f"\n[LOADING] Loading Jina-v3 model...")
            device = "cpu"
            os.environ["FLASH_ATTENTION_FORCE_BUILD"] = "FALSE"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"

            from huggingface_hub import snapshot_download
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_path = snapshot_download(
                "jinaai/jina-embeddings-v3",
                cache_dir=cache_dir,
                ignore_patterns=["*.bin"],
                local_files_only=False
            )

            model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={"use_safetensors": True}
            )
            print(f"[OK] Jina-v3 model ready")

            return collection_name, model

        except Exception as e:
            print(f"[ERROR] Collection '{collection_name}' not found: {e}")
            print(f"        Run without --use-existing to create it first")
            sys.exit(1)

    # Create new collection
    # Delete if exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except:
        pass

    # Create collection (Jina-v3 uses 1024 dimensions)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )
    print(f"[OK] Created collection '{collection_name}'")

    # Combine samples
    all_samples = []
    for sample in BENIGN_SAMPLES:
        all_samples.append({**sample, 'label': 'benign'})
    for sample in MALICIOUS_SAMPLES:
        all_samples.append({**sample, 'label': 'malicious'})

    print(f"\nAdding {len(all_samples)} samples:")
    print(f"  Benign:    {len(BENIGN_SAMPLES)} (50%)")
    print(f"  Malicious: {len(MALICIOUS_SAMPLES)} (50%)")

    # Use Jina-v3 (optimized for code retrieval)
    print(f"\n[LOADING] Loading Jina-v3 (jinaai/jina-embeddings-v3)...")

    # Force CPU to avoid flash attention issues
    device = "cpu"
    print(f"   Device: {device} (forced CPU to avoid flash attention bugs)")

    # Disable flash attention via environment variable
    import os
    os.environ["FLASH_ATTENTION_FORCE_BUILD"] = "FALSE"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Jina-v3: 570M params, 1024 dims, 8192 tokens, task adapters
    # Download model files first using huggingface_hub
    from huggingface_hub import snapshot_download
    print(f"   Downloading model files...")

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = snapshot_download(
        "jinaai/jina-embeddings-v3",
        cache_dir=cache_dir,
        ignore_patterns=["*.bin"],  # Use safetensors (torch 2.5 compatible)
        local_files_only=False
    )
    print(f"   [OK] Downloaded to {model_path}")

    # Load from downloaded path with safetensors
    model = SentenceTransformer(
        model_path,
        device=device,
        trust_remote_code=True,
        model_kwargs={
            "use_safetensors": True  # Force safetensors to avoid torch 2.6 requirement
        }
    )

    print(f"[OK] Jina-v3 ready (1024 dimensions, 8192 max tokens, task adapters available)")

    # Generate embeddings with sentence-transformers
    codes = [sample['code'] for sample in all_samples]

    print(f"\n[LOADING] Generating Jina-v3 embeddings for {len(codes)} samples...")

    # sentence-transformers handles batching automatically
    embeddings = model.encode(
        codes,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    print(f"[OK] Embeddings generated (shape: {embeddings.shape})")

    # Add points
    points = []
    for idx, sample in enumerate(all_samples):
        features = extract_features_for_sample(sample['code'])
        vector = embeddings[idx].tolist()

        points.append(models.PointStruct(
            id=idx + 1,
            vector=vector,
            payload={
                "code": sample['code'],
                "label": sample['label'],
                "category": sample['category'],
                "description": sample['description'],
                "db_id": idx + 1,
                "features": features
            }
        ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"\n[OK] Added {len(points)} samples with Jina-v3 embeddings")

    return collection_name, model


def full_evaluation_jina_v3(client, collection, model, evaluation_type="hybrid"):
    """Full evaluation with Jina embeddings."""
    print_separator(f"FULL EVALUATION: Jina-v3 {evaluation_type.title()}")

    # Get all samples
    all_results = client.scroll(collection_name=collection, limit=100, with_payload=True)
    all_points = all_results[0]

    print(f"\nEvaluating {len(all_points)} samples (leave-one-out)...")

    if evaluation_type == "hybrid":
        print("Strategy: Vector search (top 10) + feature reranking (60% vector + 40% features)")
    else:
        print("Strategy: Vector search only (semantic similarity)")

    predictions = []
    ground_truth = []
    errors = []

    for idx, query_point in enumerate(all_points):
        query_payload = query_point.payload
        query_code = query_payload['code']
        query_label = query_payload['label']
        query_id = query_point.id

        # Extract query features (if hybrid)
        query_features = None
        if evaluation_type == "hybrid":
            query_features = query_payload.get('features', extract_features_for_sample(query_code))

        # Embed query with Jina-v3
        query_vector = model.encode([query_code], convert_to_numpy=True, normalize_embeddings=False)[0]

        # Vector search
        search_limit = 11 if evaluation_type == "hybrid" else 4
        results = client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=search_limit
        ).points

        # Filter out self
        results = [r for r in results if r.id != query_id]
        results = results[:10] if evaluation_type == "hybrid" else results[:3]

        if evaluation_type == "hybrid":
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
            top_labels = [r['payload']['label'] for r in reranked[:3]]
        else:
            # Vector-only
            top_labels = [r.payload['label'] for r in results]

        # Majority vote
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

    print(f"\n[OK] Jina-v3 {evaluation_type} evaluation complete!")
    return metrics, errors


def main():
    """Run Jina comparison."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare Jina-v3 vs UniXcoder")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing collection instead of recreating")
    args = parser.parse_args()

    print_separator("JINA-V3 vs UNIXCODER COMPARISON")
    print("\nComparing code-optimized embedding models:")
    print("  UniXcoder:      125M params, 768 dims, 512 tokens")
    print("  Jina-v3:        570M params, 1024 dims, 8192 tokens (task adapters)")

    if args.use_existing:
        print("\n[MODE] Using existing collection (no re-vectorization)")
    else:
        print("\n[MODE] Creating new collection with full vectorization")

    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client_kwargs = {"host": "localhost", "port": 6333, "https": False, "timeout": 60}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)
    print("\n[OK] Connected to Qdrant")

    try:
        # Setup Jina-v3 collection
        collection, model = setup_jina_v3_collection(client, use_existing=args.use_existing)

        # Evaluation 1: Vector-only (Jina-v3)
        print("\n")
        metrics_jina_vector, _ = full_evaluation_jina_v3(
            client, collection, model,
            evaluation_type="vector"
        )

        # Evaluation 2: Hybrid (Jina-v3)
        print("\n")
        metrics_jina_hybrid, _ = full_evaluation_jina_v3(
            client, collection, model,
            evaluation_type="hybrid"
        )

        # Load UniXcoder baseline results
        print_separator("FINAL COMPARISON: Jina-v3 vs UniXcoder")

        # UniXcoder results (from previous test)
        metrics_unixcoder_vector = {
            'accuracy': 0.90,
            'precision': 0.9444,
            'recall': 0.85,
            'f1': 0.8947,
            'fpr': 0.05
        }

        metrics_unixcoder_hybrid = {
            'accuracy': 0.90,
            'precision': 0.8636,
            'recall': 0.95,
            'f1': 0.9048,
            'fpr': 0.15
        }

        # Comparison table
        print(f"\n{'Model':<30} {'Type':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
        print("-" * 105)

        # Vector-only comparison
        print(f"{'UniXcoder':<30} {'Vector-only':<15} {metrics_unixcoder_vector['accuracy']:<12.2%} {metrics_unixcoder_vector['precision']:<12.2%} {metrics_unixcoder_vector['recall']:<12.2%} {metrics_unixcoder_vector['f1']:<12.2%} {metrics_unixcoder_vector['fpr']:<12.2%}")
        print(f"{'Jina-v3':<30} {'Vector-only':<15} {metrics_jina_vector['accuracy']:<12.2%} {metrics_jina_vector['precision']:<12.2%} {metrics_jina_vector['recall']:<12.2%} {metrics_jina_vector['f1']:<12.2%} {metrics_jina_vector['fpr']:<12.2%}")

        print()

        # Hybrid comparison
        print(f"{'UniXcoder':<30} {'Hybrid':<15} {metrics_unixcoder_hybrid['accuracy']:<12.2%} {metrics_unixcoder_hybrid['precision']:<12.2%} {metrics_unixcoder_hybrid['recall']:<12.2%} {metrics_unixcoder_hybrid['f1']:<12.2%} {metrics_unixcoder_hybrid['fpr']:<12.2%}")
        print(f"{'Jina-v3':<30} {'Hybrid':<15} {metrics_jina_hybrid['accuracy']:<12.2%} {metrics_jina_hybrid['precision']:<12.2%} {metrics_jina_hybrid['recall']:<12.2%} {metrics_jina_hybrid['f1']:<12.2%} {metrics_jina_hybrid['fpr']:<12.2%}")

        # Improvement analysis
        print("\n" + "="*105)
        print("IMPROVEMENT ANALYSIS")
        print("="*105)

        recall_improvement = ((metrics_jina_hybrid['recall'] - metrics_unixcoder_hybrid['recall']) / metrics_unixcoder_hybrid['recall']) * 100
        f1_improvement = ((metrics_jina_hybrid['f1'] - metrics_unixcoder_hybrid['f1']) / metrics_unixcoder_hybrid['f1']) * 100
        fpr_change = ((metrics_jina_hybrid['fpr'] - metrics_unixcoder_hybrid['fpr']) / metrics_unixcoder_hybrid['fpr']) * 100

        print(f"\nJina-v3 Hybrid vs UniXcoder Hybrid:")
        print(f"  Recall improvement:    {recall_improvement:+.1f}%")
        print(f"  F1 improvement:        {f1_improvement:+.1f}%")
        print(f"  FPR change:            {fpr_change:+.1f}%")

        # Best model
        if metrics_jina_hybrid['f1'] > metrics_unixcoder_hybrid['f1']:
            best = "Jina-v3"
            best_f1 = metrics_jina_hybrid['f1']
        else:
            best = "UniXcoder"
            best_f1 = metrics_unixcoder_hybrid['f1']

        print(f"\n[BEST] Best model: {best} (F1: {best_f1:.2%})")

        # Recommendation
        print("\n" + "="*105)
        print("RECOMMENDATION")
        print("="*105)

        if metrics_jina_hybrid['f1'] > metrics_unixcoder_hybrid['f1'] * 1.02:  # >2% improvement
            print("\n[OK] RECOMMENDED: Migrate to Jina-v3")
            print(f"   Reason: Significantly better F1 ({metrics_jina_hybrid['f1']:.2%} vs {metrics_unixcoder_hybrid['f1']:.2%})")
            print(f"   Bonus: Task adapters + 1024 dims + 8192 token context")
        elif metrics_jina_hybrid['f1'] < metrics_unixcoder_hybrid['f1'] * 0.98:  # >2% worse
            print("\n[WARN]  NOT RECOMMENDED: Stay with UniXcoder")
            print(f"   Reason: Jina-v3 performs worse ({metrics_jina_hybrid['f1']:.2%} vs {metrics_unixcoder_hybrid['f1']:.2%})")
        else:
            print("\n[NEUTRAL] NEUTRAL: Both models perform similarly")
            print(f"   F1 difference: {abs(metrics_jina_hybrid['f1'] - metrics_unixcoder_hybrid['f1']):.1%}")
            print("   Consider:")
            print("   - UniXcoder: Smaller (125M), faster inference, proven stable")
            print("   - Jina-v3: Task adapters, 1024 dims, longer context (8192 vs 512 tokens)")

        print("\n[DONE] Comparison complete!")

        return 0

    except Exception as e:
        print(f"\n[ERROR] COMPARISON FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
