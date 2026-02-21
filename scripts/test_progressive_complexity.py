#!/usr/bin/env python3
"""
Progressive complexity testing: Test models from simple to complex samples.

This allows you to:
1. Start with simple samples (Level 1) to verify basics work
2. Progressively add more complex levels
3. See how performance degrades with complexity
4. Identify at which complexity level the model struggles

Complexity Levels:
- Level 1: Very Simple (20 samples) - print, basic math, simple exec/eval
- Level 2: Simple (20 samples) - file I/O, JSON, base64 obfuscation
- Level 3: Medium (20 samples) - web scraping, database, reverse shells
- Level 4: Complex (TBD) - async, multi-threading, advanced obfuscation
- Level 5: Very Complex (TBD) - ML inference, APT-level malware

Usage:
    # Test only level 1 (simplest)
    python scripts/test_progressive_complexity.py --level 1

    # Test levels 1-3 (simple to medium)
    python scripts/test_progressive_complexity.py --max-level 3

    # Test all levels
    python scripts/test_progressive_complexity.py --max-level 5

    # Use existing collection (for quick re-tests)
    python scripts/test_progressive_complexity.py --max-level 3 --use-existing
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from comprehensive_test_samples import get_samples_up_to_level, get_samples_by_level
from test_hybrid_balanced import extract_features_for_sample, calculate_feature_similarity
from test_hybrid_full_eval import calculate_metrics


def print_separator(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)


def setup_progressive_collection(client, max_level, use_existing=False):
    """Create or use collection with samples up to max_level."""
    #collection_name = f"code_samples_progressive_L{max_level}"
    collection_name = f"code_samples"

    if use_existing:
        try:
            info = client.get_collection(collection_name)
            print(f"[OK] Using existing collection '{collection_name}'")
            print(f"     Points: {info.points_count}")
            return collection_name
        except Exception as e:
            print(f"[ERROR] Collection not found: {e}")
            print(f"        Run without --use-existing to create it")
            sys.exit(1)

    # Create new collection
    print_separator(f"CREATING PROGRESSIVE COLLECTION (Levels 1-{max_level})")

    # Delete if exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except:
        pass

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    print(f"[OK] Created collection '{collection_name}'")

    # Get samples
    benign_samples, malicious_samples = get_samples_up_to_level(max_level)

    all_samples = []
    for sample in benign_samples:
        all_samples.append({**sample, 'label': 'benign'})
    for sample in malicious_samples:
        all_samples.append({**sample, 'label': 'malicious'})

    print(f"\nAdding {len(all_samples)} samples:")
    print(f"  Benign:    {len(benign_samples)} ({len(benign_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Malicious: {len(malicious_samples)} ({len(malicious_samples)/len(all_samples)*100:.1f}%)")

    # Breakdown by level
    for level in range(1, max_level + 1):
        b, m = get_samples_by_level(level)
        print(f"  Level {level}:   {len(b)} benign + {len(m)} malicious = {len(b)+len(m)} total")

    # Load UniXcoder
    print(f"\n[LOADING] Loading UniXcoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base", use_safetensors=True)
    model.to(device)
    model.eval()
    print(f"[OK] UniXcoder ready (device: {device})")

    # Vectorize samples
    print(f"\n[LOADING] Vectorizing {len(all_samples)} samples...")
    points = []

    for idx, sample in enumerate(all_samples):
        code = sample['code']

        # Extract features
        features = extract_features_for_sample(code)

        # Generate embedding
        with torch.no_grad():
            inputs = tokenizer([code], padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        points.append(models.PointStruct(
            id=idx + 1,
            vector=vector.tolist(),
            payload={
                "code": code,
                "label": sample['label'],
                "category": sample['category'],
                "description": sample['description'],
                "complexity": sample.get('complexity', 0),
                "db_id": idx + 1,
                "features": features
            }
        ))

        if (idx + 1) % 10 == 0:
            print(f"  Vectorized: {idx + 1}/{len(all_samples)}...")

    client.upsert(collection_name=collection_name, points=points)
    print(f"\n[OK] Added {len(points)} samples with UniXcoder embeddings")

    return collection_name


def evaluate_progressive(client, collection, tokenizer, model, device, max_level):
    """Evaluate performance on progressive complexity levels."""
    print_separator(f"EVALUATION: Levels 1-{max_level} (Leave-One-Out)")

    # Get all samples
    all_results = client.scroll(collection_name=collection, limit=1000, with_payload=True)
    all_points = all_results[0]

    print(f"\nEvaluating {len(all_points)} samples...")
    print("Strategy: Hybrid (60% vector + 40% features)")

    # Overall metrics
    predictions = []
    ground_truth = []
    errors = []

    # Per-level metrics
    level_results = {i: {'predictions': [], 'ground_truth': [], 'errors': []}
                     for i in range(1, max_level + 1)}

    for idx, query_point in enumerate(all_points):
        query_payload = query_point.payload
        query_code = query_payload['code']
        query_label = query_payload['label']
        query_complexity = query_payload.get('complexity', 0)
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

        # Record overall
        predictions.append(prediction)
        ground_truth.append(query_label)

        # Record per-level
        if query_complexity > 0:
            level_results[query_complexity]['predictions'].append(prediction)
            level_results[query_complexity]['ground_truth'].append(query_label)

        # Track errors
        if prediction != query_label:
            error = {
                'query': query_payload['description'],
                'category': query_payload['category'],
                'expected': query_label,
                'predicted': prediction,
                'complexity': query_complexity,
                'top_labels': top_labels
            }
            errors.append(error)

            if query_complexity > 0:
                level_results[query_complexity]['errors'].append(error)

        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(all_points)}...")

    # Calculate overall metrics
    print_separator("OVERALL RESULTS")
    overall_metrics = calculate_metrics(predictions, ground_truth)

    print(f"\n{'Metric':<25} {'Value':<10}")
    print("-" * 35)
    print(f"{'Accuracy':<25} {overall_metrics['accuracy']:<10.2%}")
    print(f"{'Precision':<25} {overall_metrics['precision']:<10.2%}")
    print(f"{'Recall':<25} {overall_metrics['recall']:<10.2%}")
    print(f"{'F1 Score':<25} {overall_metrics['f1']:<10.2%}")
    print(f"{'False Positive Rate':<25} {overall_metrics['fpr']:<10.2%}")
    print(f"{'Total Errors':<25} {len(errors)}/{len(all_points)}")

    # Calculate per-level metrics
    print_separator("PER-LEVEL BREAKDOWN")

    print(f"\n{'Level':<8} {'Samples':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Errors':<10}")
    print("-" * 90)

    for level in range(1, max_level + 1):
        if level_results[level]['predictions']:
            metrics = calculate_metrics(
                level_results[level]['predictions'],
                level_results[level]['ground_truth']
            )
            num_samples = len(level_results[level]['predictions'])
            num_errors = len(level_results[level]['errors'])

            print(f"{level:<8} {num_samples:<10} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.2%} "
                  f"{metrics['recall']:<12.2%} {metrics['f1']:<12.2%} {num_errors:<10}")

    # Show some errors
    if errors:
        print_separator("ERROR ANALYSIS")
        print(f"\nTotal errors: {len(errors)}")

        # Group by complexity
        errors_by_level = {i: [] for i in range(1, max_level + 1)}
        for error in errors:
            if error['complexity'] > 0:
                errors_by_level[error['complexity']].append(error)

        for level in range(1, max_level + 1):
            level_errors = errors_by_level[level]
            if level_errors:
                print(f"\nLevel {level} errors ({len(level_errors)}):")
                for error in level_errors[:3]:  # Show first 3
                    print(f"  - [{error['category']}] {error.get('query', 'N/A')}")
                    print(f"    Expected: {error['expected']}, Predicted: {error['predicted']}")

    return overall_metrics, level_results


def main():
    """Run progressive complexity testing."""
    parser = argparse.ArgumentParser(description="Progressive complexity testing")
    parser.add_argument("--level", type=int, help="Test only this specific level (1-5)")
    parser.add_argument("--max-level", type=int, default=3, help="Test up to this level (default: 3)")
    parser.add_argument("--use-existing", action="store_true", help="Use existing collection")
    args = parser.parse_args()

    if args.level:
        max_level = args.level
        print(f"Testing ONLY Level {args.level}")
    else:
        max_level = args.max_level
        print(f"Testing Levels 1-{max_level}")

    # Validate level
    if max_level < 1 or max_level > 5:
        print("Error: Level must be between 1 and 5")
        sys.exit(1)

    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client_kwargs = {"host": "localhost", "port": 6333, "https": False, "timeout": 60}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)
    print("\n[OK] Connected to Qdrant")

    try:
        # Setup collection
        collection = setup_progressive_collection(client, max_level, use_existing=args.use_existing)

        if not args.use_existing:
            # Load model for evaluation
            print(f"\n[LOADING] Loading UniXcoder for evaluation...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
            model = AutoModel.from_pretrained("microsoft/unixcoder-base", use_safetensors=True)
            model.to(device)
            model.eval()
            print(f"[OK] UniXcoder ready")
        else:
            # Just load model
            print(f"\n[LOADING] Loading UniXcoder...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
            model = AutoModel.from_pretrained("microsoft/unixcoder-base", use_safetensors=True)
            model.to(device)
            model.eval()

        # Evaluate
        print("\n")
        overall_metrics, level_metrics = evaluate_progressive(
            client, collection, tokenizer, model, device, max_level
        )

        print_separator("TESTING COMPLETE")
        print(f"\nOverall F1 Score: {overall_metrics['f1']:.2%}")
        print(f"Overall Accuracy: {overall_metrics['accuracy']:.2%}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
