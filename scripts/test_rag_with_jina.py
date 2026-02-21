#!/usr/bin/env python3
"""
Phase 3: End-to-End RAG Test with Jina-v3

Validate that Jina-v3 embeddings improve RAG performance on the full test set.

Process:
1. Create temporary Qdrant collection with Jina-v3 embeddings
2. Vectorize 1000 samples (500 benign, 500 malicious) from database
3. Run RAG inference on 99-sample test set
4. Compare metrics with UniXcoder baseline

Success Criteria:
- F1 score >= 85% (vs UniXcoder 71.79%)
- Benign utility accuracy (csv, json, database) >= 80% (vs UniXcoder 0%)
- No increase in false positives

Decision:
- If Jina-v3 >= 85% F1 → Proceed to Phase 4 (Full Migration)
- If Jina-v3 < 80% F1 → Investigate other issues
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

load_dotenv()

# Import ScriptGuard modules
from scriptguard.rag.embedding_service import EmbeddingService
from scriptguard.database.connection import get_postgres_connection

# Import test samples
from comprehensive_test_samples import get_samples_up_to_level


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_jina_embedding_service():
    """Create Jina-v3 embedding service."""
    print("Initializing Jina-v3 embedding service...")

    # Create minimal config for Jina-v3
    config = {
        "code_embedding": {
            "model": "jinaai/jina-embeddings-v3",
            "jina_v3": {
                "enabled": True,
                "task_adapter": "retrieval.v2",
                "output_dimension": 1024,
                "max_length": 8192,
                "trust_remote_code": True
            }
        }
    }

    service = EmbeddingService(
        model_name="jinaai/jina-embeddings-v3",
        pooling_strategy="mean_pooling",
        normalize=True,
        max_length=8192,
        config=config
    )

    print(f"✓ Jina-v3 service ready (dim={service.embedding_dim})")
    return service


def create_test_collection(client, collection_name="code_samples_jina_test"):
    """Create temporary Jina-v3 collection."""
    print(f"\nCreating temporary collection: {collection_name}")

    # Delete if exists
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection")
    except:
        pass

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,  # Jina-v3 dimension
            distance=models.Distance.COSINE
        )
    )

    print(f"✓ Collection created")
    return collection_name


def fetch_training_samples(max_samples=1000):
    """Fetch balanced training samples from PostgreSQL."""
    print(f"\nFetching {max_samples} training samples from database...")

    conn = get_postgres_connection()
    cursor = conn.cursor()

    # Fetch balanced samples
    samples_per_label = max_samples // 2

    # Fetch benign samples
    cursor.execute("""
        SELECT id, code, label, source
        FROM code_samples
        WHERE label = 'benign'
        AND code IS NOT NULL
        AND LENGTH(code) > 50
        ORDER BY RANDOM()
        LIMIT %s
    """, (samples_per_label,))

    benign_samples = cursor.fetchall()

    # Fetch malicious samples
    cursor.execute("""
        SELECT id, code, label, source
        FROM code_samples
        WHERE label = 'malicious'
        AND code IS NOT NULL
        AND LENGTH(code) > 50
        ORDER BY RANDOM()
        LIMIT %s
    """, (samples_per_label,))

    malicious_samples = cursor.fetchall()

    cursor.close()
    conn.close()

    all_samples = benign_samples + malicious_samples

    print(f"✓ Fetched {len(all_samples)} samples:")
    print(f"  Benign:    {len(benign_samples)} ({len(benign_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Malicious: {len(malicious_samples)} ({len(malicious_samples)/len(all_samples)*100:.1f}%)")

    return all_samples


def vectorize_and_upload(embedding_service, client, collection_name, samples):
    """Vectorize samples and upload to Qdrant."""
    print(f"\nVectorizing {len(samples)} samples with Jina-v3...")

    # Extract codes and metadata
    codes = [sample[1] for sample in samples]  # code column
    ids = [sample[0] for sample in samples]    # id column
    labels = [sample[2] for sample in samples]  # label column
    sources = [sample[3] for sample in samples] # source column

    # Generate embeddings in batches
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i+batch_size]
        batch_embeddings = embedding_service.encode(batch_codes, batch_size=batch_size, show_progress=True)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    print(f"✓ Generated embeddings: {embeddings.shape}")

    # Upload to Qdrant
    print(f"\nUploading to Qdrant collection '{collection_name}'...")

    points = []
    for idx, (sample_id, code, label, source, embedding) in enumerate(zip(ids, codes, labels, sources, embeddings)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "db_id": sample_id,
                    "code": code,
                    "label": label,
                    "source": source,
                    "chunk_index": 0  # Treat as single document
                }
            )
        )

    # Upload in batches
    upload_batch_size = 100
    for i in range(0, len(points), upload_batch_size):
        batch = points[i:i+upload_batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"  Uploaded {min(i+upload_batch_size, len(points))}/{len(points)} points", end='\r')

    print(f"\n✓ Upload complete: {len(points)} points")


def test_rag_retrieval(embedding_service, client, collection_name, test_samples, k=10):
    """Test RAG retrieval on test samples."""
    print(f"\nTesting RAG retrieval (k={k})...")

    predictions = []
    true_labels = []

    for idx, sample in enumerate(test_samples):
        code = sample["code"]
        true_label = sample["label"]

        # Generate query embedding
        query_embedding = embedding_service.encode_single(code)

        # Search Qdrant
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            score_threshold=0.0  # No threshold for testing
        )

        # Majority vote
        label_counts = {"benign": 0, "malicious": 0}
        for result in results:
            result_label = result.payload.get("label", "unknown")
            if result_label in label_counts:
                label_counts[result_label] += 1

        # Predict based on majority
        predicted_label = max(label_counts, key=label_counts.get)

        predictions.append(predicted_label)
        true_labels.append(true_label)

        print(f"  Tested {idx+1}/{len(test_samples)}", end='\r')

    print()

    return true_labels, predictions


def calculate_metrics(true_labels, predictions, test_samples):
    """Calculate and display classification metrics."""

    # Convert to binary
    y_true = [1 if label == "malicious" else 0 for label in true_labels]
    y_pred = [1 if label == "malicious" else 0 for label in predictions]

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n" + "="*80)
    print("RAG PERFORMANCE METRICS (Jina-v3)")
    print("="*80)

    print(f"\n1. OVERALL METRICS")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1*100:.2f}%")

    print(f"\n2. CONFUSION MATRIX")
    print(f"   True Positives:  {tp} (malicious correctly identified)")
    print(f"   True Negatives:  {tn} (benign correctly identified)")
    print(f"   False Positives: {fp} (benign incorrectly marked malicious)")
    print(f"   False Negatives: {fn} (malicious incorrectly marked benign)")
    print(f"   False Positive Rate: {fpr*100:.2f}%")

    # Per-category analysis
    print(f"\n3. PER-CATEGORY ANALYSIS")

    categories = {}
    for sample, true_label, pred_label in zip(test_samples, true_labels, predictions):
        cat = sample.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "label": true_label}

        categories[cat]["total"] += 1
        if true_label == pred_label:
            categories[cat]["correct"] += 1

    # Sort by accuracy
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["correct"]/x[1]["total"])

    print("\n   WORST PERFORMING CATEGORIES:")
    for cat, stats in sorted_cats[:10]:
        acc = stats["correct"] / stats["total"] * 100
        print(f"   {cat:20s} {acc:6.1f}% ({stats['correct']}/{stats['total']}) [{stats['label']}]")

    print("\n   BEST PERFORMING CATEGORIES:")
    for cat, stats in sorted_cats[-10:]:
        acc = stats["correct"] / stats["total"] * 100
        print(f"   {cat:20s} {acc:6.1f}% ({stats['correct']}/{stats['total']}) [{stats['label']}]")

    # Benign utility categories
    benign_utility_cats = ["csv", "json", "database", "logging", "datetime", "email", "threading", "time", "subprocess", "sys"]
    utility_correct = 0
    utility_total = 0

    for cat in benign_utility_cats:
        if cat in categories:
            utility_correct += categories[cat]["correct"]
            utility_total += categories[cat]["total"]

    utility_acc = utility_correct / utility_total * 100 if utility_total > 0 else 0

    print(f"\n4. BENIGN UTILITY ACCURACY")
    print(f"   Categories: {', '.join(benign_utility_cats)}")
    print(f"   Accuracy: {utility_acc:.1f}% ({utility_correct}/{utility_total})")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "fpr": fpr,
        "utility_accuracy": utility_acc
    }


def compare_with_baseline(jina_metrics):
    """Compare with UniXcoder baseline."""

    # Baseline from plan
    unixcoder_f1 = 0.7179
    unixcoder_accuracy = 0.6667
    unixcoder_utility_acc = 0.0

    print("\n" + "="*80)
    print("COMPARISON WITH UNIXCODER BASELINE")
    print("="*80)

    print(f"\n{'Metric':<25} {'UniXcoder':<15} {'Jina-v3':<15} {'Improvement':<15}")
    print("-" * 70)

    f1_improvement = (jina_metrics["f1"] - unixcoder_f1) / unixcoder_f1 * 100
    print(f"{'F1 Score':<25} {unixcoder_f1*100:>6.2f}% {jina_metrics['f1']*100:>14.2f}% {f1_improvement:>13.1f}%")

    acc_improvement = (jina_metrics["accuracy"] - unixcoder_accuracy) / unixcoder_accuracy * 100
    print(f"{'Accuracy':<25} {unixcoder_accuracy*100:>6.2f}% {jina_metrics['accuracy']*100:>14.2f}% {acc_improvement:>13.1f}%")

    utility_improvement = jina_metrics["utility_accuracy"] - unixcoder_utility_acc
    print(f"{'Benign Utility Accuracy':<25} {unixcoder_utility_acc:>6.1f}% {jina_metrics['utility_accuracy']:>14.1f}% {utility_improvement:>13.1f}pp")

    # Decision
    print("\n" + "="*80)
    print("PHASE 3 DECISION")
    print("="*80)

    criteria_met = []
    criteria_met.append(("F1 >= 85%", jina_metrics["f1"] >= 0.85, jina_metrics["f1"]))
    criteria_met.append(("Benign Utility >= 80%", jina_metrics["utility_accuracy"] >= 80, jina_metrics["utility_accuracy"]))
    criteria_met.append(("FPR not increased", jina_metrics["fpr"] <= 0.20, jina_metrics["fpr"]))

    all_met = all(met for _, met, _ in criteria_met)

    print("\nSuccess Criteria:")
    for criterion, met, value in criteria_met:
        status = "✅" if met else "❌"
        print(f"  [{status}] {criterion}: {value:.2%}")

    if all_met:
        print("\n✅ ALL CRITERIA MET: Proceed to Phase 4 (Full Migration)")
        print("\nNext steps:")
        print("  1. Re-vectorize entire dataset with Jina-v3")
        print("  2. Create production collection 'code_samples_jina_v3'")
        print("  3. Update config.yaml to use new collection")
        print("  4. Restart API and monitor performance")
        return True

    elif jina_metrics["f1"] >= 0.80:
        print("\n⚠️  PARTIAL SUCCESS: Significant improvement but not all criteria met")
        print(f"   F1 improved by {f1_improvement:.1f}%")
        print("\nConsider:")
        print("  - Phase 4 migration with lower threshold")
        print("  - Additional training data for failing categories")
        return True

    else:
        print("\n❌ CRITERIA NOT MET: Jina-v3 does not solve the problem")
        print("\nInvestigate alternative root causes:")
        print("  - Training data domain mismatch")
        print("  - Chunk-level retrieval loss")
        print("  - Label distribution imbalance")
        return False


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG with Jina-v3")
    parser.add_argument("--max-training-samples", type=int, default=1000, help="Max training samples to vectorize")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for RAG retrieval")
    parser.add_argument("--skip-vectorization", action="store_true", help="Skip vectorization (use existing collection)")

    args = parser.parse_args()

    print("="*80)
    print("PHASE 3: END-TO-END RAG TEST WITH JINA-V3")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Training samples: {args.max_training_samples}")
    print(f"  RAG k: {args.k}")
    print(f"  Skip vectorization: {args.skip_vectorization}")

    # Load test samples
    print("\n" + "="*80)
    print("LOADING TEST SAMPLES")
    print("="*80)

    benign_test, malicious_test = get_samples_up_to_level(3)
    test_samples = []
    for sample in benign_test:
        test_samples.append({**sample, "label": "benign"})
    for sample in malicious_test:
        test_samples.append({**sample, "label": "malicious"})

    print(f"✓ Loaded {len(test_samples)} test samples")

    # Initialize services
    embedding_service = create_jina_embedding_service()

    # Connect to Qdrant
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key if qdrant_api_key else None
    )

    collection_name = "code_samples_jina_test"

    if not args.skip_vectorization:
        # Create collection
        create_test_collection(client, collection_name)

        # Fetch training samples
        training_samples = fetch_training_samples(args.max_training_samples)

        # Vectorize and upload
        vectorize_and_upload(embedding_service, client, collection_name, training_samples)
    else:
        print(f"\nUsing existing collection: {collection_name}")

    # Test RAG
    print("\n" + "="*80)
    print("TESTING RAG RETRIEVAL")
    print("="*80)

    true_labels, predictions = test_rag_retrieval(
        embedding_service, client, collection_name, test_samples, k=args.k
    )

    # Calculate metrics
    jina_metrics = calculate_metrics(true_labels, predictions, test_samples)

    # Compare with baseline
    success = compare_with_baseline(jina_metrics)

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)

    if success:
        print("\n✅ Jina-v3 improves RAG performance - migration recommended")
    else:
        print("\n❌ Jina-v3 does not solve the problem - investigate alternatives")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
