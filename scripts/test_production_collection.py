#!/usr/bin/env python3
"""
Test performance on ACTUAL production code_samples collection.
No separate test collection - uses real pipeline data.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from collections import Counter

def print_separator(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

def test_production_collection():
    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(
        host="localhost",
        port=6333,
        api_key=api_key,
        https=False
    )

    collection_name = "code_samples"

    print_separator(f"TESTING PRODUCTION COLLECTION: {collection_name}")

    # Get collection info
    try:
        info = client.get_collection(collection_name)
        total_points = info.points_count
        print(f"\nTotal points in collection: {total_points}")
    except Exception as e:
        print(f"\n[ERROR] Could not access collection: {e}")
        return

    if total_points == 0:
        print("\n[ERROR] Collection is EMPTY - run pipeline first!")
        return

    # Scroll through all points and analyze
    print_separator("ANALYZING COLLECTION DATA")

    offset = None
    points_with_features = 0
    points_without_features = 0
    sample_features = None
    label_counts = Counter()
    source_counts = Counter()
    chunk_index_counts = Counter()
    points_analyzed = 0

    print("\nScanning all points...")

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            offset=offset
        )

        points, next_offset = result

        if not points:
            break

        for point in points:
            points_analyzed += 1

            # Check features
            if "features" in point.payload and point.payload["features"]:
                points_with_features += 1
                if sample_features is None:
                    sample_features = point.payload["features"]
            else:
                points_without_features += 1

            # Count labels
            label = point.payload.get("label", "unknown")
            label_counts[label] += 1

            # Count sources
            source = point.payload.get("source", "unknown")
            source_counts[source] += 1

            # Count chunk indices
            chunk_idx = point.payload.get("chunk_index", -1)
            chunk_index_counts[chunk_idx] += 1

        # Progress
        if points_analyzed % 1000 == 0:
            print(f"  Analyzed: {points_analyzed}/{total_points}...")

        offset = next_offset
        if offset is None:
            break

    print(f"\n[OK] Analyzed {points_analyzed} points")

    # Features statistics
    print_separator("FEATURES STATISTICS")

    print(f"\nPoints WITH features: {points_with_features} ({points_with_features/total_points*100:.1f}%)")
    print(f"Points WITHOUT features: {points_without_features} ({points_without_features/total_points*100:.1f}%)")

    if points_with_features == 0:
        print("\n" + "="*80)
        print("❌ NO FEATURES FOUND IN COLLECTION!")
        print("="*80)
        print("\nThis means:")
        print("  1. Pipeline was run BEFORE the bug fix")
        print("  2. You need to re-run pipeline with: python src/main.py --clear-qdrant")
        print("  3. After re-run, features will be stored in Qdrant")
        print("\nThe bug has been FIXED in code, but existing data needs regeneration.")
        print("="*80)
    else:
        print(f"\n✅ Features are being stored! ({points_with_features/total_points*100:.1f}% coverage)")

        if sample_features:
            print(f"\nSample features structure:")
            print(f"  Keys: {list(sample_features.keys())}")
            print(f"\nExample values:")
            print(f"  Entropy: {sample_features.get('entropy')}")
            print(f"  Complexity score: {sample_features.get('complexity_score')}")
            print(f"  Code length: {sample_features.get('code_length')}")
            print(f"  Code lines: {sample_features.get('code_lines')}")

            dangerous = sample_features.get('dangerous_api_calls', [])
            if dangerous:
                print(f"  Dangerous APIs (first 5): {dangerous[:5]}")
            else:
                print(f"  Dangerous APIs: []")

            print(f"\nAPI flags:")
            print(f"  has_network_api: {sample_features.get('has_network_api')}")
            print(f"  has_file_api: {sample_features.get('has_file_api')}")
            print(f"  has_process_api: {sample_features.get('has_process_api')}")
            print(f"  has_crypto_api: {sample_features.get('has_crypto_api')}")

    # Label distribution
    print_separator("LABEL DISTRIBUTION")

    print(f"\nTotal unique labels: {len(label_counts)}")
    for label, count in label_counts.most_common():
        percentage = count / total_points * 100
        print(f"  {label:15s}: {count:6d} ({percentage:5.1f}%)")

    # Source distribution
    print_separator("SOURCE DISTRIBUTION (Top 10)")

    print(f"\nTotal unique sources: {len(source_counts)}")
    for source, count in source_counts.most_common(10):
        percentage = count / total_points * 100
        print(f"  {source[:50]:50s}: {count:6d} ({percentage:5.1f}%)")

    # Chunk index distribution
    print_separator("CHUNK INDEX DISTRIBUTION")

    print(f"\nChunk indices found: {sorted(chunk_index_counts.keys())}")
    print(f"\nDistribution (top 10):")
    for chunk_idx, count in sorted(chunk_index_counts.items())[:10]:
        percentage = count / total_points * 100
        print(f"  chunk_index={chunk_idx:3d}: {count:6d} ({percentage:5.1f}%)")

    # Documents with chunk_index=0 (parent docs)
    parent_docs = chunk_index_counts.get(0, 0)
    print(f"\nParent documents (chunk_index=0): {parent_docs}")

    if parent_docs > 0:
        avg_chunks_per_doc = total_points / parent_docs
        print(f"Average chunks per document: {avg_chunks_per_doc:.2f}")

    # Summary
    print_separator("SUMMARY")

    print(f"\nCollection: {collection_name}")
    print(f"Total points: {total_points}")
    print(f"Malicious: {label_counts.get('malicious', 0)} ({label_counts.get('malicious', 0)/total_points*100:.1f}%)")
    print(f"Benign: {label_counts.get('benign', 0)} ({label_counts.get('benign', 0)/total_points*100:.1f}%)")
    print(f"Features coverage: {points_with_features/total_points*100:.1f}%")

    if points_with_features == 0:
        print("\n⚠️  ACTION REQUIRED: Re-run pipeline to store features")
        print("    Command: python src/main.py --clear-qdrant")
    elif points_with_features < total_points:
        print(f"\n⚠️  PARTIAL COVERAGE: {points_without_features} points missing features")
        print("    Some points were added before bug fix")
        print("    Recommendation: Re-run pipeline to ensure 100% coverage")
    else:
        print("\n✅ READY: All points have features, hybrid search can work!")

    print("="*80 + "\n")

if __name__ == "__main__":
    test_production_collection()
