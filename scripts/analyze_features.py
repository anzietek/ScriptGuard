#!/usr/bin/env python3
"""Analyze feature distribution in Qdrant collection."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from collections import defaultdict
import statistics

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def analyze_features():
    """Analyze features in Qdrant collection."""
    # Get API key from environment (optional for local Qdrant)
    api_key = os.getenv("QDRANT_API_KEY")

    # Only pass api_key if it's actually set (not empty string)
    client_kwargs = {
        "host": "localhost",
        "port": 6333,
        "https": False,  # Disable SSL for local Qdrant
        "timeout": 60
    }

    if api_key:
        client_kwargs["api_key"] = api_key

    client = QdrantClient(**client_kwargs)

    print("Analyzing features in code_samples collection...")
    print("=" * 60)

    # Scroll through all points
    offset = None
    samples_with_features = 0
    samples_without_features = 0

    entropies = []
    complexities = []
    code_lengths = []
    dangerous_counts = defaultdict(int)
    api_usage = defaultdict(int)
    label_distribution = defaultdict(int)

    total_processed = 0

    while True:
        results = client.scroll(
            collection_name="code_samples",
            limit=100,
            offset=offset
        )

        points, next_offset = results
        if not points:
            break

        for point in points:
            total_processed += 1

            # Track label distribution
            label = point.payload.get('label', 'unknown')
            label_distribution[label] += 1

            features = point.payload.get('features')

            if features:
                samples_with_features += 1

                # Collect metrics
                entropy = features.get('entropy', 0)
                complexity = features.get('complexity_score', 0)
                code_length = features.get('code_length', 0)

                if entropy > 0:
                    entropies.append(entropy)
                if complexity > 0:
                    complexities.append(complexity)
                if code_length > 0:
                    code_lengths.append(code_length)

                # Count dangerous patterns
                for api in features.get('dangerous_api_calls', []):
                    dangerous_counts[api] += 1

                # Count API usage
                if features.get('has_network_api'):
                    api_usage['network'] += 1
                if features.get('has_file_api'):
                    api_usage['file'] += 1
                if features.get('has_process_api'):
                    api_usage['process'] += 1
                if features.get('has_crypto_api'):
                    api_usage['crypto'] += 1
            else:
                samples_without_features += 1

        # Progress indicator
        if total_processed % 1000 == 0:
            print(f"  Processed {total_processed} samples...")

        offset = next_offset
        if offset is None:
            break

    # Print statistics
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)

    total = samples_with_features + samples_without_features
    print(f"\nTotal samples: {total}")
    print(f"✅ With features: {samples_with_features} ({samples_with_features/total*100:.1f}%)")
    print(f"❌ Without features: {samples_without_features} ({samples_without_features/total*100:.1f}%)")

    print(f"\nLabel Distribution:")
    for label, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

    if entropies:
        print(f"\nEntropy Statistics:")
        print(f"  Mean: {statistics.mean(entropies):.2f}")
        print(f"  Median: {statistics.median(entropies):.2f}")
        print(f"  Min: {min(entropies):.2f}")
        print(f"  Max: {max(entropies):.2f}")
        print(f"  Std Dev: {statistics.stdev(entropies):.2f}")

        # Entropy buckets
        high_entropy = sum(1 for e in entropies if e > 6.0)
        medium_entropy = sum(1 for e in entropies if 4.0 <= e <= 6.0)
        low_entropy = sum(1 for e in entropies if e < 4.0)

        print(f"\n  Distribution:")
        print(f"    High (>6.0): {high_entropy} samples ({high_entropy/len(entropies)*100:.1f}%)")
        print(f"    Medium (4-6): {medium_entropy} samples ({medium_entropy/len(entropies)*100:.1f}%)")
        print(f"    Low (<4.0): {low_entropy} samples ({low_entropy/len(entropies)*100:.1f}%)")

    if complexities:
        print(f"\nComplexity Statistics:")
        print(f"  Mean: {statistics.mean(complexities):.1f}")
        print(f"  Median: {statistics.median(complexities):.1f}")
        print(f"  Min: {min(complexities)}")
        print(f"  Max: {max(complexities)}")

        # Complexity buckets
        very_high = sum(1 for c in complexities if c > 70)
        high_complexity = sum(1 for c in complexities if 50 < c <= 70)
        medium_complexity = sum(1 for c in complexities if 30 <= c <= 50)
        low_complexity = sum(1 for c in complexities if c < 30)

        print(f"\n  Distribution:")
        print(f"    Very High (>70): {very_high} samples ({very_high/len(complexities)*100:.1f}%)")
        print(f"    High (50-70): {high_complexity} samples ({high_complexity/len(complexities)*100:.1f}%)")
        print(f"    Medium (30-50): {medium_complexity} samples ({medium_complexity/len(complexities)*100:.1f}%)")
        print(f"    Low (<30): {low_complexity} samples ({low_complexity/len(complexities)*100:.1f}%)")

    if code_lengths:
        print(f"\nCode Length Statistics:")
        print(f"  Mean: {statistics.mean(code_lengths):.0f} characters")
        print(f"  Median: {statistics.median(code_lengths):.0f} characters")
        print(f"  Min: {min(code_lengths)}")
        print(f"  Max: {max(code_lengths)}")

    if api_usage:
        print(f"\nAPI Usage:")
        for api_type, count in sorted(api_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = count/samples_with_features*100 if samples_with_features > 0 else 0
            print(f"  {api_type}: {count} samples ({percentage:.1f}%)")

    if dangerous_counts:
        print(f"\nTop 15 Dangerous APIs:")
        for api, count in sorted(dangerous_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {api}: {count} occurrences")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    if samples_with_features == total:
        print("✅ SUCCESS: All samples have features!")
        print("   Re-indexing completed successfully.")
    elif samples_with_features > total * 0.95:
        print("⚠️  WARNING: Most samples have features, but some are missing")
        print(f"   {samples_without_features} samples without features ({samples_without_features/total*100:.1f}%)")
        print("   Consider re-running vectorization for missing samples.")
    else:
        print("❌ FAILURE: Many samples missing features!")
        print(f"   {samples_without_features} samples without features ({samples_without_features/total*100:.1f}%)")
        print("   Re-indexing incomplete - run vectorization pipeline again.")

    # Feature quality checks
    print("\nFeature Quality:")

    if entropies:
        if statistics.mean(entropies) > 0:
            print("  ✅ Entropy values look reasonable")
        else:
            print("  ❌ Entropy values suspicious (all zeros?)")

    if complexities:
        if statistics.mean(complexities) > 0:
            print("  ✅ Complexity scores look reasonable")
        else:
            print("  ❌ Complexity scores suspicious (all zeros?)")

    if api_usage:
        print("  ✅ API usage flags populated")
    else:
        print("  ⚠️  No API usage flags found")

    if dangerous_counts:
        print("  ✅ Dangerous API patterns detected")
    else:
        print("  ⚠️  No dangerous patterns found (may be normal for benign code)")

    print("=" * 60)

    # Return summary for programmatic use
    return {
        "total_samples": total,
        "samples_with_features": samples_with_features,
        "samples_without_features": samples_without_features,
        "coverage_percent": samples_with_features/total*100 if total > 0 else 0,
        "entropy_mean": statistics.mean(entropies) if entropies else 0,
        "complexity_mean": statistics.mean(complexities) if complexities else 0,
    }


def main():
    """Main entry point."""
    try:
        summary = analyze_features()

        # Exit code based on coverage
        if summary["coverage_percent"] >= 99:
            return 0  # Success
        elif summary["coverage_percent"] >= 90:
            return 1  # Warning
        else:
            return 2  # Failure

    except Exception as e:
        print(f"\n❌ Error analyzing features: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
