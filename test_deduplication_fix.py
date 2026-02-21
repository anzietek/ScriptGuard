#!/usr/bin/env python3
"""
Quick test script to verify the deduplication fix works correctly.
Tests both exact and fuzzy deduplication with a simulated dataset.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scriptguard.database.deduplication import (
    deduplicate_exact,
    deduplicate_with_threshold,
    deduplicate_samples
)
from scriptguard.utils.logger import logger

def generate_test_samples(count: int = 1000, duplicate_rate: float = 0.2) -> list:
    """Generate test samples with some duplicates."""
    samples = []

    # Create unique samples
    unique_count = int(count * (1 - duplicate_rate))
    for i in range(unique_count):
        samples.append({
            "content": f"def function_{i}():\n    print('Hello {i}')\n    return {i}",
            "label": "benign",
            "source": "test"
        })

    # Create exact duplicates
    duplicate_count = count - unique_count
    for i in range(duplicate_count):
        original_idx = i % unique_count
        samples.append({
            "content": f"def function_{original_idx}():\n    print('Hello {original_idx}')\n    return {original_idx}",
            "label": "benign",
            "source": "test"
        })

    # Add a few fuzzy duplicates (slight variations)
    for i in range(min(50, unique_count // 10)):
        samples.append({
            "content": f"def function_{i}():\n    print('Hello {i}')\n    # Comment added\n    return {i}",
            "label": "benign",
            "source": "test"
        })

    return samples

def test_exact_deduplication():
    """Test exact hash-based deduplication."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Exact Deduplication")
    logger.info("="*60)

    samples = generate_test_samples(1000, duplicate_rate=0.3)
    logger.info(f"Generated {len(samples)} test samples")

    result = deduplicate_exact(samples)

    logger.info(f"Result: {len(samples)} → {len(result)} unique samples")
    assert len(result) < len(samples), "Should remove some duplicates"
    assert len(result) > 0, "Should keep some samples"
    logger.info("✓ Test passed!")

    return result

def test_fuzzy_deduplication_batched():
    """Test batched fuzzy deduplication doesn't crash on large datasets."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Batched Fuzzy Deduplication (5000 samples)")
    logger.info("="*60)

    # Create 5000 samples to simulate large dataset
    samples = generate_test_samples(5000, duplicate_rate=0.15)
    logger.info(f"Generated {len(samples)} test samples")

    result = deduplicate_with_threshold(
        samples,
        threshold=0.85,
        batch_size=1000,
        max_memory_mb=500
    )

    logger.info(f"Result: {len(samples)} → {len(result)} unique samples")
    assert len(result) < len(samples), "Should remove some duplicates"
    assert len(result) > 0, "Should keep some samples"
    logger.info("✓ Test passed!")

    return result

def test_two_stage_deduplication():
    """Test the full two-stage deduplication pipeline."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Two-Stage Deduplication (10000 samples)")
    logger.info("="*60)

    # Create large dataset similar to production scenario
    samples = generate_test_samples(10000, duplicate_rate=0.25)
    logger.info(f"Generated {len(samples)} test samples")

    result = deduplicate_samples(
        samples,
        threshold=0.92,
        enable_exact=True,
        enable_fuzzy=True,
        batch_size=1000,
        max_memory_mb=500
    )

    logger.info(f"Result: {len(samples)} → {len(result)} unique samples")
    assert len(result) < len(samples), "Should remove some duplicates"
    assert len(result) > 0, "Should keep some samples"
    logger.info("✓ Test passed!")

    return result

def test_memory_safety():
    """Test that large datasets don't crash the deduplicator."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Memory Safety (15000 samples)")
    logger.info("="*60)

    # Create even larger dataset to stress test
    samples = generate_test_samples(15000, duplicate_rate=0.2)
    logger.info(f"Generated {len(samples)} test samples")

    try:
        result = deduplicate_samples(
            samples,
            threshold=0.92,
            enable_exact=True,
            enable_fuzzy=True,
            batch_size=500,  # Smaller batches for memory safety
            max_memory_mb=400
        )

        logger.info(f"Result: {len(samples)} → {len(result)} unique samples")
        assert result is not None, "Should not crash"
        assert len(result) > 0, "Should return some samples"
        logger.info("✓ Test passed - no crash!")

    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        raise

def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("DEDUPLICATION FIX VERIFICATION TESTS")
    logger.info("="*80)

    try:
        # Test 1: Exact deduplication
        test_exact_deduplication()

        # Test 2: Batched fuzzy deduplication
        test_fuzzy_deduplication_batched()

        # Test 3: Two-stage pipeline
        test_two_stage_deduplication()

        # Test 4: Memory safety with large dataset
        test_memory_safety()

        logger.info("\n" + "="*80)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*80)
        logger.info("\nThe deduplication fix is working correctly:")
        logger.info("  • Exact deduplication removes duplicates efficiently")
        logger.info("  • Batched fuzzy matching prevents memory crashes")
        logger.info("  • Two-stage pipeline handles large datasets (10K+ samples)")
        logger.info("  • Memory management prevents silent crashes")
        logger.info("\nYou can now run the full pipeline without crashes!")

        return 0

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error(f"✗ TESTS FAILED: {e}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
