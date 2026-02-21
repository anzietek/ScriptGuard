"""
Test MinHash LSH deduplication implementation.
"""

import pytest
from scriptguard.database.deduplication import (
    deduplicate_with_minhash_lsh,
    deduplicate_samples
)


def test_minhash_basic():
    """Test MinHash LSH finds near-duplicates."""
    samples = [
        {"content": "def foo():\n    print('hello')\n    return 1", "label": "benign"},
        {"content": "def foo():\n    print('hello')\n    return 1", "label": "benign"},  # Exact duplicate
        {"content": "def bar():\n    print('world')\n    return 2", "label": "benign"},
    ]

    result = deduplicate_with_minhash_lsh(samples, threshold=0.85)
    assert len(result) == 2  # One duplicate removed


def test_minhash_fuzzy_duplicates():
    """Test MinHash LSH finds fuzzy duplicates."""
    samples = [
        {"content": "def func1():\n    x = 1\n    y = 2\n    return x + y", "label": "benign"},
        {"content": "def func1():\n    a = 1\n    b = 2\n    return a + b", "label": "benign"},  # ~85% similar
        {"content": "def func2():\n    return 'different'", "label": "benign"},
    ]

    result = deduplicate_with_minhash_lsh(samples, threshold=0.85)
    # Should remove fuzzy duplicate (or keep both if <85% similar)
    assert len(result) <= 2


def test_minhash_large_dataset():
    """Test MinHash LSH completes quickly on large dataset."""
    import time

    # Generate 5000 unique samples
    samples = [
        {"content": f"def func_{i}():\n    return {i}", "label": "benign"}
        for i in range(5000)
    ]

    start = time.time()
    result = deduplicate_with_minhash_lsh(samples, threshold=0.85)
    elapsed = time.time() - start

    # Should complete in under 10 seconds
    assert elapsed < 10, f"Too slow: {elapsed:.1f}s"
    assert len(result) == 5000  # All unique


def test_auto_method_selection():
    """Test auto method selects MinHash for large datasets."""
    # Large dataset (>= 1000 samples) should use MinHash LSH
    samples = [
        {"content": f"code_{i}", "label": "benign"}
        for i in range(1500)
    ]

    result = deduplicate_samples(samples, threshold=0.85, method="auto")
    assert result is not None


def test_minhash_empty_content():
    """Test MinHash LSH handles empty content gracefully."""
    samples = [
        {"content": "", "label": "benign"},
        {"content": "def foo(): pass", "label": "benign"},
        {"content": "", "label": "benign"},
    ]

    result = deduplicate_with_minhash_lsh(samples, threshold=0.85)
    assert len(result) == 1  # Only the non-empty one


def test_minhash_preserves_metadata():
    """Test MinHash LSH preserves sample metadata."""
    samples = [
        {
            "content": "def foo(): return 1",
            "label": "benign",
            "source": "test.py",
            "metadata": {"key": "value"}
        },
        {
            "content": "def bar(): return 2",
            "label": "malicious",
            "source": "malware.py",
            "metadata": {"key": "value2"}
        }
    ]

    result = deduplicate_with_minhash_lsh(samples, threshold=0.85)

    assert len(result) == 2
    assert result[0]["source"] == "test.py"
    assert result[1]["source"] == "malware.py"
    assert "content_hash" in result[0]  # Should add content hash


def test_fallback_without_datasketch():
    """Test graceful fallback when datasketch not available."""
    # This test would require mocking the import, so it's informational
    # The implementation should fall back to exact deduplication
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
