"""
Deduplication Module
SHA256-based and Jaccard similarity deduplication for code samples.
"""

import hashlib
import gc
from scriptguard.utils.logger import logger
from typing import List, Dict, Set

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

def compute_hash(content: str) -> str:
    """
    Compute SHA256 hash of content.

    Args:
        content: Text content to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def deduplicate_exact(samples: List[Dict]) -> List[Dict]:
    """
    Fast exact deduplication using content hashes.
    Removes ~90% of duplicates in O(n) time.

    Args:
        samples: List of sample dictionaries with 'content' field

    Returns:
        Deduplicated list of samples with 'content_hash' added
    """
    seen_hashes: Set[str] = set()
    unique_samples = []

    initial_count = len(samples)

    for sample in samples:
        content = sample.get("content", "")
        if not content:
            continue

        content_hash = compute_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            sample["content_hash"] = content_hash
            unique_samples.append(sample)

    duplicates_removed = initial_count - len(unique_samples)
    logger.info(
        f"Exact deduplication: {initial_count} -> {len(unique_samples)} samples "
        f"({duplicates_removed} exact duplicates removed)"
    )

    return unique_samples

def deduplicate_against_database(
    samples: List[Dict],
    existing_hashes: Set[str]
) -> List[Dict]:
    """
    Remove samples that already exist in database.

    Args:
        samples: List of sample dictionaries
        existing_hashes: Set of hashes already in database

    Returns:
        List of new samples not in database
    """
    new_samples = []

    for sample in samples:
        content = sample.get("content", "")
        if not content:
            continue

        content_hash = sample.get("content_hash") or compute_hash(content)

        if content_hash not in existing_hashes:
            sample["content_hash"] = content_hash
            new_samples.append(sample)

    logger.info(
        f"Database deduplication: {len(samples)} samples -> {len(new_samples)} new "
        f"({len(samples) - len(new_samples)} already exist)"
    )

    return new_samples

def compute_jaccard_similarity(code1: str, code2: str) -> float:
    """
    Compute Jaccard similarity between two code samples.

    Args:
        code1: First code string
        code2: Second code string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    tokens1 = set(code1.split())
    tokens2 = set(code2.split())

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def deduplicate_with_threshold(
    samples: List[Dict],
    threshold: float = 0.85,
    batch_size: int = 1000,
    max_memory_mb: int = 500
) -> List[Dict]:
    """
    Remove near-duplicate samples using batched Jaccard similarity.
    Memory-safe for large datasets.

    Args:
        samples: List of sample dictionaries
        threshold: Similarity threshold (0.85 = 85% similar)
        batch_size: Number of samples per batch (reduces memory)
        max_memory_mb: Maximum memory usage before triggering GC

    Returns:
        Deduplicated samples
    """
    if threshold >= 1.0:
        # Fall back to exact hash matching
        logger.info("Using exact hash deduplication (threshold=1.0)")
        return deduplicate_exact(samples)

    unique_samples = []
    duplicates_removed = 0
    initial_count = len(samples)

    # Process in batches to control memory
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch = samples[batch_start:batch_end]

        for i, sample in enumerate(batch):
            content = sample.get("content", "")
            if not content:
                continue

            is_duplicate = False

            # Compare only against unique samples (not entire batch)
            for existing in unique_samples:
                existing_content = existing.get("content", "")
                similarity = compute_jaccard_similarity(content, existing_content)

                if similarity >= threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    logger.debug(f"Duplicate found (similarity: {similarity:.2f})")
                    break

            if not is_duplicate:
                # Add content hash for tracking
                sample["content_hash"] = compute_hash(content)
                unique_samples.append(sample)

            # Progress logging
            total_processed = batch_start + i + 1
            if total_processed % 1000 == 0:
                logger.info(f"Processed {total_processed}/{len(samples)} samples...")

        # Force garbage collection after each batch
        gc.collect()

        # Memory monitoring (if psutil available)
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > max_memory_mb:
                    logger.warning(f"High memory usage ({memory_mb:.0f}MB), triggering GC...")
                    gc.collect()
            except Exception as e:
                logger.debug(f"Memory monitoring failed: {e}")

    logger.info(
        f"Fuzzy deduplication (threshold={threshold}): "
        f"{initial_count} -> {len(unique_samples)} samples "
        f"({duplicates_removed} fuzzy duplicates removed)"
    )

    return unique_samples

def deduplicate_with_minhash_lsh(
    samples: List[Dict],
    threshold: float = 0.85,
    num_perm: int = 128
) -> List[Dict]:
    """
    Remove near-duplicate samples using MinHash LSH.
    Fast O(n) algorithm for large-scale deduplication.

    Args:
        samples: List of sample dictionaries
        threshold: Jaccard similarity threshold (0.0-1.0)
        num_perm: Number of permutations (higher = more accurate, slower)
                  128 = 95% accuracy, 256 = 98% accuracy

    Returns:
        Deduplicated samples
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning(
            "datasketch not installed. Falling back to exact deduplication. "
            "Install with: pip install datasketch>=1.6.0"
        )
        return deduplicate_exact(samples)

    if not samples:
        return samples

    logger.info(
        f"Starting MinHash LSH deduplication with threshold={threshold}, "
        f"num_perm={num_perm}"
    )

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Track unique samples and their MinHash signatures
    unique_samples = []
    duplicates_removed = 0

    for i, sample in enumerate(samples):
        content = sample.get("content", "")
        if not content:
            continue

        # Create MinHash signature for this sample
        m = MinHash(num_perm=num_perm)
        tokens = content.split()

        if not tokens:
            continue

        for token in tokens:
            m.update(token.encode('utf-8'))

        # Query LSH index for similar samples
        similar_indices = lsh.query(m)

        if similar_indices:
            # Found similar sample(s) - this is a duplicate
            duplicates_removed += 1
            logger.debug(
                f"Sample {i} is similar to {len(similar_indices)} existing samples"
            )
        else:
            # No similar samples found - this is unique
            sample_id = f"sample_{len(unique_samples)}"
            lsh.insert(sample_id, m)

            # Add content hash for compatibility
            sample["content_hash"] = compute_hash(content)
            unique_samples.append(sample)

        # Progress logging
        if (i + 1) % 1000 == 0:
            logger.info(
                f"Processed {i + 1}/{len(samples)} samples "
                f"({len(unique_samples)} unique, {duplicates_removed} duplicates)"
            )

    logger.info(
        f"MinHash LSH deduplication: {len(samples)} -> {len(unique_samples)} samples "
        f"({duplicates_removed} duplicates removed, threshold={threshold})"
    )

    return unique_samples

def deduplicate_samples(
    samples: List[Dict],
    threshold: float = 0.85,
    enable_exact: bool = True,
    enable_fuzzy: bool = True,
    batch_size: int = 1000,
    max_memory_mb: int = 500,
    method: str = "auto"
) -> List[Dict]:
    """
    Two-stage deduplication: exact hash + fuzzy matching.

    Args:
        samples: Input samples
        threshold: Fuzzy similarity threshold
        enable_exact: Use fast exact deduplication first
        enable_fuzzy: Use fuzzy matching after exact
        batch_size: Batch size for fuzzy matching (Jaccard only)
        max_memory_mb: Memory limit (Jaccard only)
        method: Fuzzy deduplication method:
            - "auto": Use MinHash LSH if n >= 1000, else Jaccard
            - "minhash_lsh": Use MinHash LSH (fast, ~95% accuracy)
            - "jaccard": Use batched Jaccard (slow, 100% accuracy)
            - "exact": Only exact hash dedup (fastest, misses near-duplicates)

    Returns:
        Deduplicated samples
    """
    logger.info(f"Starting two-stage deduplication on {len(samples)} samples...")

    # Stage 1: Exact deduplication (fast, removes exact duplicates)
    initial_count = len(samples)
    if enable_exact:
        samples = deduplicate_exact(samples)
        exact_removed = initial_count - len(samples)
        logger.info(f"Exact dedup removed {exact_removed} duplicates")

    # Stage 2: Fuzzy deduplication (catches near-duplicates)
    if enable_fuzzy and len(samples) > 0:
        # Auto-select method based on dataset size
        if method == "auto":
            if len(samples) >= 1000:
                method = "minhash_lsh"
                logger.info("Auto-selected MinHash LSH (dataset >= 1000 samples)")
            else:
                method = "jaccard"
                logger.info("Auto-selected Jaccard (dataset < 1000 samples)")

        # Apply fuzzy deduplication
        if method == "minhash_lsh":
            samples = deduplicate_with_minhash_lsh(
                samples,
                threshold=threshold,
                num_perm=128
            )
        elif method == "jaccard":
            samples = deduplicate_with_threshold(
                samples,
                threshold=threshold,
                batch_size=batch_size,
                max_memory_mb=max_memory_mb
            )
        elif method == "exact":
            logger.info("Skipping fuzzy deduplication (method='exact')")
        else:
            logger.warning(f"Unknown method '{method}', skipping fuzzy deduplication")

    logger.info(f"✓ Final deduplicated dataset: {len(samples)} samples")
    return samples
