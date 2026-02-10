"""
Deduplication Module
SHA256-based and Jaccard similarity deduplication for code samples.
"""

import hashlib
from scriptguard.utils.logger import logger
from typing import List, Dict, Set

def compute_hash(content: str) -> str:
    """
    Compute SHA256 hash of content.

    Args:
        content: Text content to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Remove duplicate samples based on content hash.

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
        f"Deduplication: {initial_count} samples -> {len(unique_samples)} unique "
        f"({duplicates_removed} duplicates removed)"
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
    threshold: float = 0.85
) -> List[Dict]:
    """
    Remove near-duplicate samples using Jaccard similarity.

    Args:
        samples: List of sample dictionaries
        threshold: Similarity threshold (0.85 = 85% similar)

    Returns:
        Deduplicated samples
    """
    if threshold >= 1.0:
        # Fall back to exact hash matching
        logger.info("Using exact hash deduplication (threshold=1.0)")
        return deduplicate_samples(samples)

    unique_samples = []
    duplicates_removed = 0

    for i, sample in enumerate(samples):
        content = sample.get("content", "")
        if not content:
            continue

        is_duplicate = False

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

        # Progress logging for large datasets
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples...")

    logger.info(
        f"Fuzzy deduplication (threshold={threshold}): "
        f"{len(samples)} -> {len(unique_samples)} unique "
        f"({duplicates_removed} duplicates removed)"
    )

    return unique_samples
