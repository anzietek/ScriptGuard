"""
Deduplication Module
SHA256-based deduplication for code samples.
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
