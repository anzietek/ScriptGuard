"""
Qdrant-based Data Augmentation Step
Enriches training data with CVE patterns and malware signatures from Qdrant vector store.
"""

from typing import List, Dict, Any, Optional
from zenml import step
from scriptguard.utils.logger import logger


@step
def augment_with_qdrant_patterns(
    data: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Augment training data with CVE patterns from Qdrant.

    This step enriches the training dataset by adding known CVE patterns
    and malware signatures stored in Qdrant as additional training examples.

    Args:
        data: Existing training data
        config: Configuration dictionary with Qdrant settings

    Returns:
        Augmented dataset with Qdrant patterns included
    """
    logger.info(f"Starting Qdrant augmentation with {len(data)} existing samples")

    # Check if Qdrant augmentation is enabled
    augmentation_config = config.get("augmentation", {})
    if not augmentation_config.get("use_qdrant_patterns", False):
        logger.info("Qdrant augmentation disabled in config. Skipping.")
        return data

    try:
        from scriptguard.rag.qdrant_store import QdrantStore

        # Initialize Qdrant
        qdrant_config = config.get("qdrant", {})
        store = QdrantStore(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
            embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2"),
            api_key=qdrant_config.get("api_key") if qdrant_config.get("api_key") else None,
            use_https=qdrant_config.get("use_https", False)
        )

        # Get collection info
        info = store.get_collection_info()
        points_count = info.get('points_count', 0)

        if points_count == 0:
            logger.warning("Qdrant collection is empty. No patterns to augment with.")
            return data

        logger.info(f"Found {points_count} patterns in Qdrant collection")

        # Scroll through all points in collection
        # Qdrant scroll API returns batches of points
        augmented_samples = []
        offset = None
        batch_size = 100
        total_added = 0

        while True:
            # Scroll through collection
            scroll_result = store.client.scroll(
                collection_name=store.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just payload
            )

            points, next_offset = scroll_result

            if not points:
                break

            # Convert Qdrant points to training samples
            for point in points:
                payload = point.payload

                # Skip if no description
                if not payload.get('description'):
                    continue

                # Create training sample from CVE/pattern
                sample = _create_sample_from_pattern(payload, augmentation_config)
                if sample:
                    augmented_samples.append(sample)
                    total_added += 1

            # Check if we've reached the end
            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Added {total_added} CVE/pattern samples from Qdrant")

        # Combine original data with augmented samples
        combined_data = data + augmented_samples

        logger.info(f"Final dataset size: {len(combined_data)} samples")
        logger.info(f"Augmentation added: {len(augmented_samples)} samples ({len(augmented_samples)/len(combined_data)*100:.1f}%)")

        return combined_data

    except Exception as e:
        logger.warning(f"Qdrant augmentation failed: {e}")
        logger.warning("Continuing with original dataset without Qdrant augmentation")
        return data


def _create_sample_from_pattern(
    payload: Dict[str, Any],
    augmentation_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Create a training sample from a Qdrant pattern.

    Args:
        payload: Qdrant point payload with CVE/pattern info
        augmentation_config: Augmentation configuration

    Returns:
        Training sample dict or None if invalid
    """
    description = payload.get('description', '')
    pattern = payload.get('pattern', '')
    cve_id = payload.get('cve_id', '')
    severity = payload.get('severity', 'UNKNOWN')

    # Determine format style from config
    format_style = augmentation_config.get('qdrant_format_style', 'detailed')

    if format_style == 'pattern_only':
        # Just use the pattern code if available
        if pattern:
            content = pattern
        else:
            content = description

    elif format_style == 'description_only':
        # Use description as pseudo-code/comment
        content = f"# {description}\n# Pattern: {pattern if pattern else 'N/A'}"

    else:  # 'detailed' (default)
        # Create detailed example with context
        if pattern:
            content = f"""# Vulnerability: {description}
# CVE: {cve_id if cve_id else 'N/A'}
# Severity: {severity}
# Known malicious pattern:
{pattern}
"""
        else:
            # If no pattern code, create a comment-based example
            content = f"""# Known vulnerability pattern
# Description: {description}
# CVE: {cve_id if cve_id else 'N/A'}
# Severity: {severity}
"""

    # All Qdrant patterns are malicious by definition
    sample = {
        'content': content,
        'label': 'malicious',
        'source': 'qdrant',
        'metadata': {
            'cve_id': cve_id,
            'severity': severity,
            'type': payload.get('type', 'vulnerability')
        }
    }

    return sample


@step
def validate_qdrant_augmentation(
    data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate the augmented dataset and return statistics.

    Args:
        data: Augmented dataset

    Returns:
        Statistics about the augmentation
    """
    # Count samples by source
    sources = {}
    labels = {}

    for sample in data:
        source = sample.get('source', 'unknown')
        label = sample.get('label', 'unknown')

        sources[source] = sources.get(source, 0) + 1
        labels[label] = labels.get(label, 0) + 1

    stats = {
        'total_samples': len(data),
        'by_source': sources,
        'by_label': labels,
        'qdrant_samples': sources.get('qdrant', 0),
        'qdrant_percentage': sources.get('qdrant', 0) / len(data) * 100 if data else 0
    }

    logger.info(f"""
╔════════════════════════════════════════════════════════╗
║         QDRANT AUGMENTATION STATISTICS                 ║
╠════════════════════════════════════════════════════════╣
║  Total Samples:        {stats['total_samples']:>6}                       ║
║  From Qdrant:          {stats['qdrant_samples']:>6} ({stats['qdrant_percentage']:>5.1f}%)              ║
║                                                        ║
║  Label Distribution:                                   ║
║    Malicious:          {labels.get('malicious', 0):>6}                       ║
║    Benign:             {labels.get('benign', 0):>6}                       ║
╚════════════════════════════════════════════════════════╝
    """)

    return stats


__all__ = [
    'augment_with_qdrant_patterns',
    'validate_qdrant_augmentation'
]
