"""
Qdrant-based Data Augmentation Step
Enriches training data with CVE patterns and malware signatures from Qdrant vector store.
Enhanced with sanitization and context injection.
"""

from typing import List, Dict, Any, Optional
from zenml import step
from scriptguard.utils.logger import logger
from scriptguard.rag.code_sanitization import create_sanitizer, create_enricher


@step
def augment_with_qdrant_patterns(
    data: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Augment training data with patterns from Qdrant.

    Retrieves samples from TWO Qdrant collections:
    1. malware_knowledge - CVE patterns and vulnerability signatures
    2. code_samples - Few-Shot code examples from database

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

    all_augmented_samples = []

    try:
        qdrant_config = config.get("qdrant", {})

        # ========================================
        # COLLECTION 1: malware_knowledge (CVE patterns)
        # ========================================
        logger.info("Fetching CVE patterns from 'malware_knowledge' collection...")

        try:
            from scriptguard.rag.qdrant_store import QdrantStore

            store = QdrantStore(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333),
                collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
                embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2"),
                api_key=qdrant_config.get("api_key"),  # Pass directly, QdrantStore will handle env vars
                use_https=qdrant_config.get("use_https", False)
            )

            # Get collection info
            info = store.get_collection_info()
            points_count = info.get('points_count', 0)

            if points_count > 0:
                logger.info(f"Found {points_count} CVE patterns in 'malware_knowledge'")

                # Scroll through all points
                offset = None
                batch_size = 100
                cve_added = 0

                while True:
                    scroll_result = store.client.scroll(
                        collection_name=store.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )

                    points, next_offset = scroll_result

                    if not points:
                        break

                    # Convert Qdrant points to training samples
                    for point in points:
                        payload = point.payload
                        if not payload.get('description'):
                            continue

                        sample = _create_sample_from_pattern(payload, augmentation_config)
                        if sample:
                            all_augmented_samples.append(sample)
                            cve_added += 1

                    if next_offset is None:
                        break

                    offset = next_offset

                logger.info(f"✓ Added {cve_added} CVE patterns from 'malware_knowledge'")
            else:
                logger.warning("'malware_knowledge' collection is empty")

        except Exception as e:
            logger.warning(f"Failed to fetch from 'malware_knowledge': {e}")

        # ========================================
        # COLLECTION 2: code_samples (Few-Shot examples)
        # ========================================
        logger.info("Fetching code samples from 'code_samples' collection...")

        try:
            from scriptguard.rag.code_similarity_store import CodeSimilarityStore
            from scriptguard.database.dataset_manager import DatasetManager

            # Get config path from environment or use passed config value
            import os
            default_config_path = os.getenv("CONFIG_PATH", "config.yaml")
            config_path = config.get("config_path", default_config_path)

            code_store = CodeSimilarityStore(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333),
                collection_name="code_samples",  # Fixed collection name
                enable_chunking=False,  # No chunking for augmentation
                config_path=config_path
            )

            # Get collection info
            info = code_store.get_collection_info()
            points_count = info.get('total_samples', 0)

            if points_count > 0:
                logger.info(f"Found {points_count} code samples in 'code_samples'")

                # Initialize DB manager to fetch full content
                db_manager = DatasetManager()

                # Collect unique db_ids first
                db_ids_to_fetch = set()
                offset = None
                batch_size = 100

                while True:
                    scroll_result = code_store.client.scroll(
                        collection_name="code_samples",
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )

                    points, next_offset = scroll_result

                    if not points:
                        break

                    # Collect db_ids
                    for point in points:
                        payload = point.payload
                        db_id = payload.get('db_id')
                        if db_id and payload.get('chunk_index', 0) == 0:  # Only first chunk per doc
                            db_ids_to_fetch.add(db_id)

                    if next_offset is None:
                        break

                    offset = next_offset

                # Batch fetch full content from PostgreSQL
                logger.info(f"Fetching full content for {len(db_ids_to_fetch)} unique documents...")

                from scriptguard.database.db_schema import get_connection

                conn = get_connection()
                cursor = conn.cursor()

                placeholders = ','.join(['%s'] * len(db_ids_to_fetch))
                query = f"""
                    SELECT id, content, label, source, metadata
                    FROM samples
                    WHERE id IN ({placeholders})
                """

                cursor.execute(query, tuple(db_ids_to_fetch))
                rows = cursor.fetchall()

                # Initialize sanitizer and enricher (if enabled in config)
                code_emb_config = config.get("code_embedding", {})

                sanitization_config = code_emb_config.get("sanitization", {})
                sanitizer = None
                if sanitization_config.get("enabled", True):
                    sanitizer = create_sanitizer(sanitization_config)
                    logger.info("  Sanitization enabled for code samples")

                context_injection_config = code_emb_config.get("context_injection", {})
                enricher = None
                if context_injection_config.get("enabled", True):
                    enricher = create_enricher(context_injection_config)
                    logger.info("  Context injection enabled for code samples")

                code_added = 0
                code_rejected = 0
                rejection_reasons = {}

                for row in rows:
                    raw_content = row['content']

                    # SANITIZATION PASS
                    if sanitizer:
                        cleaned_content, report = sanitizer.sanitize(
                            content=raw_content,
                            language="python",
                            metadata=row.get('metadata', {})
                        )

                        if not report.get("valid", False):
                            code_rejected += 1
                            reason = report.get("reason", "unknown")
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            logger.debug(f"Rejected code sample {row['id']}: {reason}")
                            continue

                        processed_content = cleaned_content
                    else:
                        processed_content = raw_content

                    # CONTEXT INJECTION PASS
                    if enricher:
                        enrichment_metadata = {
                            "file_path": row.get('metadata', {}).get('file_path'),
                            "repository": row.get('metadata', {}).get('repository'),
                            "language": "python",
                            "source": row['source'],
                            "label": row['label']
                        }
                        processed_content = enricher.enrich(processed_content, enrichment_metadata)

                    sample = {
                        'content': processed_content,  # SANITIZED + ENRICHED content
                        'label': row['label'],
                        'source': 'qdrant_code_samples',
                        'metadata': {
                            'db_id': row['id'],
                            'original_source': row['source'],
                            'db_metadata': row['metadata'] or {},
                            'sanitized': bool(sanitizer),
                            'enriched': bool(enricher)
                        }
                    }

                    all_augmented_samples.append(sample)
                    code_added += 1

                cursor.close()
                conn.close()

                logger.info(
                    f"✓ Added {code_added} code samples with full content from PostgreSQL "
                    f"({code_rejected} rejected by sanitization)"
                )

                if rejection_reasons:
                    logger.info("  Rejection reasons:")
                    for reason, count in rejection_reasons.items():
                        logger.info(f"    - {reason}: {count}")
            else:
                logger.warning("'code_samples' collection is empty")

        except Exception as e:
            logger.warning(f"Failed to fetch from 'code_samples': {e}", exc_info=True)

        # Combine all
        if all_augmented_samples:
            logger.info(f"✓ Total augmented: {len(all_augmented_samples)} samples from Qdrant")
        else:
            logger.warning("No samples retrieved from any Qdrant collection")

        combined_data = data + all_augmented_samples
        logger.info(f"Final dataset size: {len(combined_data)} samples")

        return combined_data

    except Exception as e:
        logger.error(f"Qdrant augmentation failed: {e}", exc_info=True)
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

    # Calculate Qdrant totals
    qdrant_cve = sources.get('qdrant', 0)
    qdrant_code = sources.get('qdrant_code_samples', 0)
    qdrant_total = qdrant_cve + qdrant_code

    stats = {
        'total_samples': len(data),
        'by_source': sources,
        'by_label': labels,
        'qdrant_samples_total': qdrant_total,
        'qdrant_cve_patterns': qdrant_cve,
        'qdrant_code_samples': qdrant_code,
        'qdrant_percentage': qdrant_total / len(data) * 100 if data else 0
    }

    logger.info(f"""
╔════════════════════════════════════════════════════════╗
║         QDRANT AUGMENTATION STATISTICS                 ║
╠════════════════════════════════════════════════════════╣
║  Total Samples:        {stats['total_samples']:>6}                       ║
║  From Qdrant:          {stats['qdrant_samples_total']:>6} ({stats['qdrant_percentage']:>5.1f}%)              ║
║    - CVE patterns:     {stats['qdrant_cve_patterns']:>6}                       ║
║    - Code samples:     {stats['qdrant_code_samples']:>6}                       ║
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
