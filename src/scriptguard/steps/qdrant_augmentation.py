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
                api_key=qdrant_config.get("api_key"),  # Pass API key
                config_path=config_path
            )

            # Log connection information for diagnostics
            connection_info = code_store.get_connection_info()
            logger.info("=" * 60)
            logger.info("AUGMENTATION CONNECTION INFO")
            logger.info(f"  Type: {connection_info['connection_type']}")
            logger.info(f"  URL: {connection_info['connection_url']}")
            logger.info(f"  Collection: {connection_info['collection_name']}")
            logger.info(f"  API key: {'SET' if connection_info['has_api_key'] else 'NOT SET'}")
            logger.info("=" * 60)

            # Get collection info with enhanced error handling
            try:
                info = code_store.get_collection_info()
                points_count = info.get('total_samples', 0)

                logger.info(f"Collection status:")
                logger.info(f"  Total points: {points_count}")
                logger.info(f"  Status: {info.get('status', 'unknown')}")

            except Exception as e:
                logger.error("=" * 60)
                logger.error("❌ COLLECTION ACCESS FAILED")
                logger.error(f"Collection: {code_store.collection_name}")
                logger.error(f"Error: {e}")
                logger.error("")
                logger.error("Possible causes:")
                logger.error("  1. Connection mismatch between vectorization and augmentation")
                logger.error("  2. Wrong API key or credentials")
                logger.error("  3. Collection not created - run vectorization first")
                logger.error(f"Connection: {connection_info}")
                logger.error("=" * 60)
                raise

            # Warn if collection is empty
            if points_count == 0:
                logger.warning("=" * 60)
                logger.warning("⚠️ EMPTY COLLECTION DETECTED")
                logger.warning(f"Collection '{code_store.collection_name}' has 0 points")
                logger.warning("")
                logger.warning("Troubleshooting:")
                logger.warning("  1. Check if vectorization completed successfully")
                logger.warning("  2. Verify connection settings match between steps:")
                logger.warning(f"     - Type: {connection_info['connection_type']}")
                logger.warning(f"     - URL: {connection_info['connection_url']}")
                logger.warning("  3. Review QDRANT_API_KEY environment variable")
                logger.warning("=" * 60)

            if points_count > 0:
                logger.info(f"Found {points_count} code samples in 'code_samples'")

                # Initialize DB manager to fetch full content
                db_manager = DatasetManager()

                # Collect unique db_ids first
                db_ids_to_fetch = set()
                offset = None
                batch_size = 100

                # Initialize tracking for debugging
                total_points_scrolled = 0
                chunk_index_distribution = {}
                points_with_db_id = 0
                points_with_null_db_id = 0  # Track synthetic/augmented samples
                points_missing_chunk_index = 0

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

                    total_points_scrolled += len(points)

                    # Collect db_ids with defensive type handling
                    for point in points:
                        payload = point.payload
                        db_id = payload.get('db_id')

                        if not db_id:
                            points_with_null_db_id += 1
                            continue

                        points_with_db_id += 1

                        # Normalize db_id to integer (handle string/int types)
                        try:
                            if isinstance(db_id, str):
                                db_id = int(db_id)
                            elif not isinstance(db_id, int):
                                logger.debug(f"Unexpected db_id type: {type(db_id)} = {db_id}")
                                db_id = int(db_id)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to convert db_id to int: {db_id} ({type(db_id)})")
                            continue

                        # Get chunk_index with type safety
                        chunk_idx = payload.get('chunk_index')

                        # Track distribution
                        chunk_index_distribution[chunk_idx] = chunk_index_distribution.get(chunk_idx, 0) + 1

                        if chunk_idx is None:
                            points_missing_chunk_index += 1
                            # If no chunk_index, assume it's a single document (treat as chunk 0)
                            db_ids_to_fetch.add(db_id)
                            continue

                        # Type-safe comparison: handle both int and string
                        is_first_chunk = False
                        try:
                            # Try int comparison first
                            if isinstance(chunk_idx, int) and chunk_idx == 0:
                                is_first_chunk = True
                            # Handle string type
                            elif isinstance(chunk_idx, str) and chunk_idx == "0":
                                is_first_chunk = True
                            # Handle numeric string
                            elif str(chunk_idx) == "0":
                                is_first_chunk = True
                        except (ValueError, TypeError):
                            logger.debug(f"Invalid chunk_index type: {type(chunk_idx)} = {chunk_idx}")
                            continue

                        if is_first_chunk:
                            db_ids_to_fetch.add(db_id)

                    if next_offset is None:
                        break

                    offset = next_offset

                # Log collection statistics
                logger.info(f"Scrolled {total_points_scrolled} points from code_samples collection")
                logger.info(f"Points with db_id: {points_with_db_id}")
                logger.info(f"Points with NULL db_id (synthetic): {points_with_null_db_id}")
                logger.info(f"Points missing chunk_index: {points_missing_chunk_index}")
                logger.info(f"Chunk index distribution (top 10): {dict(sorted(chunk_index_distribution.items(), key=lambda x: x[1], reverse=True)[:10])}")
                logger.info(f"Collected {len(db_ids_to_fetch)} unique db_ids for augmentation (from database samples only)")

                # CRITICAL: Validate db_ids_to_fetch is not empty
                if not db_ids_to_fetch:
                    logger.warning(
                        f"❌ No documents with chunk_index==0 found in code_samples collection!\n"
                        f"   Total points in collection: {points_count}\n"
                        f"   Points scrolled: {total_points_scrolled}\n"
                        f"   Points with db_id: {points_with_db_id}\n"
                        f"   Chunk index distribution: {chunk_index_distribution}\n"
                        f"   This indicates a data integrity issue - all chunks have index > 0.\n"
                        f"   Skipping code sample augmentation."
                    )
                    # Skip to avoid SQL syntax error with empty IN clause
                    logger.warning("'code_samples' collection has no valid first chunks to fetch")
                else:
                    logger.info(f"✓ Found {len(db_ids_to_fetch)} unique documents to augment")

                # Only proceed if we have IDs to fetch
                if db_ids_to_fetch:
                    # Batch fetch full content from PostgreSQL
                    logger.info(f"Fetching full content for {len(db_ids_to_fetch)} unique documents...")

                    # Debug: Log sample of db_ids
                    sample_ids = list(db_ids_to_fetch)[:10]
                    logger.debug(f"Sample db_ids (first 10): {sample_ids}")
                    logger.debug(f"Sample db_id types: {[type(x).__name__ for x in sample_ids[:3]]}")

                    from scriptguard.database.db_schema import get_connection

                    conn = get_connection()
                    cursor = conn.cursor()

                    # Verify database has data
                    cursor.execute("SELECT COUNT(*) as count, MIN(id) as min_id, MAX(id) as max_id FROM samples")
                    db_stats = cursor.fetchone()
                    logger.info(f"PostgreSQL database stats: {db_stats['count']} total samples, ID range: {db_stats['min_id']} to {db_stats['max_id']}")

                    placeholders = ','.join(['%s'] * len(db_ids_to_fetch))
                    query = f"""
                        SELECT id, content, label, source, metadata
                        FROM samples
                        WHERE id IN ({placeholders})
                    """

                    logger.debug(f"Executing SQL query with {len(db_ids_to_fetch)} IDs...")
                    cursor.execute(query, tuple(db_ids_to_fetch))
                    rows = cursor.fetchall()

                    logger.info(f"SQL query returned {len(rows)} rows from PostgreSQL")

                    if len(rows) == 0 and len(db_ids_to_fetch) > 0:
                        logger.error("=" * 60)
                        logger.error("❌ ZERO ROWS RETURNED FROM DATABASE")
                        logger.error(f"Expected: {len(db_ids_to_fetch)} rows")
                        logger.error(f"Got: 0 rows")
                        logger.error(f"Sample IDs queried: {sample_ids}")
                        logger.error(f"Database has IDs from {db_stats['min_id']} to {db_stats['max_id']}")
                        logger.error("")
                        logger.error("Possible causes:")
                        logger.error("  1. db_ids in Qdrant don't match PostgreSQL IDs")
                        logger.error("     → Check if vectorization used actual DB IDs or generated hashes")
                        logger.error("  2. Data was deleted from PostgreSQL after vectorization")
                        logger.error("  3. Need to re-vectorize data with updated code")
                        logger.error("")
                        logger.error("Solution: Clear Qdrant collection and re-run vectorization:")
                        logger.error("  1. Delete the collection in Qdrant")
                        logger.error("  2. Set ENABLE_QDRANT_VECTORIZATION=true")
                        logger.error("  3. Run the training pipeline again")
                        logger.error("=" * 60)

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
            logger.error(
                f"Failed to fetch from 'code_samples' collection: {e}\n"
                f"  Points scrolled: {total_points_scrolled if 'total_points_scrolled' in locals() else 'N/A'}\n"
                f"  DB IDs collected: {len(db_ids_to_fetch) if 'db_ids_to_fetch' in locals() else 'N/A'}\n"
                f"  Consider checking Qdrant data integrity and chunk_index values.",
                exc_info=True
            )

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
