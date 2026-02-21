#!/usr/bin/env python3
"""
Create feature indexes on existing Qdrant collection.

This script creates payload indexes for static features to enable
efficient hybrid search (vector similarity + feature filtering).

Usage:
    python scripts/create_feature_indexes.py --collection code_samples
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from scriptguard.config_loader import load_raw_config as load_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Create feature indexes on Qdrant collection")
    parser.add_argument(
        "--collection",
        type=str,
        default="code_samples",
        help="Qdrant collection name (default: code_samples)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify indexes were created successfully"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    qdrant_config = config.get("qdrant", {})

    logger.info("=" * 60)
    logger.info("FEATURE INDEX CREATION")
    logger.info("=" * 60)
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Host: {qdrant_config.get('host', 'localhost')}")
    logger.info(f"Port: {qdrant_config.get('port', 6333)}")

    # Initialize store
    try:
        store = CodeSimilarityStore(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=args.collection,
            api_key=qdrant_config.get("api_key"),
            use_https=qdrant_config.get("use_https", False),
            timeout=qdrant_config.get("timeout", 60)
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return 1

    # Create indexes
    logger.info("\nCreating feature indexes...")
    try:
        store._create_feature_indexes()
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        return 1

    # Verify if requested
    if args.verify:
        logger.info("\nVerifying indexes...")
        try:
            collection_info = store.client.get_collection(args.collection)
            payload_schema = collection_info.config.params.payload_schema if hasattr(collection_info.config.params, 'payload_schema') else None

            if payload_schema:
                logger.info("Payload schema:")
                for field, schema in payload_schema.items():
                    logger.info(f"  - {field}: {schema}")
            else:
                logger.warning("Payload schema not available (indexes may still be created)")

            # Test a sample query with feature filter
            logger.info("\nTesting feature filter query...")
            results = store.client.scroll(
                collection_name=args.collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="features.entropy",
                            range=models.Range(gte=0.0)
                        )
                    ]
                ),
                limit=1
            )

            if results[0]:
                sample = results[0][0]
                logger.info("✓ Feature filter query successful")
                logger.info(f"  Sample features: {sample.payload.get('features', {})}")
            else:
                logger.warning("⚠️  No results with feature filters (collection may be empty or features not yet added)")

        except Exception as e:
            logger.warning(f"Verification failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ INDEX CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Verify indexes in Qdrant UI: http://localhost:6333/dashboard")
    logger.info("2. Re-vectorize samples if they don't have features yet")
    logger.info("3. Test hybrid search with feature filters")

    return 0


if __name__ == "__main__":
    from qdrant_client import models
    sys.exit(main())
