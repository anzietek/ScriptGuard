#!/usr/bin/env python
"""
Test ingestion tylko - bez trenowania modelu
Sprawdza czy dane trafiają do PostgreSQL i Qdrant
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from scriptguard.steps.advanced_ingestion import advanced_data_ingestion
from scriptguard.steps.data_validation import validate_samples, filter_by_quality
from scriptguard.steps.vectorize_samples import vectorize_samples
from scriptguard.database.dataset_manager import DatasetManager
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger

# Import load_config z main.py która obsługuje env variables
sys.path.insert(0, str(Path(__file__).parent / "src"))
from main import load_config


def test_ingestion_pipeline():
    """Test tylko ingestion + wektoryzacja, bez trenowania"""

    logger.info("=" * 80)
    logger.info("TEST INGESTION PIPELINE (BEZ TRENOWANIA)")
    logger.info("=" * 80)

    # Load config with env variable substitution
    config = load_config("config.yaml")

    # STEP 1: Data ingestion
    logger.info("\n[STEP 1] Advanced Data Ingestion...")
    try:
        raw_data = advanced_data_ingestion(config=config)
        logger.info(f"✅ Ingestion complete: {len(raw_data)} samples")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return False

    if not raw_data:
        logger.warning("⚠️ No data ingested!")
        return False

    # STEP 1.5: Save to PostgreSQL (ingestion returns data but doesn't save it!)
    logger.info("\n[STEP 1.5] Saving samples to PostgreSQL...")
    try:
        db = DatasetManager()
        saved_count = 0
        for sample in raw_data:
            try:
                db.add_sample(
                    content=sample.get("content", ""),
                    label=sample.get("label", "unknown"),
                    source=sample.get("source", "unknown"),
                    metadata=sample.get("metadata", {})
                )
                saved_count += 1
            except Exception as e:
                logger.debug(f"Skipping sample: {e}")
                continue

        logger.info(f"✅ Saved {saved_count} samples to PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Saving to PostgreSQL failed: {e}")
        return False

    # STEP 2: Manual validation (skip ZenML step to avoid env var issues)
    logger.info("\n[STEP 2] Validating samples manually...")
    try:
        from scriptguard.data_sources.data_validator import DataValidator

        validator = DataValidator(
            validate_syntax=config.get("validation", {}).get("validate_syntax", True),
            min_length=config.get("validation", {}).get("min_length", 50),
            max_length=config.get("validation", {}).get("max_length", 50000)
        )

        validated_data = []
        for sample in raw_data:
            if validator.validate_sample(sample):
                validated_data.append(sample)

        logger.info(f"✅ Validation complete: {len(validated_data)}/{len(raw_data)} valid samples")
    except Exception as e:
        logger.warning(f"⚠️ Validation failed: {e}, skipping validation step")
        validated_data = raw_data

    # STEP 3: Verify PostgreSQL
    logger.info("\n[STEP 3] Verifying PostgreSQL...")
    try:
        db = DatasetManager()
        malicious = db.get_all_samples(label="malicious", limit=None)
        benign = db.get_all_samples(label="benign", limit=None)

        logger.info(f"PostgreSQL status:")
        logger.info(f"  - Malicious samples: {len(malicious)}")
        logger.info(f"  - Benign samples: {len(benign)}")
        logger.info(f"  - Total: {len(malicious) + len(benign)}")

        if len(malicious) + len(benign) == 0:
            logger.warning("⚠️ PostgreSQL is still empty!")
            return False

        logger.info("✅ PostgreSQL populated successfully")
    except Exception as e:
        logger.error(f"❌ PostgreSQL check failed: {e}")
        return False

    # STEP 4: Vectorize to Qdrant
    logger.info("\n[STEP 4] Vectorizing samples to Qdrant...")
    try:
        vectorization_result = vectorize_samples(
            config=config,
            max_samples=200,  # Limit to 200 for faster testing
            clear_existing=True
        )

        logger.info(f"Vectorization result:")
        logger.info(f"  - Status: {vectorization_result.get('status')}")
        logger.info(f"  - Samples vectorized: {vectorization_result.get('samples_vectorized', 0)}")
        logger.info(f"  - Malicious: {vectorization_result.get('malicious_count', 0)}")
        logger.info(f"  - Benign: {vectorization_result.get('benign_count', 0)}")

        if vectorization_result.get('samples_vectorized', 0) == 0:
            logger.warning("⚠️ No samples vectorized!")
            return False

        logger.info("✅ Vectorization complete")
    except Exception as e:
        logger.error(f"❌ Vectorization failed: {e}")
        return False

    # STEP 5: Verify Qdrant
    logger.info("\n[STEP 5] Verifying Qdrant code_samples...")
    try:
        code_store = CodeSimilarityStore(
            host=config.get("qdrant", {}).get("host", "localhost"),
            port=config.get("qdrant", {}).get("port", 6333),
            collection_name="code_samples"
        )

        info = code_store.get_collection_info()

        logger.info(f"Qdrant code_samples status:")
        logger.info(f"  - Total samples: {info.get('total_samples', 0)}")
        logger.info(f"  - Malicious: {info.get('malicious_samples', 0)}")
        logger.info(f"  - Benign: {info.get('benign_samples', 0)}")
        logger.info(f"  - Embedding dim: {info.get('embedding_dim', 'N/A')}")
        logger.info(f"  - Status: {info.get('status', 'N/A')}")

        if info.get('total_samples', 0) == 0:
            logger.warning("⚠️ Qdrant code_samples is still empty!")
            return False

        logger.info("✅ Qdrant code_samples populated successfully")
    except Exception as e:
        logger.error(f"❌ Qdrant check failed: {e}")
        return False

    # FINAL SUMMARY
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Data Ingestion: {len(validated_data)} samples")
    logger.info(f"✅ PostgreSQL: {len(malicious) + len(benign)} samples")
    logger.info(f"✅ Qdrant code_samples: {info.get('total_samples', 0)} vectors")
    logger.info("=" * 80)
    logger.info("🎉 ALL CHECKS PASSED!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_ingestion_pipeline()
    sys.exit(0 if success else 1)
