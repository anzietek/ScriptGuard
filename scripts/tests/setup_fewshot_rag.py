#!/usr/bin/env python3
"""
Quick Setup Script for Few-Shot RAG
Automates the setup and testing of Code Similarity Search functionality.
"""

import sys
import subprocess
import yaml

from scriptguard.utils.logger import logger


def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")

    required_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("qdrant_client", "qdrant-client"),
        ("sentence_transformers", "sentence-transformers")
    ]

    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            logger.warning(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: uv pip install " + " ".join(missing))
        return False

    return True


def check_qdrant():
    """Check if Qdrant is running."""
    logger.info("Checking Qdrant connection...")

    try:
        from qdrant_client import QdrantClient

        with open("../../config.yaml", "r") as f:
            config = yaml.safe_load(f)

        qdrant_config = config.get("qdrant", {})
        host = qdrant_config.get("host", "localhost")
        port = qdrant_config.get("port", 6333)

        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()

        logger.info(f"  ✓ Qdrant running at {host}:{port}")
        logger.info(f"  Collections: {len(collections.collections)}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Qdrant connection failed: {e}")
        logger.info("  Start Qdrant with Docker:")
        logger.info("    docker run -p 6333:6333 qdrant/qdrant")
        return False


def check_postgresql():
    """Check if PostgreSQL has data."""
    logger.info("Checking PostgreSQL database...")

    try:
        from scriptguard.database.dataset_manager import DatasetManager

        db = DatasetManager()
        malicious = db.get_all_samples(label="malicious", limit=1)
        benign = db.get_all_samples(label="benign", limit=1)

        if not malicious or not benign:
            logger.error("  ✗ PostgreSQL is empty!")
            logger.info("  Run data ingestion first:")
            logger.info("    python -m scriptguard.pipelines.training_pipeline")
            return False

        logger.info(f"  ✓ PostgreSQL has data")
        return True

    except Exception as e:
        logger.error(f"  ✗ PostgreSQL connection failed: {e}")
        return False


def run_vectorization():
    """Run vectorization process."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING VECTORIZATION")
    logger.info("="*60)

    try:
        from scriptguard.steps.vectorize_samples import vectorize_samples

        with open("../../config.yaml", "r") as f:
            config = yaml.safe_load(f)

        result = vectorize_samples(
            config=config,
            clear_existing=True
            # max_samples not specified = vectorize all samples
        )

        if result["status"] == "success":
            logger.info("\n✓ Vectorization successful!")
            logger.info(f"  Total samples: {result['samples_vectorized']}")
            logger.info(f"  Malicious: {result['malicious_count']}")
            logger.info(f"  Benign: {result['benign_count']}")
            return True
        else:
            logger.error(f"\n✗ Vectorization failed: {result}")
            return False

    except Exception as e:
        logger.error(f"\n✗ Vectorization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_test():
    """Run test script."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING TESTS")
    logger.info("="*60)

    try:
        result = subprocess.run(
            [sys.executable, "test_fewshot_rag.py"],
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            logger.info("\n✓ All tests passed!")
            return True
        else:
            logger.error(f"\n✗ Tests failed with code {result.returncode}")
            return False

    except Exception as e:
        logger.error(f"\n✗ Test execution error: {e}")
        return False


def main():
    """Main setup workflow."""
    logger.info("="*60)
    logger.info("FEW-SHOT RAG SETUP WIZARD")
    logger.info("="*60)
    logger.info("")

    # Step 1: Dependencies
    logger.info("Step 1/5: Checking dependencies...")
    if not check_dependencies():
        logger.error("\n❌ Setup failed: Missing dependencies")
        return 1

    # Step 2: Qdrant
    logger.info("\nStep 2/5: Checking Qdrant...")
    if not check_qdrant():
        logger.error("\n❌ Setup failed: Qdrant not available")
        return 1

    # Step 3: PostgreSQL
    logger.info("\nStep 3/5: Checking PostgreSQL...")
    if not check_postgresql():
        logger.error("\n❌ Setup failed: PostgreSQL empty")
        return 1

    # Step 4: Vectorization
    logger.info("\nStep 4/5: Running vectorization...")
    if not run_vectorization():
        logger.error("\n❌ Setup failed: Vectorization error")
        return 1

    # Step 5: Test
    logger.info("\nStep 5/5: Running tests...")
    if not run_test():
        logger.error("\n❌ Setup failed: Tests failed")
        return 1

    # Success
    logger.info("\n" + "="*60)
    logger.info("✅ FEW-SHOT RAG SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("\nYou can now:")
    logger.info("1. Run training with Few-Shot RAG:")
    logger.info("   python -m scriptguard.pipelines.training_pipeline")
    logger.info("")
    logger.info("2. Use the API with RAG-enhanced inference:")
    logger.info("   uvicorn scriptguard.api.main:app --reload")
    logger.info("")
    logger.info("3. Check the documentation:")
    logger.info("   docs/FEW_SHOT_RAG_GUIDE.md")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Setup failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
