#!/usr/bin/env python
"""
Bootstrap Qdrant with initial CVE and malware pattern data.
Run this script to initialize the Qdrant vector store with knowledge base.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.rag.qdrant_store import QdrantStore, bootstrap_cve_data
from scriptguard.utils.logger import logger
import os
import yaml


def load_config():
    """
    Load configuration from config.yaml and substitute environment variables.

    Supports syntax: ${ENV_VAR:-default_value} or ${ENV_VAR}
    """
    config_filename = os.getenv("CONFIG_PATH", "config.yaml")
    config_path = Path(__file__).parent.parent / config_filename
    with open(config_path) as f:
        config = yaml.safe_load(f)

    def substitute_env_vars(obj):
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_expr = obj[2:-1]
            if ":-" in env_expr:
                env_var, default = env_expr.split(":-", 1)
                return os.getenv(env_var, default)
            else:
                env_var = env_expr
                return os.getenv(env_var, "")
        else:
            return obj

    return substitute_env_vars(config)


def main():
    logger.info("Starting Qdrant bootstrap...")

    # Load config
    try:
        config = load_config()
        qdrant_config = config.get("qdrant", {})
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        qdrant_config = {}

    # Initialize Qdrant store
    try:
        store = QdrantStore(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
            embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2"),
            api_key=qdrant_config.get("api_key"),
            use_https=qdrant_config.get("use_https", False)
        )
        logger.info("✅ Qdrant store initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant store: {e}")
        logger.error("Make sure Qdrant is running: docker-compose up -d")
        return 1

    # Bootstrap with CVE data
    try:
        bootstrap_cve_data(store)
        logger.info("✅ CVE data bootstrapped successfully")
    except Exception as e:
        logger.error(f"❌ Failed to bootstrap CVE data: {e}")
        return 1

    # Get collection info
    try:
        info = store.get_collection_info()
        logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║              QDRANT COLLECTION INFO                      ║
╠══════════════════════════════════════════════════════════╣
║  Points Count:    {info.get('points_count', 0):>5}                             ║
║  Vectors Count:   {info.get('vectors_count', 0):>5}                             ║
║  Indexed Vectors: {info.get('indexed_vectors_count', 0):>5}                             ║
║  Status:          {str(info.get('status', 'unknown')):<30} ║
╚══════════════════════════════════════════════════════════╝
        """)
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")

    # Test search
    logger.info("\n🔍 Testing search functionality...")
    try:
        test_queries = [
            "remote code execution python",
            "command injection vulnerability",
            "base64 encoded payload"
        ]

        for query in test_queries:
            results = store.search(query, limit=2)
            logger.info(f"\nQuery: '{query}'")
            if results:
                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. [Score: {result['score']:.3f}] {result['payload']['description'][:80]}")
            else:
                logger.warning(f"  No results found")
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return 1

    logger.info("\n✅ Qdrant bootstrap completed successfully!")
    logger.info("\nYou can now:")
    logger.info("  1. Start training: uv run python src/main.py")
    logger.info("  2. Start API: uv run uvicorn scriptguard.api.main:app --reload")
    logger.info("  3. View Qdrant dashboard: http://localhost:6333/dashboard")

    return 0


if __name__ == "__main__":
    sys.exit(main())
