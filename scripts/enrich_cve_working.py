#!/usr/bin/env python
"""
WORKING CVE Enrichment - uses simple logging like test_enrich_simple.py
This version actually works because it doesn't import scriptguard.utils.logger first.
"""

import sys
import logging
from pathlib import Path

# Setup simple logging FIRST (before any scriptguard imports)
log_file = Path(__file__).parent.parent / "cve_enrichment.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# NOW import scriptguard modules
from scriptguard.data_sources.cve_feeds import CVEFeedSource
from scriptguard.rag.qdrant_store import QdrantStore
import os
import yaml


def load_config():
    """
    Load configuration from config.yaml and substitute environment variables.

    Supports syntax: ${ENV_VAR:-default_value} or ${ENV_VAR}
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
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
    logger.info("="*70)
    logger.info("WORKING CVE ENRICHMENT FOR QDRANT")
    logger.info("="*70)

    # Load config
    try:
        config = load_config()
        qdrant_config = config.get("qdrant", {})
        cve_config = config.get("data_sources", {}).get("cve_feeds", {})
        api_keys = config.get("api_keys", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Check if CVE feeds are enabled
    if not cve_config.get("enabled", False):
        logger.warning("CVE feeds are disabled in config.yaml")
        return 1

    # Initialize Qdrant store
    logger.info("\n1️⃣ Connecting to Qdrant...")
    try:
        store = QdrantStore(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
            embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        initial_info = store.get_collection_info()
        initial_count = initial_info.get('points_count', 0)
        logger.info(f"✅ Connected! Initial collection size: {initial_count} points")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        return 1

    # Initialize CVE source
    logger.info("\n2️⃣ Initializing CVE source...")
    nvd_api_key = api_keys.get("nvd_api_key")
    if nvd_api_key:
        logger.info("Using NVD API key")
    else:
        logger.info("No NVD API key (rate limits will be lower)")

    cve_source = CVEFeedSource(api_key=nvd_api_key)

    # Fetch exploit patterns
    logger.info("\n3️⃣ Adding exploit patterns...")
    try:
        patterns = cve_source.fetch_exploit_patterns()
        pattern_data = []
        for pattern in patterns:
            pattern_data.append({
                "cve_id": ", ".join(pattern.get("cve_examples", [])),
                "description": pattern["description"],
                "severity": pattern["severity"],
                "pattern": pattern["pattern"],
                "type": "exploit_pattern"
            })

        if pattern_data:
            store.upsert_vulnerabilities(pattern_data)
            logger.info(f"✅ Added {len(pattern_data)} exploit patterns")
    except Exception as e:
        logger.error(f"❌ Failed to add exploit patterns: {e}")

    # Fetch real CVE data from NVD
    logger.info("\n4️⃣ Fetching CVE data from NVD API...")
    days_back = cve_config.get("days_back", 30)
    keywords = cve_config.get("keywords", ["script", "code execution"])

    logger.info(f"   Days back: {days_back}")
    logger.info(f"   Keywords: {keywords}")

    try:
        cves = cve_source.fetch_recent_cves(days=days_back, keywords=keywords)

        if cves:
            logger.info(f"✅ Found {len(cves)} CVEs from NVD")

            # Convert CVEs to Qdrant format
            cve_data = []
            for cve in cves:
                cve_data.append({
                    "cve_id": cve["cve_id"],
                    "description": cve["description"],
                    "severity": cve.get("severity", "UNKNOWN"),
                    "pattern": "",
                    "type": "cve",
                    "cvss_score": cve.get("cvss_score"),
                    "published": cve.get("published")
                })

            # Add to Qdrant in batches
            logger.info(f"   Adding {len(cve_data)} CVEs to Qdrant in batches...")
            batch_size = 50
            added = 0
            for i in range(0, len(cve_data), batch_size):
                batch = cve_data[i:i+batch_size]
                try:
                    store.upsert_vulnerabilities(batch)
                    added += len(batch)
                    logger.info(f"   Batch {i//batch_size + 1}: Added {len(batch)} CVEs (total: {added}/{len(cve_data)})")
                except Exception as batch_error:
                    logger.error(f"   Batch {i//batch_size + 1} failed: {batch_error}")

            logger.info(f"✅ Successfully added {added} real CVEs from NVD")
        else:
            logger.warning("⚠️  No CVEs found. Try adjusting keywords or date range.")
    except Exception as e:
        logger.error(f"❌ Failed to fetch CVE data: {e}", exc_info=True)

    # Add synthetic exploit samples
    logger.info("\n5️⃣ Adding synthetic exploit samples...")
    try:
        samples = cve_source.get_exploit_pattern_samples()
        sample_data = []
        for sample in samples:
            metadata = sample.get("metadata", {})
            sample_data.append({
                "cve_id": "",
                "description": f"{metadata.get('description', 'Synthetic')}: {metadata.get('pattern', '')}",
                "severity": metadata.get('severity', 'HIGH'),
                "pattern": sample["content"],
                "type": "synthetic_exploit"
            })

        if sample_data:
            store.upsert_vulnerabilities(sample_data)
            logger.info(f"✅ Added {len(sample_data)} synthetic exploit samples")
    except Exception as e:
        logger.error(f"❌ Failed to add synthetic samples: {e}")

    # Final statistics
    logger.info("\n" + "="*70)
    final_info = store.get_collection_info()
    final_count = final_info.get('points_count', 0)
    added_count = final_count - initial_count

    logger.info("ENRICHMENT COMPLETE!")
    logger.info(f"  Initial:  {initial_count:>6} points")
    logger.info(f"  Final:    {final_count:>6} points")
    logger.info(f"  Added:    {added_count:>6} points")
    logger.info("="*70)
    logger.info(f"\nFull log saved to: {log_file.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
