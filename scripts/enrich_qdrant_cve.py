#!/usr/bin/env python
"""
Enrich Qdrant with real CVE data from NVD API.
This script fetches CVE data and adds it to the malware_knowledge collection.
"""

import sys
import logging
from pathlib import Path

# Set up logging BEFORE importing ScriptGuard modules
# This ensures urllib3/requests debug logs work properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Clear any cached imports to ensure we use latest code
if 'scriptguard.data_sources.cve_feeds' in sys.modules:
    del sys.modules['scriptguard.data_sources.cve_feeds']
if 'scriptguard.rag.qdrant_store' in sys.modules:
    del sys.modules['scriptguard.rag.qdrant_store']

from scriptguard.rag.qdrant_store import QdrantStore
from scriptguard.data_sources.cve_feeds import CVEFeedSource
import yaml

# Use standard logging instead of scriptguard logger to avoid conflicts
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    logger.info("Starting Qdrant CVE enrichment...")

    # Load .env file for API keys
    import os
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded .env file")

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
        logger.info("Set data_sources.cve_feeds.enabled to true to enable")
        return 1

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
        logger.info("✅ Connected to Qdrant")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        return 1

    # Get initial collection info
    initial_info = store.get_collection_info()
    initial_count = initial_info.get('points_count', 0)
    logger.info(f"Initial collection size: {initial_count} points")

    # Initialize CVE source - prioritize env variable over config
    nvd_api_key = os.getenv("NVD_API_KEY") or api_keys.get("nvd_api_key")
    if nvd_api_key:
        logger.info(f"Using NVD API key: {nvd_api_key[:8]}...")
    else:
        logger.warning("No NVD API key found. Rate limits will be lower.")
        logger.info("Set NVD_API_KEY environment variable for higher rate limits")

    cve_source = CVEFeedSource(api_key=nvd_api_key)

    # ========================================
    # 1. Fetch exploit patterns (static)
    # ========================================
    logger.info("\n📋 Step 1: Adding exploit patterns...")
    try:
        patterns = cve_source.fetch_exploit_patterns()

        # Convert patterns to Qdrant format
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
        else:
            logger.warning("No exploit patterns found")

    except Exception as e:
        logger.error(f"❌ Failed to add exploit patterns: {e}")

    # ========================================
    # 2. Fetch real CVE data from NVD
    # ========================================
    logger.info("\n🌐 Step 2: Fetching CVE data from NVD API...")

    days_back = cve_config.get("days_back", 30)
    keywords = cve_config.get("keywords", [])

    logger.info(f"  Looking back {days_back} days")
    logger.info(f"  Keywords: {keywords}")

    try:
        cves = cve_source.fetch_recent_cves(
            days=days_back,
            keywords=keywords
        )

        if cves:
            # Convert CVEs to Qdrant format
            cve_data = []
            for cve in cves:
                cve_data.append({
                    "cve_id": cve["cve_id"],
                    "description": cve["description"],
                    "severity": cve.get("severity", "UNKNOWN"),
                    "pattern": "",  # No pattern from NVD, just descriptions
                    "type": "cve",
                    "cvss_score": cve.get("cvss_score"),
                    "published": cve.get("published"),
                    "references": cve.get("references", [])
                })

            # Add to Qdrant in batches
            batch_size = 50
            for i in range(0, len(cve_data), batch_size):
                batch = cve_data[i:i+batch_size]
                store.upsert_vulnerabilities(batch)
                logger.info(f"  Added batch {i//batch_size + 1}/{(len(cve_data) + batch_size - 1)//batch_size}")

            logger.info(f"✅ Added {len(cve_data)} real CVEs from NVD")
        else:
            logger.warning("No CVEs found matching criteria")
            logger.info("Try:")
            logger.info("  - Increasing days_back in config")
            logger.info("  - Broadening keywords")
            logger.info("  - Checking NVD API status")

    except Exception as e:
        logger.error(f"❌ Failed to fetch CVE data: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # 3. Add synthetic exploit samples
    # ========================================
    logger.info("\n🧪 Step 3: Adding synthetic exploit samples...")
    try:
        samples = cve_source.get_exploit_pattern_samples()

        # Convert samples to Qdrant format
        sample_data = []
        for sample in samples:
            metadata = sample.get("metadata", {})
            sample_data.append({
                "cve_id": "",
                "description": f"{metadata.get('description', 'Synthetic exploit pattern')}: {metadata.get('pattern', '')}",
                "severity": metadata.get('severity', 'HIGH'),
                "pattern": sample["content"],
                "type": "synthetic_exploit"
            })

        if sample_data:
            store.upsert_vulnerabilities(sample_data)
            logger.info(f"✅ Added {len(sample_data)} synthetic exploit samples")

    except Exception as e:
        logger.error(f"❌ Failed to add synthetic samples: {e}")

    # ========================================
    # Final statistics
    # ========================================
    logger.info("\n" + "="*60)
    final_info = store.get_collection_info()
    final_count = final_info.get('points_count', 0)
    added_count = final_count - initial_count

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║              QDRANT ENRICHMENT SUMMARY                   ║
╠══════════════════════════════════════════════════════════╣
║  Initial Count:   {initial_count:>5}                                 ║
║  Final Count:     {final_count:>5}                                 ║
║  Added:           {added_count:>5}                                 ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Test search
    logger.info("\n🔍 Testing search with new data...")
    test_queries = [
        "remote code execution",
        "command injection",
        "code execution vulnerability"
    ]

    for query in test_queries:
        results = store.search(query, limit=3)
        logger.info(f"\nQuery: '{query}'")
        if results:
            for i, result in enumerate(results, 1):
                payload = result['payload']
                desc = payload.get('description', 'No description')
                cve_id = payload.get('cve_id', 'N/A')
                logger.info(f"  {i}. [Score: {result['score']:.3f}] {cve_id}: {desc[:80]}...")
        else:
            logger.warning("  No results found")

    logger.info("\n✅ Qdrant CVE enrichment completed!")

    if added_count == 0:
        logger.warning("\n⚠️  No new data was added. Check:")
        logger.warning("  1. NVD API is accessible")
        logger.warning("  2. NVD_API_KEY is set (optional but recommended)")
        logger.warning("  3. CVE keywords match recent vulnerabilities")
        logger.warning("  4. Date range covers period with relevant CVEs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
