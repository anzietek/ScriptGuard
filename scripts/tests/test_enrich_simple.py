"""Simplified enrichment test that logs to file properly"""
import sys
import logging
from pathlib import Path

# Setup file logging FIRST
log_file = Path(__file__).parent / "enrich_simple.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger.info("="*60)
logger.info("SIMPLIFIED CVE ENRICHMENT TEST")
logger.info("="*60)

from scriptguard.data_sources.cve_feeds import CVEFeedSource

logger.info("\n1. Testing CVEFeedSource directly...")
cve_source = CVEFeedSource()

logger.info("2. Fetching CVEs (30 days, keywords=['script'])...")
cves = cve_source.fetch_recent_cves(days=30, keywords=["script", "code execution"])

logger.info(f"\n3. RESULT: Found {len(cves)} CVEs")

if cves:
    logger.info(f"\nFirst 3 CVEs:")
    for i, cve in enumerate(cves[:3], 1):
        logger.info(f"  {i}. {cve['cve_id']}: {cve['description'][:80]}...")
else:
    logger.error("ERROR: No CVEs found!")

logger.info("\n" + "="*60)
logger.info(f"Full log saved to: {log_file.absolute()}")
logger.info("="*60)

print(f"\nCheck the log file: {log_file.absolute()}")
