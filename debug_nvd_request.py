"""Debug NVD API request - zapisuje wszystko do pliku"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import requests
from datetime import datetime, timedelta

# Setup logging to file
log_file = Path(__file__).parent / "nvd_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("NVD API DEBUG TEST")
logger.info("="*80)

# Test 1: Direct request like in test_nvd_api.py (this works)
logger.info("\n=== TEST 1: Direct request (working version) ===")
url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

params = {
    "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
    "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
    "resultsPerPage": 10
}

logger.info(f"URL: {url}")
logger.info(f"Params: {params}")

try:
    response = requests.get(url, params=params, timeout=30)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Full URL: {response.url}")
    logger.info(f"Response headers: {dict(response.headers)}")

    if response.status_code == 200:
        data = response.json()
        total = data.get('totalResults', 0)
        logger.info(f"SUCCESS! Found {total} total CVEs")
    else:
        logger.error(f"FAILED! Response: {response.text[:500]}")
except Exception as e:
    logger.error(f"Exception: {e}", exc_info=True)

# Test 2: Using CVEFeedSource class (failing version)
logger.info("\n=== TEST 2: Using CVEFeedSource class ===")

try:
    from scriptguard.data_sources.cve_feeds import CVEFeedSource

    cve_source = CVEFeedSource()
    logger.info(f"CVEFeedSource initialized")
    logger.info(f"NVD_API_URL: {cve_source.NVD_API_URL}")

    # Try fetching
    cves = cve_source.fetch_recent_cves(days=30, keywords=["script"])
    logger.info(f"Result: {len(cves)} CVEs found")

    if cves:
        logger.info(f"First CVE: {cves[0]}")
    else:
        logger.error("NO CVEs returned!")

except Exception as e:
    logger.error(f"CVEFeedSource failed: {e}", exc_info=True)

logger.info("\n" + "="*80)
logger.info(f"Log saved to: {log_file.absolute()}")
logger.info("="*80)

print(f"\n\nFull log saved to: {log_file.absolute()}")
