"""Test NVD API with exact same params as cve_feeds.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scriptguard.data_sources.cve_feeds import CVEFeedSource
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize source
cve_source = CVEFeedSource()

# Try fetching CVEs
print("="*60)
print("Testing CVE fetch with same parameters as pipeline...")
print("="*60)

cves = cve_source.fetch_recent_cves(days=30, keywords=["script", "code execution"])

print(f"\nResult: {len(cves)} CVEs found")

if cves:
    print("\nFirst few CVEs:")
    for cve in cves[:3]:
        print(f"  - {cve['cve_id']}: {cve['description'][:80]}...")
