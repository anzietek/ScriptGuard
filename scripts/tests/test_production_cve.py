"""Test production CVEFeedSource after fix"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing PRODUCTION CVEFeedSource...")

from scriptguard.data_sources.cve_feeds import CVEFeedSource

cve_source = CVEFeedSource()
print("Fetching CVEs (30 days, keywords=['script', 'code execution'])...")

cves = cve_source.fetch_recent_cves(days=30, keywords=["script", "code execution"])

print(f"\n{'='*60}")
if cves:
    print(f"SUCCESS! Fetched {len(cves)} CVEs")
    print(f"First CVE: {cves[0]['cve_id']}")
    print(f"{'='*60}")
else:
    print(f"FAILED! No CVEs fetched")
    print(f"{'='*60}")
    sys.exit(1)
