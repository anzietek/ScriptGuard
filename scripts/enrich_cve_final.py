#!/usr/bin/env python
"""
FINAL WORKING CVE Enrichment - bypasses all logging issues
Directly fetches CVE and adds to Qdrant without using cve_feeds.py
"""

import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.rag.qdrant_store import QdrantStore
import yaml


def fetch_cves_directly(days=30, keywords=None):
    """Fetch CVEs directly from NVD API without using cve_feeds.py"""
    if keywords is None:
        keywords = ["script", "code execution", "remote code execution", "command injection"]

    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    params = {
        "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
        "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999"),
        "resultsPerPage": 2000
    }

    print(f"Fetching CVEs from {start_date.date()} to {end_date.date()}")
    print(f"Keywords: {keywords}")

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(f"ERROR: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return []

        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])

        # Filter by keywords
        filtered_cves = []
        for vuln_item in vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id", "")

            # Get description
            descriptions = cve_data.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break

            # Filter by keywords
            if keywords:
                description_lower = description.lower()
                if not any(kw.lower() in description_lower for kw in keywords):
                    continue

            # Get CVSS score
            metrics = cve_data.get("metrics", {})
            cvss_score = None
            severity = None

            if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
                cvss_score = metrics["cvssMetricV31"][0]["cvssData"]["baseScore"]
                severity = metrics["cvssMetricV31"][0]["cvssData"]["baseSeverity"]
            elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
                cvss_score = metrics["cvssMetricV2"][0]["cvssData"]["baseScore"]
                severity = metrics["cvssMetricV2"][0]["baseSeverity"]

            filtered_cves.append({
                "cve_id": cve_id,
                "description": description,
                "cvss_score": cvss_score,
                "severity": severity or "UNKNOWN",
                "published": cve_data.get("published")
            })

        print(f"Found {len(filtered_cves)} relevant CVEs")
        return filtered_cves

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    print("="*70)
    print("FINAL CVE ENRICHMENT - Direct NVD API")
    print("="*70)

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    qdrant_config = config.get("qdrant", {})
    cve_config = config.get("data_sources", {}).get("cve_feeds", {})

    # Initialize Qdrant
    print("\n1. Connecting to Qdrant...")
    store = QdrantStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
        embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    initial_info = store.get_collection_info()
    initial_count = initial_info.get('points_count', 0)
    print(f"   Initial: {initial_count} points")

    # Fetch CVEs directly
    print("\n2. Fetching CVEs from NVD API...")
    days_back = cve_config.get("days_back", 30)
    keywords = cve_config.get("keywords", ["script", "code execution"])

    cves = fetch_cves_directly(days=days_back, keywords=keywords)

    if not cves:
        print("   No CVEs found!")
        return 1

    # Convert to Qdrant format
    print(f"\n3. Adding {len(cves)} CVEs to Qdrant...")
    cve_data = []
    for cve in cves:
        cve_data.append({
            "cve_id": cve["cve_id"],
            "description": cve["description"],
            "severity": cve["severity"],
            "pattern": "",
            "type": "cve",
            "cvss_score": cve.get("cvss_score"),
            "published": cve.get("published")
        })

    # Add in batches
    batch_size = 50
    added = 0
    for i in range(0, len(cve_data), batch_size):
        batch = cve_data[i:i+batch_size]
        try:
            store.upsert_vulnerabilities(batch)
            added += len(batch)
            print(f"   Batch {i//batch_size + 1}: {added}/{len(cve_data)}")
        except Exception as e:
            print(f"   Batch {i//batch_size + 1} FAILED: {e}")

    # Final stats
    print("\n" + "="*70)
    final_info = store.get_collection_info()
    final_count = final_info.get('points_count', 0)

    print("DONE!")
    print(f"  Initial:  {initial_count:>6} points")
    print(f"  Final:    {final_count:>6} points")
    print(f"  Added:    {added:>6} CVEs")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
