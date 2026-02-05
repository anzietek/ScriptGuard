"""Quick test of NVD API access."""
import requests
from datetime import datetime, timedelta

# Test basic NVD API access
url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Try with a specific CVE first
print("Testing NVD API access...")
print(f"URL: {url}")
print()

# Test 1: Get a specific known CVE
print("Test 1: Fetching a known CVE (CVE-2021-44228 - Log4Shell)")
try:
    response = requests.get(f"{url}", params={"cveId": "CVE-2021-44228"}, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Found {data.get('totalResults', 0)} results")
        print(f"Response keys: {list(data.keys())}")
    else:
        print(f"❌ Failed: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("-" * 60)
print()

# Test 2: Get recent CVEs with date range
print("Test 2: Fetching recent CVEs (last 7 days)")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    params = {
        "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": 10
    }

    print(f"Start: {params['pubStartDate']}")
    print(f"End:   {params['pubEndDate']}")

    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        total = data.get('totalResults', 0)
        vulnerabilities = data.get('vulnerabilities', [])
        print(f"✅ Success! Found {total} total results")
        print(f"Retrieved {len(vulnerabilities)} in this page")

        if vulnerabilities:
            print(f"\nFirst CVE:")
            first_cve = vulnerabilities[0].get('cve', {})
            print(f"  ID: {first_cve.get('id')}")
            desc = first_cve.get('descriptions', [{}])[0].get('value', 'N/A')
            print(f"  Description: {desc[:100]}...")
    else:
        print(f"❌ Failed: {response.text[:200]}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("If tests fail, NVD API might be down or have changed.")
print("Check: https://nvd.nist.gov/developers")
