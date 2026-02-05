"""Test whether NVD API is accessible RIGHT NOW"""
import requests
from datetime import datetime, timedelta

url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

params = {
    "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
    "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999"),
    "resultsPerPage": 10
}

print(f"Testing NVD API at {datetime.now()}")
print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Full URL: {response.url}")

    if response.status_code == 200:
        data = response.json()
        total = data.get('totalResults', 0)
        print(f"✅ SUCCESS! Found {total} CVEs")
    elif response.status_code == 404:
        print(f"❌ 404 ERROR")
        print(f"Response: {response.text[:200]}")
    elif response.status_code == 403:
        print(f"❌ 403 RATE LIMIT / FORBIDDEN")
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
