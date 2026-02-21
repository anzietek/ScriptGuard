"""Test which dashboard URLs are accessible."""
import requests
import os
from dotenv import load_dotenv
load_dotenv()

base_url = "http://localhost:8237"

# Test various URLs
urls_to_test = [
    "/",
    "/health",
    "/api/v1/info",
    "/api/v1/pipelines",
    "/api/v1/runs",
    "/pipelines",
    "/runs",
    "/workspaces",
]

print("Testing Dashboard URLs:")
print("="*70)

for path in urls_to_test:
    url = f"{base_url}{path}"
    try:
        response = requests.get(url, timeout=5)
        status = response.status_code

        if status == 200:
            symbol = "[OK]"
        elif status == 401:
            symbol = "[AUTH]"
            status = f"{status} (Auth required)"
        elif status == 404:
            symbol = "[404]"
            status = f"{status} (Not found)"
        else:
            symbol = "[???]"

        print(f"{symbol} {url:<50} {status}")

    except Exception as e:
        print(f"[ERR] {url:<50} ERROR: {e}")

print("\n" + "="*70)
print("TRY THESE URLs IN YOUR BROWSER:")
print("="*70)
print(f"\n1. Main dashboard: {base_url}/")
print(f"2. API info: {base_url}/api/v1/info")
print(f"3. Direct to runs: {base_url}/runs")
print(f"4. Direct to pipelines: {base_url}/pipelines")
