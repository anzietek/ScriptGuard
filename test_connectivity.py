"""
Quick diagnostic test for data sources
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import requests

print("\n" + "=" * 70)
print("DATA SOURCE CONNECTIVITY TEST")
print("=" * 70)

print("\n[1] Testing MalwareBazaar API...")
try:
    resp = requests.post(
        "https://mb-api.abuse.ch/api/v1/",
        data={"query": "get_recent", "selector": "time"},
        timeout=10
    )
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Response: {data.get('query_status')}")
        if data.get('data'):
            print(f"    Samples available: {len(data['data'])}")
    elif resp.status_code == 401:
        print("    [!] API key required")
    print("    [+] Endpoint accessible")
except Exception as e:
    print(f"    [-] Error: {e}")

print("\n[2] Testing VX-Underground GitHub...")
try:
    resp = requests.get(
        "https://api.github.com/repos/vxunderground/MalwareSourceCode",
        timeout=10
    )
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Repo: {data.get('full_name')}")
        print(f"    Stars: {data.get('stargazers_count')}")
        print(f"    [+] Repository accessible")
    elif resp.status_code == 404:
        print("    [-] Repository not found")
except Exception as e:
    print(f"    [-] Error: {e}")

print("\n[3] Testing TheZoo GitHub...")
try:
    resp = requests.get(
        "https://api.github.com/repos/ytisf/theZoo",
        timeout=10
    )
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Repo: {data.get('full_name')}")
        print(f"    Stars: {data.get('stargazers_count')}")
        print(f"    [+] Repository accessible")
    elif resp.status_code == 404:
        print("    [-] Repository not found")
except Exception as e:
    print(f"    [-] Error: {e}")

print("\n[4] Testing GitHub API rate limit...")
try:
    resp = requests.get(
        "https://api.github.com/rate_limit",
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        core = data.get('resources', {}).get('core', {})
        print(f"    Remaining: {core.get('remaining')}/{core.get('limit')}")
        print(f"    [+] GitHub API accessible")
except Exception as e:
    print(f"    [-] Error: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
