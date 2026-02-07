import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

env_file = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

import requests

api_key = os.getenv("MALWAREBAZAAR_API_KEY")
print(f"API Key: {api_key[:10]}..." if api_key else "NO KEY")

print("\nTest 1: Simple query without API key")
resp = requests.post("https://mb-api.abuse.ch/api/v1/", data={"query": "get_taginfo", "tag": "exe", "limit": 1})
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300]}")

print("\nTest 2: With API key in Auth-Key header")
resp = requests.post("https://mb-api.abuse.ch/api/v1/", data={"query": "get_taginfo", "tag": "exe", "limit": 1}, headers={"Auth-Key": api_key})
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300]}")

print("\nTest 3: get_recent query with Auth-Key")
resp = requests.post("https://mb-api.abuse.ch/api/v1/", data={"query": "get_recent", "selector": "time"}, headers={"Auth-Key": api_key})
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300]}")
