"""
Final ingestion verification - loads .env manually
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
    print("[+] Loaded .env file")
else:
    print("[!] No .env file found")

from scriptguard.data_sources.malwarebazaar_api import MalwareBazaarDataSource
from scriptguard.data_sources.thezoo_api import TheZooDataSource
import tempfile

print("\n" + "=" * 70)
print("FINAL INGESTION VERIFICATION")
print("=" * 70)

print("\n[+] Environment check:")
print(f"    MALWAREBAZAAR_API_KEY: {'SET' if os.getenv('MALWAREBAZAAR_API_KEY') else 'NOT SET'}")
print(f"    GITHUB_API_TOKEN: {'SET' if os.getenv('GITHUB_API_TOKEN') else 'NOT SET'}")

results = {}

print("\n" + "=" * 70)
print("TEST 1: MalwareBazaar (with API key)")
print("=" * 70)
try:
    api_key = os.getenv("MALWAREBAZAAR_API_KEY")
    if api_key:
        source = MalwareBazaarDataSource(api_key=api_key)
        print(f"[+] API key configured: {api_key[:8]}...")

        print("\nFetching 1 sample with tag 'script'...")
        samples = source.fetch_malicious_samples(tags=["script"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            print(f"[+] SUCCESS - Downloaded {len(sample['content'])} bytes")
            print(f"    File: {sample['metadata'].get('file_name', 'unknown')}")
            print(f"    Source: {sample['source']}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False, encoding='utf-8') as f:
                f.write(sample['content'])
                temp_path = f.name

            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                print(f"    Saved: {temp_path} ({size} bytes)")
                os.unlink(temp_path)
                results['malwarebazaar'] = True
            else:
                print("[-] File not saved")
                results['malwarebazaar'] = False
        else:
            print("[-] No samples downloaded")
            results['malwarebazaar'] = False
    else:
        print("[-] No API key found in environment")
        results['malwarebazaar'] = False
except Exception as e:
    print(f"[-] ERROR: {e}")
    results['malwarebazaar'] = False

print("\n" + "=" * 70)
print("TEST 2: TheZoo (with GitHub token)")
print("=" * 70)
try:
    github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")
    if github_token:
        source = TheZooDataSource(github_token=github_token)
        print(f"[+] GitHub token configured: {github_token[:8]}...")

        print("\nFetching 1 Python sample...")
        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            print(f"[+] SUCCESS - Downloaded {len(sample['content'])} bytes")
            print(f"    File: {sample['metadata'].get('file_name', 'unknown')}")
            print(f"    Path: {sample['metadata'].get('path', 'unknown')}")
            print(f"    Source: {sample['source']}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(sample['content'])
                temp_path = f.name

            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                print(f"    Saved: {temp_path} ({size} bytes)")
                os.unlink(temp_path)
                results['thezoo'] = True
            else:
                print("[-] File not saved")
                results['thezoo'] = False
        else:
            print("[-] No samples downloaded")
            results['thezoo'] = False
    else:
        print("[-] No GitHub token found in environment")
        results['thezoo'] = False
except Exception as e:
    print(f"[-] ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['thezoo'] = False

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for source, success in results.items():
    status = "[+] SUCCESS" if success else "[-] FAILED"
    print(f"{source.upper():20s} {status}")

passed = sum(results.values())
total = len(results)
print(f"\nRESULT: {passed}/{total} sources working")
print("=" * 70)

sys.exit(0 if passed > 0 else 1)
