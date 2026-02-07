import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

env_file = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

from scriptguard.data_sources.vxunderground_api import VXUndergroundDataSource
from scriptguard.data_sources.thezoo_api import TheZooDataSource
from scriptguard.data_sources.malwarebazaar_api import MalwareBazaarDataSource
from scriptguard.database import DatasetManager

print("=" * 70)
print("DATABASE INGESTION TEST")
print("=" * 70)

try:
    db = DatasetManager()
    print("[+] Database connection OK")
except Exception as e:
    print(f"[-] Database connection FAILED: {e}")
    sys.exit(1)

print("\nQuerying database...")
before_all = db.get_all_samples(limit=None)
before_mal = db.get_all_samples(label="malicious", limit=None)
before_count = len(before_all)
before_mal_count = len(before_mal)
print(f"Before: {before_count} total ({before_mal_count} malicious)")

added = 0

print("\n[1] VX-Underground -> Database")
github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")
if github_token:
    try:
        source = VXUndergroundDataSource(github_token=github_token)
        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=3)

        if not samples:
            print("    [!] No samples fetched")

        for sample in samples:
            content = sample['content']
            fname = sample['metadata'].get('file_name')

            if '\x00' in content:
                print(f"    [!] Rejected (binary): {fname}")
                continue

            if len(content) < 100:
                print(f"    [!] Rejected (too small): {fname}")
                continue

            try:
                result = db.add_sample(
                    content=content,
                    label=sample['label'],
                    source=sample['source'],
                    metadata=sample['metadata']
                )
                if result:
                    print(f"    [+] Added: {fname} ({len(content)} bytes)")
                    added += 1
                else:
                    print(f"    [~] Duplicate: {fname}")
            except Exception as e:
                print(f"    [-] ERROR: {fname} - {str(e)[:60]}")
    except Exception as e:
        print(f"    [-] ERROR: {e}")
else:
    print("    [!] No GitHub token")

print("\n[2] TheZoo -> Database")
if github_token:
    try:
        source = TheZooDataSource(github_token=github_token)
        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=3)

        if not samples:
            print("    [!] No samples fetched")

        for sample in samples:
            content = sample['content']
            fname = sample['metadata'].get('file_name')

            if '\x00' in content:
                print(f"    [!] Rejected (binary): {fname}")
                continue

            if len(content) < 100:
                print(f"    [!] Rejected (too small): {fname}")
                continue

            try:
                result = db.add_sample(
                    content=content,
                    label=sample['label'],
                    source=sample['source'],
                    metadata=sample['metadata']
                )
                if result:
                    print(f"    [+] Added: {fname} ({len(content)} bytes)")
                    added += 1
                else:
                    print(f"    [~] Duplicate: {fname}")
            except Exception as e:
                print(f"    [-] ERROR: {fname} - {str(e)[:60]}")
    except Exception as e:
        print(f"    [-] ERROR: {e}")
else:
    print("    [!] No GitHub token")

print("\n[3] MalwareBazaar -> Database")
api_key = os.getenv("MALWAREBAZAAR_API_KEY")
if api_key:
    try:
        source = MalwareBazaarDataSource(api_key=api_key)
        samples = source.fetch_malicious_samples(tags=["bat", "vbs", "js"], max_samples=3)

        if not samples:
            print("    [!] No samples fetched")

        for sample in samples:
            content = sample['content']
            fname = sample['metadata'].get('file_name')

            if '\x00' in content:
                print(f"    [!] Rejected (binary): {fname}")
                continue

            if len(content) < 100:
                print(f"    [!] Rejected (too small): {fname}")
                continue

            try:
                result = db.add_sample(
                    content=content,
                    label=sample['label'],
                    source=sample['source'],
                    metadata=sample['metadata']
                )
                if result:
                    print(f"    [+] Added: {fname} ({len(content)} bytes)")
                    added += 1
                else:
                    print(f"    [~] Duplicate: {fname}")
            except Exception as e:
                print(f"    [-] ERROR: {fname} - {str(e)[:60]}")
    except Exception as e:
        print(f"    [-] ERROR: {e}")
else:
    print("    [!] No API key")

print("\nQuerying database after ingestion...")
after_all = db.get_all_samples(limit=None)
after_mal = db.get_all_samples(label="malicious", limit=None)
after_count = len(after_all)
after_mal_count = len(after_mal)

print("\n" + "=" * 70)
print(f"After:  {after_count} total ({after_mal_count} malicious)")
print(f"Delta:  +{after_count - before_count} total (+{after_mal_count - before_mal_count} malicious)")
print(f"Result: Added {added} new samples to database")
print("=" * 70)

if added > 0:
    print("\n[SUCCESS] Ingestion to database working!")
    sys.exit(0)
else:
    print("\n[WARNING] No new samples added (might be duplicates)")
    sys.exit(0)
