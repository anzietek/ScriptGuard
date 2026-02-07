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

from scriptguard.data_sources.malwarebazaar_api import MalwareBazaarDataSource
from scriptguard.data_sources.thezoo_api import TheZooDataSource
from scriptguard.data_sources.vxunderground_api import VXUndergroundDataSource
from scriptguard.utils.archive_extractor import is_archive
import tempfile

print("=" * 70)
print("LIVE INGESTION SMOKE TEST - ARCHIVE EXTRACTION")
print("=" * 70)

results = {}

print("\n[TEST 1] MalwareBazaar with archive extraction")
print("-" * 70)
try:
    api_key = os.getenv("MALWAREBAZAAR_API_KEY")
    if api_key:
        source = MalwareBazaarDataSource(api_key=api_key)
        print(f"API Key: {api_key[:8]}...")

        samples = source.fetch_malicious_samples(tags=["script"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            content = sample['content']

            if is_archive(content.encode('utf-8', errors='ignore')[:512]):
                print("FAIL: Content is still an archive")
                results['malwarebazaar'] = False
            elif len(content) > 100 and content.strip():
                print(f"SUCCESS: Extracted {len(content)} bytes of text")
                print(f"File: {sample['metadata'].get('file_name')}")
                print(f"Preview: {content[:100].strip()}")

                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(content)
                    temp_path = f.name

                size = os.path.getsize(temp_path)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                print(f"Saved: {temp_path} ({size} bytes)")
                print(f"First line: {first_line}")
                os.unlink(temp_path)
                results['malwarebazaar'] = True
            else:
                print("FAIL: Content too small or binary")
                results['malwarebazaar'] = False
        else:
            print("FAIL: No samples downloaded")
            results['malwarebazaar'] = False
    else:
        print("SKIP: No API key")
        results['malwarebazaar'] = False
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['malwarebazaar'] = False

print("\n[TEST 2] TheZoo with archive extraction")
print("-" * 70)
try:
    github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")
    if github_token:
        source = TheZooDataSource(github_token=github_token)
        print(f"GitHub Token: {github_token[:8]}...")

        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            content = sample['content']

            if is_archive(content.encode('utf-8', errors='ignore')[:512]):
                print("FAIL: Content is still an archive")
                results['thezoo'] = False
            elif len(content) > 100 and content.strip():
                print(f"SUCCESS: Extracted {len(content)} bytes of text")
                print(f"File: {sample['metadata'].get('file_name')}")
                print(f"Preview: {content[:100].strip()}")

                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(content)
                    temp_path = f.name

                size = os.path.getsize(temp_path)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                print(f"Saved: {temp_path} ({size} bytes)")
                print(f"First line: {first_line}")
                os.unlink(temp_path)
                results['thezoo'] = True
            else:
                print("FAIL: Content too small or binary")
                results['thezoo'] = False
        else:
            print("FAIL: No samples downloaded")
            results['thezoo'] = False
    else:
        print("SKIP: No GitHub token")
        results['thezoo'] = False
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['thezoo'] = False

print("\n[TEST 3] VX-Underground with archive extraction")
print("-" * 70)
try:
    github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")
    if github_token:
        source = VXUndergroundDataSource(github_token=github_token)
        print(f"GitHub Token: {github_token[:8]}...")

        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            content = sample['content']

            if is_archive(content.encode('utf-8', errors='ignore')[:512]):
                print("FAIL: Content is still an archive")
                results['vxunderground'] = False
            elif len(content) > 100 and content.strip():
                print(f"SUCCESS: Extracted {len(content)} bytes of text")
                print(f"File: {sample['metadata'].get('file_name')}")
                print(f"Preview: {content[:100].strip()}")

                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(content)
                    temp_path = f.name

                size = os.path.getsize(temp_path)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                print(f"Saved: {temp_path} ({size} bytes)")
                print(f"First line: {first_line}")
                os.unlink(temp_path)
                results['vxunderground'] = True
            else:
                print("FAIL: Content too small or binary")
                results['vxunderground'] = False
        else:
            print("FAIL: No samples downloaded")
            results['vxunderground'] = False
    else:
        print("SKIP: No GitHub token")
        results['vxunderground'] = False
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['vxunderground'] = False

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
for source, success in results.items():
    status = "SUCCESS" if success else "FAILED"
    print(f"{source.upper():20s} [{status}]")

passed = sum(results.values())
total = len(results)
print(f"\nRESULT: {passed}/{total} sources working")
print("=" * 70)

sys.exit(0 if passed > 0 else 1)
