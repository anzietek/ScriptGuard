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

from scriptguard.data_sources.vxunderground_api import VXUndergroundDataSource
from scriptguard.utils.archive_extractor import is_archive
import tempfile

print("=" * 70)
print("VX-UNDERGROUND QUICK TEST")
print("=" * 70)

github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")

if not github_token:
    print("ERROR: No GitHub token found")
    sys.exit(1)

print(f"\nGitHub Token: {github_token[:8]}...")

source = VXUndergroundDataSource(github_token=github_token)

print("\nAttempting to fetch 1 Python sample...")
samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

if len(samples) > 0:
    sample = samples[0]
    content = sample['content']

    print(f"\n[SUCCESS] Downloaded {len(content)} bytes")
    print(f"File: {sample['metadata'].get('file_name')}")
    print(f"Original: {sample['metadata'].get('original_file', 'N/A')}")
    print(f"Path: {sample['metadata'].get('path')}")
    print(f"\nFirst 200 chars:")
    print(content[:200])

    if is_archive(content.encode('utf-8', errors='ignore')[:512]):
        print("\n[FAIL] Content is still an archive!")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    size = os.path.getsize(temp_path)
    with open(temp_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    print(f"\n[SUCCESS] Saved to: {temp_path} ({size} bytes)")
    print(f"First line: {first_line}")
    os.unlink(temp_path)

    print("\n" + "=" * 70)
    print("VX-UNDERGROUND TEST: PASSED")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n[FAIL] No samples downloaded")
    print("\n" + "=" * 70)
    print("VX-UNDERGROUND TEST: FAILED")
    print("=" * 70)
    sys.exit(1)
