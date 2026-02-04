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

from scriptguard.data_sources.malwarebazaar_api import MalwareBazaarDataSource
from scriptguard.data_sources.thezoo_api import TheZooDataSource
from scriptguard.data_sources.vxunderground_api import VXUndergroundDataSource
from scriptguard.utils.sample_saver import save_samples_to_disk

OUTPUT_DIR = "extracted_samples"

print("=" * 70)
print("SAVE EXTRACTED SAMPLES TO DISK")
print("=" * 70)

results = []

print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

print("\n[1] MalwareBazaar")
api_key = os.getenv("MALWAREBAZAAR_API_KEY")
if api_key:
    source = MalwareBazaarDataSource(api_key=api_key)
    samples = source.fetch_malicious_samples(tags=["script"], max_samples=3)
    if samples:
        paths = save_samples_to_disk(samples, OUTPUT_DIR)
        results.extend(paths)
        print(f"Saved {len(paths)} files")

print("\n[2] TheZoo")
github_token = os.getenv("GITHUB_API_TOKEN") or os.getenv("GITHUB_TOKEN")
if github_token:
    source = TheZooDataSource(github_token=github_token)
    samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=3)
    if samples:
        paths = save_samples_to_disk(samples, OUTPUT_DIR)
        results.extend(paths)
        print(f"Saved {len(paths)} files")

print("\n[3] VX-Underground")
if github_token:
    source = VXUndergroundDataSource(github_token=github_token)
    samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=3)
    if samples:
        paths = save_samples_to_disk(samples, OUTPUT_DIR)
        results.extend(paths)
        print(f"Saved {len(paths)} files")

print("\n" + "=" * 70)
print(f"TOTAL: {len(results)} files saved to {OUTPUT_DIR}/")
print("=" * 70)

for path in results:
    size = os.path.getsize(path)
    print(f"  {os.path.basename(path)} ({size} bytes)")

print("\n" + "=" * 70)
