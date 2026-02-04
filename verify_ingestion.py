"""
LIVE Ingestion Smoke Test
Tests actual data collection from all sources with NO MOCKS.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


from scriptguard.data_sources.malwarebazaar_api import MalwareBazaarDataSource
from scriptguard.data_sources.vxunderground_api import VXUndergroundDataSource
from scriptguard.data_sources.thezoo_api import TheZooDataSource
from scriptguard.utils.logger import logger
import tempfile
import json

def test_malwarebazaar():
    """Test MalwareBazaar live connection and download."""
    print("\n" + "=" * 70)
    print("TEST 1: MalwareBazaar Live Connection")
    print("=" * 70)

    try:
        api_key = os.getenv("MALWAREBAZAAR_API_KEY")
        source = MalwareBazaarDataSource(api_key=api_key)

        print(f"API Key configured: {bool(api_key)}")

        if not api_key:
            print("[!] WARNING: No API key - trying public API with recent samples...")
            recent = source.get_recent_samples(limit=5)
            print(f"Found {len(recent)} recent samples")
            if len(recent) > 0:
                for sample_meta in recent[:1]:
                    sha256 = sample_meta.get("sha256_hash")
                    print(f"Trying to download: {sha256[:16]}...")
                    content_bytes = source.download_sample(sha256)
                    if content_bytes:
                        try:
                            content = content_bytes.decode("utf-8", errors="ignore")
                            if len(content) > 50:
                                print(f"[+] SUCCESS - Downloaded {len(content)} bytes")
                                print(f"  SHA256: {sha256[:16]}...")

                                with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False, encoding='utf-8') as f:
                                    f.write(content)
                                    temp_path = f.name

                                if os.path.exists(temp_path):
                                    size = os.path.getsize(temp_path)
                                    print(f"  Saved to: {temp_path} ({size} bytes)")
                                    os.unlink(temp_path)
                                    return True
                        except:
                            pass
        else:
            print("\nAttempting to fetch 1 sample from tag 'python'...")
            samples = source.fetch_malicious_samples(tags=["python"], max_samples=1)

            if len(samples) > 0:
                sample = samples[0]
                print(f"[+] SUCCESS - Downloaded {len(sample['content'])} bytes")
                print(f"  File: {sample['metadata'].get('file_name', 'unknown')}")
                print(f"  SHA256: {sample['metadata'].get('sha256', 'unknown')[:16]}...")
                print(f"  Source: {sample['source']}")

                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(sample['content'])
                    temp_path = f.name

                if os.path.exists(temp_path):
                    size = os.path.getsize(temp_path)
                    print(f"  Saved to: {temp_path} ({size} bytes)")
                    os.unlink(temp_path)
                    return True

        print("[-] ERROR - No samples downloaded")
        return False

    except Exception as e:
        print(f"[-] ERROR - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vxunderground():
    """Test VX-Underground live connection and download."""
    print("\n" + "=" * 70)
    print("TEST 2: VX-Underground Live Connection")
    print("=" * 70)

    try:
        github_token = os.getenv("GITHUB_TOKEN")
        source = VXUndergroundDataSource(github_token=github_token)

        print(f"GitHub token configured: {bool(github_token)}")

        print("\nAttempting to fetch 1 Python sample...")
        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            print(f"[+] SUCCESS - Downloaded {len(sample['content'])} bytes")
            print(f"  File: {sample['metadata'].get('file_name', 'unknown')}")
            print(f"  Path: {sample['metadata'].get('path', 'unknown')}")
            print(f"  Source: {sample['source']}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(sample['content'])
                temp_path = f.name

            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                print(f"  Saved to: {temp_path} ({size} bytes)")
                os.unlink(temp_path)
                return True
            else:
                print("[-] ERROR - File not saved")
                return False
        else:
            print("[-] ERROR - No samples downloaded")
            return False

    except Exception as e:
        print(f"[-] ERROR - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_thezoo():
    """Test TheZoo live connection and download."""
    print("\n" + "=" * 70)
    print("TEST 3: TheZoo Live Connection")
    print("=" * 70)

    try:
        github_token = os.getenv("GITHUB_TOKEN")
        source = TheZooDataSource(github_token=github_token)

        print(f"GitHub token configured: {bool(github_token)}")

        print("\nAttempting to fetch 1 Python sample...")
        samples = source.fetch_malicious_samples(script_types=[".py"], max_samples=1)

        if len(samples) > 0:
            sample = samples[0]
            print(f"[+] SUCCESS - Downloaded {len(sample['content'])} bytes")
            print(f"  File: {sample['metadata'].get('file_name', 'unknown')}")
            print(f"  Path: {sample['metadata'].get('path', 'unknown')}")
            print(f"  Source: {sample['source']}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(sample['content'])
                temp_path = f.name

            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                print(f"  Saved to: {temp_path} ({size} bytes)")
                os.unlink(temp_path)
                return True
            else:
                print("[-] ERROR - File not saved")
                return False
        else:
            print("[-] ERROR - No samples downloaded")
            return False

    except Exception as e:
        print(f"[-] ERROR - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LIVE INGESTION SMOKE TEST - NO MOCKS")
    print("Testing real network I/O and file operations")
    print("=" * 70)

    results = {
        "malwarebazaar": test_malwarebazaar(),
        "vxunderground": test_vxunderground(),
        "thezoo": test_thezoo()
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for source, success in results.items():
        status = "[+] SUCCESS" if success else "[-] FAILED"
        print(f"{source.upper():20s} {status}")

    total = len(results)
    passed = sum(results.values())

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {passed}/{total} sources working")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)
