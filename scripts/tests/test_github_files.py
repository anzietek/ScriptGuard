"""
Simple file fetch test for GitHub repositories
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import requests
import time

def test_vx_files():
    """Test fetching actual files from VX-Underground"""
    print("\n" + "=" * 70)
    print("VX-UNDERGROUND FILE TEST")
    print("=" * 70)

    base_url = "https://api.github.com/repos/vxunderground/MalwareSourceCode/contents"

    print("\n[1] Listing root directories...")
    resp = requests.get(base_url, timeout=10)
    if resp.status_code == 200:
        dirs = [item for item in resp.json() if item['type'] == 'dir']
        print(f"    Found {len(dirs)} directories")
        for d in dirs[:5]:
            print(f"      - {d['name']}")

        print("\n[2] Exploring first directory with files...")
        for d in dirs[:3]:
            time.sleep(1)
            print(f"\n    Checking: {d['name']}")
            resp2 = requests.get(d['url'], timeout=10)
            if resp2.status_code == 200:
                items = resp2.json()
                if isinstance(items, list):
                    files = [item for item in items if item['type'] == 'file' and item['name'].endswith(('.py', '.ps1', '.js'))]
                    if files:
                        print(f"      Found {len(files)} script files!")
                        f = files[0]
                        print(f"      Downloading: {f['name']}")

                        resp3 = requests.get(f['download_url'], timeout=10)
                        if resp3.status_code == 200:
                            content = resp3.text
                            print(f"      [+] SUCCESS - Downloaded {len(content)} bytes")
                            print(f"      Path: {f['path']}")
                            return True
    return False

def test_thezoo_files():
    """Test fetching actual files from TheZoo"""
    print("\n" + "=" * 70)
    print("THEZOO FILE TEST")
    print("=" * 70)

    base_url = "https://api.github.com/repos/ytisf/theZoo/contents"

    print("\n[1] Listing root directories...")
    resp = requests.get(base_url, timeout=10)
    if resp.status_code == 200:
        dirs = [item for item in resp.json() if item['type'] == 'dir']
        print(f"    Found {len(dirs)} directories")
        for d in dirs[:5]:
            print(f"      - {d['name']}")

        print("\n[2] Exploring directories for script files...")
        for d in dirs[:3]:
            time.sleep(1)
            print(f"\n    Checking: {d['name']}")
            resp2 = requests.get(d['url'], timeout=10)
            if resp2.status_code == 200:
                items = resp2.json()
                if isinstance(items, list):
                    files = [item for item in items if item['type'] == 'file' and item['name'].endswith(('.py', '.ps1', '.js', '.sh'))]
                    if files:
                        print(f"      Found {len(files)} script files!")
                        f = files[0]
                        print(f"      Downloading: {f['name']}")

                        resp3 = requests.get(f['download_url'], timeout=10)
                        if resp3.status_code == 200:
                            content = resp3.text
                            print(f"      [+] SUCCESS - Downloaded {len(content)} bytes")
                            print(f"      Path: {f['path']}")
                            return True
    return False

if __name__ == "__main__":
    vx_result = test_vx_files()
    zoo_result = test_thezoo_files()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"VX-Underground: {'[+] SUCCESS' if vx_result else '[-] FAILED'}")
    print(f"TheZoo:         {'[+] SUCCESS' if zoo_result else '[-] FAILED'}")
    print("=" * 70)
