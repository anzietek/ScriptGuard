"""
Diagnostic script to debug Qdrant authentication issues.
Checks environment variables and tests connection.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

print("=" * 70)
print("QDRANT AUTHENTICATION DIAGNOSTICS")
print("=" * 70)

# Step 1: Check if .env file exists
print("\n1. Checking .env file...")
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    print(f"   ✓ .env file found: {env_path}")
else:
    print(f"   ✗ .env file NOT found: {env_path}")
    print("   Create .env file with QDRANT_API_KEY")
    sys.exit(1)

# Step 2: Load .env explicitly
print("\n2. Loading .env file...")
from dotenv import load_dotenv
loaded = load_dotenv(env_path)
print(f"   {'✓' if loaded else '✗'} load_dotenv() returned: {loaded}")

# Step 3: Check environment variables
print("\n3. Checking environment variables...")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"   QDRANT_HOST: {qdrant_host or '(not set)'}")
print(f"   QDRANT_PORT: {qdrant_port or '(not set)'}")
if qdrant_api_key:
    masked_key = "***" + qdrant_api_key[-8:] if len(qdrant_api_key) > 8 else "***"
    print(f"   QDRANT_API_KEY: {masked_key} (length: {len(qdrant_api_key)})")
else:
    print(f"   QDRANT_API_KEY: (not set)")
    print("   ⚠ WARNING: API key is not set!")

# Step 4: Test direct Qdrant connection
print("\n4. Testing direct Qdrant connection...")
try:
    from qdrant_client import QdrantClient

    if qdrant_api_key:
        client = QdrantClient(
            url=f"http://{qdrant_host or 'localhost'}:{qdrant_port or 6333}",
            api_key=qdrant_api_key,
            timeout=5
        )
        print(f"   ✓ QdrantClient initialized with API key")
    else:
        client = QdrantClient(
            host=qdrant_host or "localhost",
            port=int(qdrant_port or 6333),
            timeout=5
        )
        print(f"   ⚠ QdrantClient initialized WITHOUT API key")

    # Try to list collections
    collections = client.get_collections()
    print(f"   ✓ Connected successfully!")
    print(f"   Collections: {len(collections.collections)}")
    for coll in collections.collections:
        print(f"     - {coll.name}")

except Exception as e:
    print(f"   ✗ Connection failed: {e}")
    print("\n   Troubleshooting:")
    print("   - Verify Qdrant is running")
    print("   - Check firewall settings")
    print("   - Verify API key is correct")
    sys.exit(1)

# Step 5: Test QdrantStore initialization
print("\n5. Testing QdrantStore initialization...")
try:
    from scriptguard.rag.qdrant_store import QdrantStore

    print("   Initializing QdrantStore...")
    store = QdrantStore(
        collection_name="test_collection"
    )
    print(f"   ✓ QdrantStore initialized successfully!")
    print(f"   Host: {store.host}")
    print(f"   Port: {store.port}")
    print(f"   API Key: {'set' if store.api_key else 'not set'}")

    # Try to get collection info
    info = store.get_collection_info()
    print(f"   ✓ Collection accessible: {info.get('points_count', 0)} points")

except Exception as e:
    print(f"   ✗ QdrantStore initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test CodeSimilarityStore initialization
print("\n6. Testing CodeSimilarityStore initialization...")
try:
    from scriptguard.rag.code_similarity_store import CodeSimilarityStore

    print("   Initializing CodeSimilarityStore...")
    code_store = CodeSimilarityStore(
        collection_name="test_code_samples"
    )
    print(f"   ✓ CodeSimilarityStore initialized successfully!")
    print(f"   Host: {code_store.host}")
    print(f"   Port: {code_store.port}")
    print(f"   API Key: {'set' if code_store.api_key else 'not set'}")

except Exception as e:
    print(f"   ✗ CodeSimilarityStore initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED! ✓")
print("=" * 70)
print("\nYour Qdrant authentication is configured correctly.")
print("If you're still seeing 401 errors, check:")
print("  1. Make sure .env is in the project root")
print("  2. Restart your Python process/IDE")
print("  3. Verify the API key on Qdrant server matches .env")
