#!/usr/bin/env python3
"""List all Qdrant collections and their stats."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

# Connect to Qdrant
api_key = os.getenv("QDRANT_API_KEY")
client_kwargs = {
    "host": "localhost",
    "port": 6333,
    "https": False,
    "timeout": 60
}
if api_key:
    client_kwargs["api_key"] = api_key

client = QdrantClient(**client_kwargs)

print("=" * 70)
print("QDRANT COLLECTIONS")
print("=" * 70)

# List all collections
collections = client.get_collections()

if not collections.collections:
    print("\n❌ No collections found in Qdrant!")
    print("\nYou need to run the vectorization pipeline first:")
    print("  python -m scriptguard.pipelines.train_pipeline")
    sys.exit(1)

print(f"\nFound {len(collections.collections)} collection(s):\n")

for collection in collections.collections:
    name = collection.name

    # Get detailed info
    info = client.get_collection(name)
    points_count = info.points_count
    vector_size = info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else "unknown"

    print(f"📦 {name}")
    print(f"   Points: {points_count:,}")
    print(f"   Vector size: {vector_size}")

    # Sample a few points to check structure
    if points_count > 0:
        sample = client.scroll(collection_name=name, limit=1, with_payload=True)
        if sample[0]:
            point = sample[0][0]
            payload_keys = list(point.payload.keys())
            has_features = "features" in payload_keys
            has_code = any(k in payload_keys for k in ["code", "content", "code_preview"])

            print(f"   Payload fields: {', '.join(payload_keys[:5])}{'...' if len(payload_keys) > 5 else ''}")
            print(f"   Has features: {'✅' if has_features else '❌'}")
            print(f"   Has code: {'✅' if has_code else '❌'}")

    print()

print("=" * 70)
