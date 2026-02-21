"""
Diagnostic script to check Qdrant collections and their contents.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Connect to Qdrant
host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", "6333"))
api_key = os.getenv("QDRANT_API_KEY")

print(f"Connecting to Qdrant at {host}:{port}")
if api_key:
    client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
else:
    client = QdrantClient(host=host, port=port)

# List all collections
collections = client.get_collections().collections
print(f"\n{'='*60}")
print(f"Found {len(collections)} collections:")
print(f"{'='*60}")

for collection in collections:
    print(f"\nCollection: {collection.name}")
    info = client.get_collection(collection.name)
    print(f"  Points count: {info.points_count}")
    print(f"  Vector size: {info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'N/A'}")
    print(f"  Status: {info.status}")

    # Sample some points
    if info.points_count > 0:
        print(f"\n  Sampling 3 random points:")
        try:
            scroll_result = client.scroll(
                collection_name=collection.name,
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            points, _ = scroll_result

            for i, point in enumerate(points, 1):
                print(f"\n    Point {i} (ID: {point.id}):")
                payload = point.payload
                for key, value in payload.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"      {key}: {value[:100]}...")
                    else:
                        print(f"      {key}: {value}")
        except Exception as e:
            print(f"    Error sampling points: {e}")

# Test search on each collection
print(f"\n{'='*60}")
print("Testing search functionality:")
print(f"{'='*60}")

test_query = "malicious python code reverse shell"
print(f"\nTest query: '{test_query}'")

encoder = SentenceTransformer("all-MiniLM-L6-v2")
query_vector = encoder.encode(test_query).tolist()

for collection in collections:
    print(f"\n  Collection: {collection.name}")

    try:
        # Try with different score thresholds
        for threshold in [0.0, 0.3, 0.5]:
            results = client.search(
                collection_name=collection.name,
                query_vector=query_vector,
                limit=3,
                score_threshold=threshold,
                with_payload=True
            )

            print(f"    Threshold {threshold}: {len(results)} results")
            if len(results) > 0:
                for i, hit in enumerate(results[:2], 1):
                    print(f"      Result {i} - Score: {hit.score:.4f}")
                    desc = hit.payload.get('description', hit.payload.get('code', 'N/A'))
                    if isinstance(desc, str):
                        print(f"        {desc[:80]}...")
    except Exception as e:
        print(f"    Error searching: {e}")

print(f"\n{'='*60}")
print("Diagnosis complete!")
print(f"{'='*60}")
