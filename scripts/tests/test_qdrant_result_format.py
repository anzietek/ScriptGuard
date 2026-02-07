"""Debug Qdrant search results format"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

client = QdrantClient(":memory:")

# Create test collection
client.create_collection(
    collection_name="test",
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
)

# Insert test point
client.upsert(
    collection_name="test",
    points=[
        models.PointStruct(
            id=1,
            vector=[1, 2, 3, 4],
            payload={"text": "test"}
        )
    ]
)

# Check available methods
print("QdrantClient has 'search':", hasattr(client, 'search'))
print("QdrantClient has 'query':", hasattr(client, 'query'))
print("QdrantClient has 'query_points':", hasattr(client, 'query_points'))

search_methods = [m for m in dir(client) if 'search' in m.lower() or 'query' in m.lower()]
print("\nMethods with 'search' or 'query':")
for m in search_methods:
    if not m.startswith('_'):
        print(f"  - {m}")

# Try the correct method
if hasattr(client, 'query_points'):
    print("\nUsing query_points...")
    results = client.query_points(
        collection_name="test",
        query=[1, 2, 3, 4],
        limit=1
    )
    print("Results:", results)
elif hasattr(client, 'query'):
    print("\nUsing query...")
    results = client.query(
        collection_name="test",
        query_vector=[1, 2, 3, 4],
        limit=1
    )
    print("Results:", results)
