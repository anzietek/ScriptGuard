"""Check how many CVE patterns are in malware_knowledge collection."""
from scriptguard.rag.qdrant_store import QdrantStore

# Connect to malware_knowledge
store = QdrantStore(
    host='localhost',
    port=6333,
    collection_name='malware_knowledge',
    embedding_model='all-MiniLM-L6-v2'
)

# Get collection info
info = store.get_collection_info()
points_count = info.get('points_count', 0)

print(f"\n{'='*60}")
print(f"Collection: malware_knowledge")
print(f"{'='*60}")
print(f"Total points: {points_count}")

if points_count > 0:
    # Fetch a few samples to check content
    print(f"\nFetching sample data...")

    scroll_result = store.client.scroll(
        collection_name='malware_knowledge',
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    points, _ = scroll_result

    print(f"\nFound {len(points)} samples. First few:")
    print(f"{'='*60}")

    for i, point in enumerate(points[:5], 1):
        payload = point.payload
        print(f"\nSample {i}:")
        print(f"  CVE ID: {payload.get('cve_id', 'N/A')}")
        print(f"  Description: {payload.get('description', 'N/A')[:100]}...")
        print(f"  Has pattern: {bool(payload.get('pattern'))}")
        print(f"  Severity: {payload.get('severity', 'N/A')}")

print(f"\n{'='*60}")
