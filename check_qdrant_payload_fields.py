"""
Check what fields are actually stored in Qdrant code_samples payload.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", "6333"))
api_key = os.getenv("QDRANT_API_KEY")

if api_key:
    client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
else:
    client = QdrantClient(host=host, port=port)

print("Checking code_samples payload structure...")
print("=" * 60)

# Get 5 samples
scroll_result = client.scroll(
    collection_name="code_samples",
    limit=5,
    with_payload=True,
    with_vectors=False
)

points, _ = scroll_result

for i, point in enumerate(points, 1):
    print(f"\nPoint {i} (ID: {point.id}):")
    print(f"  Payload keys: {list(point.payload.keys())}")

    # Check if code field exists and its content
    code = point.payload.get('code', None)
    code_preview = point.payload.get('code_preview', None)

    print(f"  Has 'code' field: {code is not None}")
    if code:
        print(f"    Length: {len(code)}")
        print(f"    Preview: {code[:80]}...")

    print(f"  Has 'code_preview' field: {code_preview is not None}")
    if code_preview:
        print(f"    Length: {len(code_preview)}")
        print(f"    Preview: {code_preview[:80]}...")

    print(f"  db_id: {point.payload.get('db_id')}")
    print(f"  label: {point.payload.get('label')}")
    print(f"  chunk_index: {point.payload.get('chunk_index')}")
    print(f"  total_chunks: {point.payload.get('total_chunks')}")

print("\n" + "=" * 60)
print("Analysis complete.")