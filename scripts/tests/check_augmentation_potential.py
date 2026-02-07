#!/usr/bin/env python
"""
Quick test to see how many CVE patterns would be used in augmentation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scriptguard.rag.qdrant_store import QdrantStore

print("="*70)
print("CHECKING QDRANT AUGMENTATION POTENTIAL")
print("="*70)

# Connect to malware_knowledge
print("\n1. Connecting to malware_knowledge collection...")
store = QdrantStore(
    host='localhost',
    port=6333,
    collection_name='malware_knowledge',
    embedding_model='all-MiniLM-L6-v2'
)

info = store.get_collection_info()
points_count = info.get('points_count', 0)

print(f"   Total points in malware_knowledge: {points_count}")

# Scroll through and count what would be used
print("\n2. Counting usable CVE patterns...")
offset = None
batch_size = 100
usable = 0
skipped = 0

while True:
    scroll_result = store.client.scroll(
        collection_name='malware_knowledge',
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    points, next_offset = scroll_result

    if not points:
        break

    for point in points:
        payload = point.payload
        if payload.get('description'):
            usable += 1
        else:
            skipped += 1

    if next_offset is None:
        break

    offset = next_offset

print(f"   Usable CVE patterns: {usable}")
print(f"   Skipped (no description): {skipped}")

# Connect to code_samples
print("\n3. Checking code_samples collection...")
from scriptguard.rag.code_similarity_store import CodeSimilarityStore

code_store = CodeSimilarityStore(
    host='localhost',
    port=6333,
    collection_name='code_samples',
    enable_chunking=False
)

code_info = code_store.get_collection_info()
code_points = code_info.get('total_samples', 0)

print(f"   Total points in code_samples: {code_points}")

# Summary
print("\n" + "="*70)
print("AUGMENTATION POTENTIAL:")
print(f"  CVE patterns:  {usable:>6} samples")
print(f"  Code samples:  {code_points:>6} samples")
print(f"  TOTAL:         {usable + code_points:>6} samples available for augmentation")
print("="*70)

ratio = (usable / code_points * 100) if code_points > 0 else 0
print(f"\nRatio: CVE patterns are {ratio:.1f}% of code samples")

if ratio < 5:
    print("\n⚠️  WARNING: CVE patterns are less than 5% of code samples!")
    print("   Consider:")
    print("   1. Increasing days_back in config (currently 30)")
    print("   2. Broadening keywords")
    print("   3. Adding more exploit patterns")
else:
    print(f"\n✅ Good ratio! CVE patterns represent {ratio:.1f}% of code samples")
