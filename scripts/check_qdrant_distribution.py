#!/usr/bin/env python3
"""Check label distribution in Qdrant collection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from collections import Counter

def main():
    # Connect to Qdrant
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(
        host="localhost",
        port=6333,
        api_key=api_key,
        https=False
    )

    # Count labels for parent docs only (chunk_index=0)
    labels = []
    offset = None

    print("Counting labels in Qdrant (chunk_index=0 only)...")

    while True:
        result = client.scroll(
            collection_name="code_samples",
            limit=100,
            with_payload=["label"],
            offset=offset,
            scroll_filter={"must": [{"key": "chunk_index", "match": {"value": 0}}]}
        )

        points, offset = result
        if not points:
            break

        for p in points:
            label = p.payload.get("label")
            if label:
                labels.append(label)

        if len(labels) % 1000 == 0:
            print(f"  Scanned {len(labels)} documents...")

        if offset is None:
            break

    # Count
    counter = Counter(labels)
    total = len(labels)

    print(f"\n{'='*60}")
    print("QDRANT LABEL DISTRIBUTION (Parent Documents)")
    print('='*60)
    print(f"\nTotal parent docs: {total}")
    for label, count in counter.most_common():
        pct = count / total * 100
        print(f"  {label:12s}: {count:6d} ({pct:5.1f}%)")

    if "malicious" in counter and "benign" in counter:
        ratio = counter["malicious"] / counter["benign"]
        print(f"\nRatio: {ratio:.2f}x malicious per benign")

        if 0.8 <= ratio <= 1.2:
            print("[OK] Distribution is BALANCED")
        elif ratio > 1.5:
            print(f"[WARN] IMBALANCED: {ratio:.1f}x more malicious!")
        elif ratio < 0.67:
            print(f"[WARN] IMBALANCED: {1/ratio:.1f}x more benign!")

    print('='*60)

if __name__ == "__main__":
    main()
