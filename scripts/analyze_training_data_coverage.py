#!/usr/bin/env python3
"""
Analyze Qdrant training data coverage for benign utility categories.

Check if Qdrant contains benign samples for failing categories:
- csv, json, database, logging, datetime, email, threading, time, subprocess, sys

If coverage is low (<5 samples per category), this explains the 0% accuracy.
"""

import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def analyze_category_keywords(code):
    """Detect category keywords in code."""
    code_lower = code.lower()

    categories_found = []

    # CSV
    if any(kw in code_lower for kw in ["csv", "dictreader", "dictwriter"]):
        categories_found.append("csv")

    # JSON
    if any(kw in code_lower for kw in ["json.loads", "json.dumps", "json.load", "json.dump"]):
        categories_found.append("json")

    # Database
    if any(kw in code_lower for kw in ["sqlite", "cursor.execute", "database", "sql", "db.query"]):
        categories_found.append("database")

    # Logging
    if any(kw in code_lower for kw in ["logging.info", "logging.error", "logger.", "log.info"]):
        categories_found.append("logging")

    # Datetime
    if any(kw in code_lower for kw in ["datetime.now", "datetime.date", "timedelta", "strftime"]):
        categories_found.append("datetime")

    # Email
    if any(kw in code_lower for kw in ["smtplib", "email.message", "send_mail", "mimemultipart"]):
        categories_found.append("email")

    # Threading
    if any(kw in code_lower for kw in ["threading.thread", "thread.start", "lock.acquire", "threadpool"]):
        categories_found.append("threading")

    # Time
    if any(kw in code_lower for kw in ["time.sleep", "time.time", "time.strftime"]):
        categories_found.append("time")

    # Subprocess
    if any(kw in code_lower for kw in ["subprocess.run", "subprocess.popen", "subprocess.call"]):
        categories_found.append("subprocess")

    # Sys
    if any(kw in code_lower for kw in ["sys.argv", "sys.exit", "sys.path", "sys.stdin"]):
        categories_found.append("sys")

    return categories_found


def analyze_coverage():
    """Analyze training data coverage for failing categories."""

    print("="*80)
    print("TRAINING DATA COVERAGE ANALYSIS")
    print("="*80)

    # Connect to Qdrant
    print("\nConnecting to Qdrant...")

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY")

    if api_key:
        # Try HTTP first (some Qdrant Cloud setups use HTTP with API key)
        try:
            client = QdrantClient(
                url=f"http://{host}:{port}",
                api_key=api_key,
                timeout=60,
                prefer_grpc=False
            )
            print(f"[OK] Connected via HTTP with API key")
        except Exception as e:
            print(f"[INFO] HTTP connection failed, trying HTTPS...")
            # Fallback to HTTPS
            client = QdrantClient(
                url=f"https://{host}:{port}",
                api_key=api_key,
                timeout=60,
                prefer_grpc=False,
                https=True
            )
            print(f"[OK] Connected via HTTPS with API key")
    else:
        # Local connection without API key
        client = QdrantClient(
            host=host,
            port=port,
            timeout=60,
            prefer_grpc=False
        )
        print(f"[OK] Connected locally without API key")

    collection_name = "code_samples"
    print(f"[OK] Connected to collection: {collection_name}")

    # Get collection info
    try:
        info = client.get_collection(collection_name)
        print(f"[OK] Collection points: {info.points_count}")
    except Exception as e:
        print(f"[ERROR] Failed to get collection info: {e}")
        return

    # Failing benign categories (from test results)
    failing_categories = [
        "csv", "json", "database", "logging", "datetime",
        "email", "threading", "time", "subprocess", "sys"
    ]

    print(f"\nAnalyzing coverage for {len(failing_categories)} failing categories:")
    print(f"  {', '.join(failing_categories)}")

    # Scroll through benign samples
    print(f"\nScrolling benign samples...")
    benign_samples = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value="benign")
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True
        )

        benign_samples.extend(results)

        if next_offset is None:
            break

        offset = next_offset
        print(f"  Scrolled {len(benign_samples)} samples...", end='\r')

    print(f"\n[OK] Scrolled {len(benign_samples)} benign samples")

    # Analyze category coverage
    print("\nAnalyzing category keywords in code...")

    category_samples = {cat: [] for cat in failing_categories}
    category_counts = Counter()

    for record in benign_samples:
        code = record.payload.get("code", "")
        source = record.payload.get("source", "unknown")

        # Detect categories
        categories = analyze_category_keywords(code)

        for cat in categories:
            if cat in category_samples:
                category_counts[cat] += 1
                if len(category_samples[cat]) < 5:  # Store first 5 examples
                    category_samples[cat].append({
                        "code": code[:200] + "..." if len(code) > 200 else code,
                        "source": source
                    })

    # Display results
    print("\n" + "="*80)
    print("CATEGORY COVERAGE RESULTS")
    print("="*80)

    print(f"\n{'Category':<15} {'Samples':<10} {'Status':<10}")
    print("-" * 40)

    low_coverage = []
    adequate_coverage = []

    for cat in failing_categories:
        count = category_counts.get(cat, 0)

        if count < 5:
            status = "[FAIL]"
            low_coverage.append(cat)
        elif count < 20:
            status = "[WARNING]"
            low_coverage.append(cat)
        else:
            status = "[OK]"
            adequate_coverage.append(cat)

        print(f"{cat:<15} {count:<10} {status:<10}")

    # Show examples for low-coverage categories
    if low_coverage:
        print("\n" + "="*80)
        print("LOW COVERAGE CATEGORIES - EXAMPLES")
        print("="*80)

        for cat in low_coverage:
            print(f"\n{cat.upper()} ({category_counts.get(cat, 0)} samples):")

            if category_samples[cat]:
                for i, sample in enumerate(category_samples[cat], 1):
                    print(f"\n  Example {i} (source: {sample['source']}):")
                    print(f"  {sample['code']}")
            else:
                print("  [NO SAMPLES FOUND]")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if len(low_coverage) >= 5:
        print(f"\n[CONFIRMED] Training data domain mismatch detected!")
        print(f"\n  {len(low_coverage)}/{len(failing_categories)} categories have <20 samples:")
        print(f"  {', '.join(low_coverage)}")
        print(f"\n  This explains why RAG fails on benign utility code:")
        print(f"  - RAG searches for similar benign examples")
        print(f"  - Finds NO benign csv/json/database/logging samples")
        print(f"  - Falls back to nearest match (often malicious or web framework)")
        print(f"  - Incorrectly classifies benign utility as malicious")

        print(f"\n  RECOMMENDED FIX:")
        print(f"  1. Collect 50-100 benign samples per category:")
        print(f"     - PyPI packages: pandas, csv module, sqlite3, logging module")
        print(f"     - GitHub repos: data processing scripts, ETL pipelines")
        print(f"     - Python stdlib: official docs, tutorials")
        print(f"  2. Add to code_samples database table")
        print(f"  3. Re-vectorize: python -m scriptguard.steps.vectorize_samples")
        print(f"  4. Re-test RAG performance")

    else:
        print(f"\n[REJECTED] Training data coverage is adequate")
        print(f"  Only {len(low_coverage)} categories have <20 samples")
        print(f"  Problem may be elsewhere:")
        print(f"  - Prompt engineering (label leakage)")
        print(f"  - RAG retrieval configuration (k, threshold, reranking)")
        print(f"  - Chunk-level context loss")

    # Analyze sources
    print("\n" + "="*80)
    print("BENIGN SAMPLE SOURCES")
    print("="*80)

    source_counts = Counter()
    for record in benign_samples:
        source = record.payload.get("source", "unknown")
        source_counts[source] += 1

    print(f"\nBenign samples by source:")
    for source, count in source_counts.most_common():
        print(f"  {source:<30} {count:>5} samples")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Import models here to avoid circular import
    from qdrant_client import models

    analyze_coverage()
