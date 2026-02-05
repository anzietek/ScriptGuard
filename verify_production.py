#!/usr/bin/env python
"""
FINAL VERIFICATION SCRIPT
Confirms that CVE enrichment works correctly in production code.
This is the ONLY verification script you need.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("PRODUCTION CODE VERIFICATION")
print("="*70)

# Test 1: Verify CVEFeedSource works
print("\n[TEST 1] CVEFeedSource.fetch_recent_cves()")
print("-" * 70)

from scriptguard.data_sources.cve_feeds import CVEFeedSource

cve_source = CVEFeedSource()
cves = cve_source.fetch_recent_cves(days=30, keywords=["script", "code execution"])

if cves:
    print(f"✅ SUCCESS: Fetched {len(cves)} CVEs")
    print(f"   First CVE: {cves[0]['cve_id']}")
else:
    print(f"❌ FAILED: No CVEs fetched")
    sys.exit(1)

# Test 2: Verify Qdrant augmentation
print("\n[TEST 2] Qdrant Augmentation (malware_knowledge)")
print("-" * 70)

from scriptguard.rag.qdrant_store import QdrantStore

store = QdrantStore(
    host='localhost',
    port=6333,
    collection_name='malware_knowledge',
    embedding_model='all-MiniLM-L6-v2'
)

info = store.get_collection_info()
cve_count = info.get('points_count', 0)

print(f"   Total CVE patterns in Qdrant: {cve_count}")

if cve_count < 100:
    print(f"⚠️  WARNING: Only {cve_count} CVE patterns (recommended: 400+)")
    print(f"   Run: python scripts/enrich_cve_final.py")
elif cve_count < 400:
    print(f"✅ GOOD: {cve_count} CVE patterns (could be better)")
else:
    print(f"✅ EXCELLENT: {cve_count} CVE patterns")

# Test 3: Verify code_samples
print("\n[TEST 3] Qdrant Augmentation (code_samples)")
print("-" * 70)

from scriptguard.rag.code_similarity_store import CodeSimilarityStore

code_store = CodeSimilarityStore(
    host='localhost',
    port=6333,
    collection_name='code_samples',
    enable_chunking=False
)

code_info = code_store.get_collection_info()
code_count = code_info.get('total_samples', 0)

print(f"   Total code samples in Qdrant: {code_count}")

if code_count == 0:
    print(f"❌ FAILED: No code samples in Qdrant")
    sys.exit(1)
else:
    print(f"✅ SUCCESS: {code_count} code samples")

# Test 4: Calculate ratio
print("\n[TEST 4] CVE/Code Ratio")
print("-" * 70)

ratio = (cve_count / code_count * 100) if code_count > 0 else 0
print(f"   Ratio: {ratio:.2f}% (CVE patterns / code samples)")

if ratio < 1:
    print(f"❌ CRITICAL: Ratio too low ({ratio:.2f}%)")
    print(f"   Need at least 1% CVE patterns relative to code samples")
    print(f"   Target: {int(code_count * 0.01)} CVE patterns (have {cve_count})")
    recommendation = int(code_count * 0.01) - cve_count
    print(f"   → Need {recommendation} more CVE patterns")
    print(f"   → Increase days_back in config.yaml or broaden keywords")
elif ratio < 3:
    print(f"⚠️  WARNING: Ratio is low ({ratio:.2f}%)")
    print(f"   Recommended: 3-5% for balanced augmentation")
else:
    print(f"✅ GOOD: Ratio is healthy ({ratio:.2f}%)")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  CVE Fetching:      {'✅ Working' if cves else '❌ Failed'}")
print(f"  Qdrant CVE Store:  {cve_count} patterns")
print(f"  Qdrant Code Store: {code_count} samples")
print(f"  CVE/Code Ratio:    {ratio:.2f}%")
print("="*70)

if cves and cve_count >= 400 and ratio >= 1:
    print("\n🎉 ALL CHECKS PASSED - Production code is working correctly!")
    sys.exit(0)
elif cves and cve_count >= 100:
    print("\n✅ Working, but could be improved (add more CVE patterns)")
    sys.exit(0)
else:
    print("\n❌ Issues detected - see warnings above")
    sys.exit(1)
