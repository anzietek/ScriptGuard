# Phase 1 Diagnosis Results

## Summary

**HYPOTHESIS REJECTED**: UniXcoder is NOT the bottleneck!

## Embedding Similarity Analysis

```
UNIXCODER EMBEDDING ANALYSIS
================================================================================

1. INTRA-CLASS SIMILARITY (Within Same Label)
   Benign-Benign:       0.2217 ± 0.1124
   Malicious-Malicious: 0.2250 ± 0.1124

2. INTER-CLASS SIMILARITY (Between Different Labels)
   Benign-Malicious:    0.1724 ± 0.0935

3. SEPARATION QUALITY
   Benign clustering - Cross similarity: 0.0493
```

## Interpretation

### ✅ Good News
- **Cross-similarity (0.1724) << 0.65**: UniXcoder CAN distinguish benign from malicious code
- **Positive separation gap (0.0493)**: Benign samples cluster together better than with malicious
- **No need for Jina-v3 migration**: Current embedding model is sufficient

### ❌ Bad News
- The poor RAG performance (66.67% accuracy, 0% benign utility accuracy) is **NOT** caused by weak embeddings
- The problem is somewhere else in the pipeline

## Root Cause Analysis

Since embeddings are good but RAG fails on benign utilities, the likely culprits are:

### 1. **Training Data Domain Mismatch** (Most Likely)

**Hypothesis**: Qdrant contains benign samples from web frameworks (Django, Flask, Ansible) but NOT benign utility code (csv, json, database operations).

**Test**:
```bash
# Check what benign categories exist in Qdrant
python scripts/check_qdrant_distribution.py --collection code_samples --analyze-categories
```

**Expected Finding**:
- Qdrant has `web_scraping`, `api_client`, `database_ops` (framework code)
- Qdrant LACKS `csv`, `json`, `logging`, `datetime` (utility code)
- When RAG searches for "csv.DictReader", it finds NO similar benign examples
- Falls back to nearest match → web scraping or malicious CSV parsing → wrong classification

**Fix**:
1. Identify missing benign categories (csv, json, database, logging, datetime, email, threading, time, subprocess, sys)
2. Collect benign examples from:
   - PyPI packages: `pandas`, `csv` module, `sqlite3`, `logging` module
   - GitHub repos: data processing scripts, ETL pipelines, system utilities
   - Python stdlib examples: official docs, tutorials
3. Add 50-100 samples per category to training data
4. Re-vectorize and test

---

### 2. **RAG Retrieval Configuration Issues**

**Hypothesis**: RAG is retrieving irrelevant or low-quality examples.

**Test**:
```python
# Analyze RAG retrieval quality for failing samples
python scripts/analyze_rag_retrieval.py \
  --test-categories "csv,json,database,logging" \
  --show-top-k 10
```

**Possible Issues**:
- **k=10 too high**: Includes noisy/irrelevant examples
- **No score threshold**: Low-similarity examples pollute few-shot prompt
- **Chunking artifacts**: Retrieves partial code chunks missing context
- **Reranking bias**: Feature boosting amplifies malicious matches

**Fix**:
- Reduce k to 5
- Add strict score threshold (>= 0.60)
- Disable chunking for code < 2048 tokens
- Disable feature boosting (already done, but verify)

---

### 3. **Prompt Engineering / Label Leakage**

**Hypothesis**: Few-shot prompt construction has issues.

**Test**:
```python
# Inspect actual prompts sent to model
python scripts/debug_few_shot_prompts.py \
  --sample "import csv; csv.DictReader(open('data.csv'))" \
  --show-prompt
```

**Possible Issues**:
- Labels visible in prompt (already fixed, but verify)
- Poor example selection (all malicious examples are obfuscated, benign are not)
- Insufficient diversity in few-shot examples

**Fix**:
- Verify label masking is working
- Improve example diversity (mix simple and complex malicious)
- Add explicit instruction: "Focus on INTENT, not syntax similarity"

---

### 4. **Label Distribution Imbalance**

**Hypothesis**: 56.2% malicious training data biases RAG toward "malicious" predictions.

**Test**:
```python
# Test RAG with balanced vs imbalanced collections
python scripts/test_label_balance_impact.py \
  --balance-ratio "50:50" "56:44" "60:40"
```

**Fix**:
- Rebalance Qdrant collection to 50/50
- Or use weighted sampling during retrieval
- Or adjust decision threshold based on base rate

---

## Recommended Action Plan

### Priority 1: Training Data Domain Analysis (HIGH IMPACT)

**Objective**: Verify that Qdrant lacks benign utility code

**Steps**:
1. Query Qdrant for benign samples with categories: csv, json, database, logging, datetime
2. Count matches (expect: 0 or very few)
3. If confirmed, collect 50-100 benign utility samples
4. Add to training data and re-vectorize

**Script to Create**:
```python
# scripts/analyze_training_data_coverage.py
# Check Qdrant for benign utility coverage
```

**Timeline**: 1-2 days

---

### Priority 2: RAG Retrieval Quality Analysis (MEDIUM IMPACT)

**Objective**: Debug why RAG retrieves bad examples for benign utilities

**Steps**:
1. Log RAG retrieval for failing test cases
2. Inspect top-k results (scores, labels, code snippets)
3. Identify patterns (low scores? wrong labels? irrelevant code?)

**Script to Create**:
```python
# scripts/debug_rag_retrieval.py
# Show RAG results for failing samples
```

**Timeline**: 1 day

---

### Priority 3: Verify Existing Fixes (LOW EFFORT)

**Objective**: Ensure label leakage fix and feature boosting disable are working

**Steps**:
1. Test API on "hello world" sample
2. Inspect few-shot prompt (no labels visible?)
3. Check feature boosting is disabled (config + code)

**Timeline**: 1 hour

---

## Next Steps

1. **DON'T** proceed to Phase 2 (Jina-v3 benchmark) - not needed
2. **DON'T** proceed to Phase 4 (migration) - would waste time
3. **DO** investigate training data domain mismatch (Priority 1)
4. **DO** analyze RAG retrieval quality (Priority 2)
5. **DO** verify existing fixes are working (Priority 3)

---

## Updated Diagnostic Script

Create `scripts/analyze_training_data_coverage.py`:

```python
#!/usr/bin/env python3
"""
Analyze Qdrant training data coverage for benign utility categories.

Check if Qdrant contains benign samples for failing categories:
- csv, json, database, logging, datetime, email, threading, time, subprocess, sys

If coverage is low (<5 samples per category), this explains the 0% accuracy.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def analyze_coverage():
    # Connect to Qdrant
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    # Failing benign categories
    failing_categories = [
        "csv", "json", "database", "logging", "datetime",
        "email", "threading", "time", "subprocess", "sys"
    ]

    print("TRAINING DATA COVERAGE ANALYSIS")
    print("="*80)

    # Scroll through benign samples
    benign_samples = []
    for record in client.scroll(
        collection_name="code_samples",
        scroll_filter={"must": [{"key": "label", "match": {"value": "benign"}}]},
        limit=10000
    )[0]:
        benign_samples.append(record.payload)

    print(f"\nTotal benign samples in Qdrant: {len(benign_samples)}")

    # Count by category
    category_counts = {}
    for sample in benign_samples:
        # Check if code mentions category keywords
        code = sample.get("code", "").lower()
        for cat in failing_categories:
            if cat in code:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nBenign samples containing failing category keywords:")
    for cat in failing_categories:
        count = category_counts.get(cat, 0)
        status = "[OK]" if count >= 5 else "[FAIL]"
        print(f"  {status} {cat:15s} {count:3d} samples")

    # Decision
    low_coverage = [cat for cat in failing_categories if category_counts.get(cat, 0) < 5]

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if len(low_coverage) >= 5:
        print(f"\n[CONFIRMED] Training data domain mismatch detected!")
        print(f"\n{len(low_coverage)}/{len(failing_categories)} categories have <5 samples:")
        print(f"  {', '.join(low_coverage)}")
        print(f"\nThis explains why RAG fails on benign utility code.")
        print(f"\nNext steps:")
        print(f"  1. Collect benign samples for missing categories")
        print(f"  2. Add to database and re-vectorize")
        print(f"  3. Re-test RAG performance")
    else:
        print(f"\n[REJECTED] Training data coverage is adequate")
        print(f"  Only {len(low_coverage)} categories have <5 samples")
        print(f"  Problem may be elsewhere (prompting, retrieval config)")

if __name__ == "__main__":
    analyze_coverage()
```

---

## Conclusion

**Phase 1 Result**: UniXcoder embeddings are NOT the problem (cross-sim = 0.1724 << 0.65)

**Root Cause**: Most likely **training data domain mismatch**
- Qdrant has benign web framework code (Django, Flask)
- Qdrant LACKS benign utility code (csv, json, logging)
- RAG can't find similar benign examples → defaults to malicious

**Action**: Investigate training data coverage (Priority 1) before considering Jina-v3 migration
