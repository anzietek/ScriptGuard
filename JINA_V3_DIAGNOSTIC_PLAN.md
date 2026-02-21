# Jina-v3 vs UniXcoder Diagnostic Plan

## Problem Summary

RAG system has poor performance on diverse test set:
- **Original test (40 samples)**: 87.50% accuracy, 88.37% F1
- **New test (99 samples)**: 66.67% accuracy, 71.79% F1
- **CRITICAL**: Benign utility categories have 0% accuracy:
  - csv, json, database, logging, datetime, email, threading, time, subprocess, sys

## Hypothesis

UniXcoder is a **weak embedding model** with:
- Only 768 dimensions (vs Jina-v3 1024d)
- 512 token context limit (vs Jina-v3 8192 tokens)
- 125M parameters (vs Jina-v3 570M)
- Older training data (pre-2023)

This causes poor semantic understanding → cannot distinguish benign utilities from malicious code.

## Diagnostic Process

### Phase 1: Verify UniXcoder Bottleneck

**Objective**: Confirm that UniXcoder embeddings fail to separate benign from malicious

**Script**: `scripts/diagnose_unixcoder_embeddings.py`

**What it does**:
1. Loads 40 benign + 40 malicious samples (Level 1-3)
2. Generates UniXcoder embeddings
3. Measures cosine similarity:
   - Within benign samples (should be high)
   - Within malicious samples (should be high)
   - Between benign and malicious (should be LOW)

**Expected Result**:
- If cross-similarity >= 0.65 → UniXcoder is the problem ✅
- If cross-similarity < 0.55 → Problem is elsewhere ❌

**Run**:
```bash
python scripts/diagnose_unixcoder_embeddings.py
```

**Output Example**:
```
UNIXCODER EMBEDDING ANALYSIS
================================================================================

1. INTRA-CLASS SIMILARITY (Same Label)
   Benign-Benign:       0.72 ± 0.12
   Malicious-Malicious: 0.75 ± 0.10

2. INTER-CLASS SIMILARITY (Different Labels)
   Benign-Malicious:    0.68 ± 0.15  ← TOO HIGH!

3. SEPARATION QUALITY
   Benign clustering - Cross similarity: 0.04

   ❌ POOR SEPARATION: Cross-similarity (0.68) >= 0.65
      UniXcoder CANNOT distinguish benign from malicious code!
      This confirms the bottleneck hypothesis.

CONCLUSION
================================================================================

✅ HYPOTHESIS CONFIRMED: UniXcoder is the bottleneck!
   → Proceed to Phase 2: Benchmark Jina-v3
```

---

### Phase 2: Benchmark Jina-v3

**Objective**: Test if Jina-v3 embeddings improve separation

**Script**: `scripts/benchmark_jina_vs_unixcoder.py`

**What it does**:
1. Loads same test samples
2. Generates embeddings with BOTH models
3. Compares separation metrics
4. Makes GO/NO-GO decision

**Decision Criteria**:
- ✅ **GO**: Jina-v3 cross-similarity < 0.50 AND UniXcoder >= 0.65
  - **Action**: Proceed to Phase 3 (End-to-End RAG test)
- ❌ **NO-GO**: Jina-v3 shows similar poor separation
  - **Action**: Investigate other root causes

**Run**:
```bash
python scripts/benchmark_jina_vs_unixcoder.py
```

**Output Example**:
```
COMPARATIVE ANALYSIS
================================================================================

1. CROSS-SIMILARITY (Benign-Malicious)
   UniXcoder:  0.68
   Jina-v3:    0.42  ← MUCH BETTER!
   Improvement: 0.26 (+38.2%)

2. SEPARATION GAP (Benign Clustering - Cross)
   UniXcoder:  0.04
   Jina-v3:    0.30  ← MUCH BETTER!
   Improvement: +0.26 (+650%)

GO/NO-GO DECISION
================================================================================

Criteria:
  [✓] UniXcoder cross-similarity >= 0.65: 0.68
  [✓] Jina-v3 cross-similarity < 0.50: 0.42

✅ GO: Proceed to Phase 3 (End-to-End RAG Test)
   Jina-v3 shows significant improvement (+38% separation)
   Expected F1 score improvement: 71.79% → 85%+
```

---

### Phase 3: End-to-End RAG Test

**Objective**: Validate Jina-v3 improves real-world RAG performance

**Script**: `scripts/test_rag_with_jina.py`

**What it does**:
1. Creates temporary Qdrant collection `code_samples_jina_test`
2. Vectorizes 1000 samples (500 benign, 500 malicious) from database with Jina-v3
3. Runs RAG inference on 99-sample test set
4. Compares with UniXcoder baseline

**Success Criteria**:
- ✅ F1 score >= 85% (vs 71.79%)
- ✅ Benign utility accuracy >= 80% (vs 0%)
- ✅ FPR not increased (< 20%)

**Run**:
```bash
# Full test (vectorize + test)
python scripts/test_rag_with_jina.py --max-training-samples 1000 --k 10

# Quick re-test (use existing collection)
python scripts/test_rag_with_jina.py --skip-vectorization --k 10
```

**Output Example**:
```
RAG PERFORMANCE METRICS (Jina-v3)
================================================================================

1. OVERALL METRICS
   Accuracy:  86.87%
   Precision: 88.24%
   Recall:    85.71%
   F1 Score:  86.96%

2. BENIGN UTILITY ACCURACY
   Categories: csv, json, database, logging, datetime, email, threading, time, subprocess, sys
   Accuracy: 85.0% (17/20)

COMPARISON WITH UNIXCODER BASELINE
================================================================================

Metric                    UniXcoder       Jina-v3         Improvement
----------------------------------------------------------------------
F1 Score                     71.79%         86.96%         +21.1%
Accuracy                     66.67%         86.87%         +30.3%
Benign Utility Accuracy       0.0%          85.0%         +85.0pp

PHASE 3 DECISION
================================================================================

Success Criteria:
  [✅] F1 >= 85%: 86.96%
  [✅] Benign Utility >= 80%: 85.00%
  [✅] FPR not increased: 18.00%

✅ ALL CRITERIA MET: Proceed to Phase 4 (Full Migration)
```

---

### Phase 4: Full Migration (Conditional)

**Objective**: Migrate entire Qdrant collection to Jina-v3

**Trigger**: Only if Phase 3 shows F1 >= 85%

**Steps**:

1. **Re-vectorize all samples**:
```bash
# Modify config.yaml
code_embedding:
  model: "jinaai/jina-embeddings-v3"
  pooling_strategy: "mean_pooling"
  normalize: true
  max_code_length: 8192

  jina_v3:
    enabled: true
    task_adapter: "retrieval.v2"
    output_dimension: 1024
    max_length: 8192
    trust_remote_code: true

# Run vectorization pipeline
python -m scriptguard.steps.vectorize_samples
```

2. **Create production collection**:
```python
# In vectorize_samples.py, update collection name
collection_name = "code_samples_jina_v3"
```

3. **Update API config**:
```yaml
# config.yaml
rag:
  collection_name: "code_samples_jina_v3"
  embedding_model: "jinaai/jina-embeddings-v3"
```

4. **Restart API and test**:
```bash
# Restart API
python src/scriptguard/api/main.py

# Test benign utility code
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: $KEY" \
  -d '{"script_content": "import csv\nwith open(\"data.csv\") as f:\n    reader = csv.DictReader(f)", "include_rag": true}'

# Expected: is_malicious=false, confidence < 0.3
```

5. **Monitor production metrics**:
- Check F1 score maintains >= 85%
- Verify benign utility accuracy >= 80%
- Monitor API latency (should increase < 10%)

**Rollback Plan**:
- Keep old `code_samples` collection (UniXcoder)
- Instant rollback via config change if issues occur

---

## Alternative: If Jina-v3 Doesn't Fix It

If Phase 2 or 3 shows NO improvement, investigate:

### Root Cause A: Training Data Domain Mismatch
- **Issue**: Benign samples in Qdrant are from web frameworks (django, flask), not utility code (csv, json)
- **Test**: Check Qdrant benign samples for csv/json/database patterns
- **Fix**: Augment training data with benign utility code from PyPI packages

### Root Cause B: Chunk-Level Retrieval Loss
- **Issue**: Code samples chunked → context loss → can't distinguish intent
- **Test**: Compare retrieval quality on full documents vs chunks
- **Fix**: Disable chunking for samples < 2048 tokens

### Root Cause C: Imbalanced Label Distribution
- **Issue**: 56.2% malicious samples → model biased toward "malicious"
- **Test**: Balance training data to 50/50
- **Fix**: Undersample malicious or oversample benign

---

## Quick Start

### 1. Run Phase 1 (5 minutes)
```bash
python scripts/diagnose_unixcoder_embeddings.py
```

**If cross-similarity >= 0.65 → UniXcoder is the problem**

### 2. Run Phase 2 (10 minutes)
```bash
python scripts/benchmark_jina_vs_unixcoder.py
```

**If Jina-v3 cross-similarity < 0.50 → Proceed to Phase 3**

### 3. Run Phase 3 (30-45 minutes)
```bash
python scripts/test_rag_with_jina.py --max-training-samples 1000 --k 10
```

**If F1 >= 85% → Migrate to Jina-v3 in production**

### 4. (Optional) Full Migration (2-3 hours)
```bash
# Update config.yaml with Jina-v3 settings
# Re-run vectorization pipeline
# Update API config
# Restart and test
```

---

## Timeline

**Fast Track** (if Jina-v3 fixes it):
- Phase 1: 5 minutes (diagnosis)
- Phase 2: 10 minutes (benchmark)
- Phase 3: 30-45 minutes (end-to-end test)
- Phase 4: 2-3 hours (migration)
- **Total**: 3-4 hours

**Slow Track** (if Jina-v3 doesn't fix it):
- Phase 1-3: Same as fast track
- Investigation: 1-2 weeks (training data, chunking, balancing)

---

## Files Created

### Phase 1-3 Scripts
- ✅ `scripts/diagnose_unixcoder_embeddings.py` - Verify UniXcoder bottleneck
- ✅ `scripts/benchmark_jina_vs_unixcoder.py` - Compare models
- ✅ `scripts/test_rag_with_jina.py` - End-to-end RAG test

### Configuration Files
- `config.yaml` - Will be updated in Phase 4

### Core Files (already support Jina-v3)
- `src/scriptguard/rag/embedding_service.py` - Already has Jina-v3 support
- `src/scriptguard/rag/code_similarity_store.py` - Works with any embedding model
- `src/scriptguard/steps/vectorize_samples.py` - Will use updated config

---

## Expected Results

### Best Case (Jina-v3 fixes it)
- Phase 1: Confirms UniXcoder cross-sim >= 0.65
- Phase 2: Jina-v3 cross-sim < 0.50 (+38% improvement)
- Phase 3: F1 score 86%+, benign utility 85%+
- **Action**: Full migration to Jina-v3

### Worst Case (Jina-v3 doesn't help)
- Phase 1: UniXcoder cross-sim >= 0.65
- Phase 2: Jina-v3 cross-sim >= 0.65 (no improvement)
- Phase 3: F1 score still ~72%
- **Action**: Investigate training data domain mismatch, chunking, or label balance

---

## Next Steps

1. ✅ Run Phase 1 diagnostic
2. ⏳ Analyze results and decide on Phase 2
3. ⏳ If GO, run Phase 2 benchmark
4. ⏳ If GO, run Phase 3 end-to-end test
5. ⏳ If successful, execute Phase 4 migration
