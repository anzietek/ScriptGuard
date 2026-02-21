# Jina-v3 Migration - Quick Start Guide

## TL;DR - Run Phase 1 Now

```bash
# Step 1: Create Jina-v3 test collection (~2 hours)
python scripts/create_jina_test_collection.py --samples 5000

# Step 2: Test RAG with Jina-v3 (~30 min)
python scripts/test_rag_with_new_samples.py \
  --collection code_samples_jina_test \
  --k 10 \
  --no-features

# Step 3: Compare with baseline
python scripts/test_rag_with_new_samples.py \
  --collection code_samples \
  --k 10 \
  --no-features
```

---

## Decision Criteria (After Step 2)

**GO to Phase 2 if:**
- ✅ Accuracy ≥ 78% (baseline: 66.67%)
- ✅ F1 Score ≥ 82% (baseline: 72.27%)
- ✅ Benign utility ≥ 50% (baseline: 0%)

**NO-GO if:**
- ❌ F1 Score < 75%
- ❌ Benign utility < 30%

---

## Expected Output (Step 1)

```
================================================================================
PHASE 1: CREATE JINA-V3 TEST COLLECTION
================================================================================

Configuration:
  Samples: 5000
  Collection: code_samples_jina_test
  Model: jinaai/jina-embeddings-v3
  Dimension: 768 (Matryoshka)
  Task adapter: retrieval.v2
  Max length: 8192 tokens

Initializing Jina-v3 embedding service...
✓ Jina-v3 service ready
  Dimension: 768
  Max length: 8192 tokens
  Task adapter: retrieval.v2

✓ Connected to Qdrant: localhost:6333

Creating temporary collection: code_samples_jina_test
✓ Collection created: code_samples_jina_test

Fetching 5000 training samples from database...
✓ Fetched 5000 samples:
  Benign:    2500 (50.0%)
  Malicious: 2500 (50.0%)

Vectorizing 5000 samples with Jina-v3...
Generating embeddings (batch_size=32)...
  Progress: 320/5000 samples vectorized
  Progress: 640/5000 samples vectorized
  ...
  Progress: 5000/5000 samples vectorized
✓ Generated embeddings: (5000, 768)
  L2 norm stats: mean=1.0000, min=0.9998, max=1.0002

Uploading to Qdrant collection 'code_samples_jina_test'...
  Uploaded 100/5000 points
  Uploaded 200/5000 points
  ...
  Uploaded 5000/5000 points
✓ Upload complete: 5000 points

Collection verification:
  Total points: 5000
  Vector dimension: 768

================================================================================
PHASE 1 COMPLETE
================================================================================

✓ Test collection created: code_samples_jina_test
✓ Total samples: 5000

Next step:
  python scripts/test_rag_with_new_samples.py \
    --collection code_samples_jina_test \
    --k 10 \
    --strategy majority_vote \
    --no-features

================================================================================
```

---

## Expected Output (Step 2 - Success)

```
================================================================================
RAG TEST WITH NEW SAMPLES (Level 3 Expansion)
================================================================================

Configuration:
  Collection: code_samples_jina_test
  K neighbors: 10
  Strategy: majority_vote
  Feature boosting: DISABLED (embeddings only)

Test set: 99 samples
  Benign: 49
  Malicious: 50

Initializing RAG store...
[OK] Connected to collection: code_samples_jina_test

================================================================================
EVALUATION (k=10, strategy=majority_vote)
================================================================================

Processing 99 queries...
  Progress: 10/99...
  Progress: 20/99...
  ...
  Progress: 90/99...

[OK] Evaluated 99 queries

================================================================================
RESULTS
================================================================================

Overall Metrics:
  Accuracy:  84.85%  ← TARGET: ≥78% ✅
  Precision: 86.27%
  Recall:    88.00%
  F1 Score:  87.12%  ← TARGET: ≥82% ✅
  FPR:       18.37%

Confusion Matrix:
  True Positives:  44 (malicious correctly identified)
  True Negatives:  40 (benign correctly identified)
  False Positives:  9 (benign marked as malicious)
  False Negatives:  6 (malicious marked as benign)

================================================================================
CATEGORY BREAKDOWN
================================================================================

Accuracy by category:
  [OK] network_exfiltration         : 10/10 (100.0%)
  [OK] ransomware_simulation        :  5/ 5 (100.0%)
  [OK] reverse_shell                :  5/ 5 (100.0%)
  [OK] process_injection            :  5/ 5 (100.0%)
  [OK] datetime                     :  4/ 5 ( 80.0%)
  [OK] csv                          :  4/ 5 ( 80.0%)  ← WAS 0%! ✅
  [OK] json                         :  4/ 5 ( 80.0%)  ← WAS 0%! ✅
  [OK] database                     :  3/ 5 ( 60.0%)  ← WAS 0%! ✅
  [OK] logging                      :  3/ 5 ( 60.0%)  ← WAS 0%! ✅
  [FAIL] email                      :  2/ 5 ( 40.0%)

================================================================================
SUMMARY
================================================================================

RAG Performance on NEW Data:
  Test samples: 99 (NOT in database)
  Accuracy: 84.85%
  F1 Score: 87.12%
  Error rate: 15.15%

[PASS] EXCELLENT: F1 >= 90% - RAG works very well!

This test uses samples NOT in the database, so it's a real evaluation.
================================================================================

✅ DECISION: GO TO PHASE 2 (Production Migration)
```

---

## Expected Output (Step 2 - Failure)

```
================================================================================
RESULTS
================================================================================

Overall Metrics:
  Accuracy:  68.69%  ← TARGET: ≥78% ❌
  Precision: 72.55%
  Recall:    74.00%
  F1 Score:  73.26%  ← TARGET: ≥82% ❌
  FPR:       36.73%

================================================================================
SUMMARY
================================================================================

RAG Performance on NEW Data:
  Test samples: 99 (NOT in database)
  Accuracy: 68.69%
  F1 Score: 73.26%
  Error rate: 31.31%

[POOR] POOR: F1 < 70% - RAG needs significant improvement

❌ DECISION: NO-GO - Jina-v3 does not solve the problem

Root cause is NOT the embedding model. Investigate:
  - Training data domain mismatch
  - Label quality issues
  - Feature engineering needed
```

---

## What to Do After Results

### ✅ If GO (F1 ≥ 82%):

```bash
# Phase 2: Production Migration

# Step 1: Re-vectorize all 20,869 samples (~6 hours)
python -m scriptguard.steps.vectorize_samples \
  --collection code_samples_jina_v3 \
  --model jinaai/jina-embeddings-v3 \
  --dimension 768 \
  --max-length 8192

# Step 2: Update config.yaml
# Change:
#   collection_name: "code_samples"
#   model: "microsoft/unixcoder-base"
# To:
#   collection_name: "code_samples_jina_v3"
#   model: "jinaai/jina-embeddings-v3"
#   jina_v3.enabled: true

# Step 3: Restart API
pkill -f "python.*start_api.py"
python start_api.py

# Step 4: Verify
python scripts/test_rag_with_new_samples.py --k 10
```

### ❌ If NO-GO (F1 < 75%):

**Stop migration. Investigate root causes:**

1. **Training data quality**
   ```bash
   python scripts/analyze_training_data_coverage.py
   ```

2. **Label quality issues**
   ```bash
   python scripts/analyze_false_positives.py
   ```

3. **Feature engineering**
   ```bash
   # Re-enable features and test
   python scripts/test_rag_with_new_samples.py \
     --collection code_samples_jina_test \
     --k 10
   # (without --no-features)
   ```

---

## Troubleshooting

### Script fails at "Initializing Jina-v3"
```bash
# Install dependencies
pip install transformers sentence-transformers torch

# Test model download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)"
```

### Script fails at "Fetching samples"
```bash
# Check PostgreSQL connection
python -c "from scriptguard.database.connection import get_postgres_connection; print(get_postgres_connection())"

# Check sample count
python -c "from scriptguard.database.connection import get_postgres_connection; c = get_postgres_connection().cursor(); c.execute('SELECT COUNT(*) FROM code_samples WHERE label = \\'benign\\''); print('Benign:', c.fetchone()[0]); c.execute('SELECT COUNT(*) FROM code_samples WHERE label = \\'malicious\\''); print('Malicious:', c.fetchone()[0])"
```

### Script fails at "Creating collection"
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

### Out of memory
```bash
# Reduce samples
python scripts/create_jina_test_collection.py --samples 2000

# Or reduce batch size (edit script line 155)
```

---

## Progress Tracking

- [ ] Step 1: Create test collection (2 hours)
- [ ] Step 2: Test RAG (30 min)
- [ ] Step 3: Evaluate results (15 min)
- [ ] Decision: GO / NO-GO
- [ ] (If GO) Phase 2: Production migration

---

## Time Estimate

- **Best case:** 2.5 hours
- **Worst case:** 3 hours (if model download is slow)

---

Ready to start! Run Step 1 now 🚀
