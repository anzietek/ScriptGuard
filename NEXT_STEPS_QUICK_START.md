# RAG Redesign - Quick Start Guide for Next Steps

**Current Status**: Phase 1B + Component 2 Stages 2A-2B ✅ COMPLETE

---

## 🚀 What to Do Next

### **Option 1: Run Jina-v3 Benchmark First** (Recommended)

This validates if Jina-v3 is worth migrating to before continuing.

#### Step 1: Install Dependencies

Ensure you have the required packages:
```bash
pip install transformers torch sentence-transformers qdrant-client psutil numpy
```

For Jina-v3 specifically (optional, only if testing):
```bash
pip install jina-embeddings-v3
```

#### Step 2: Run Benchmark

```bash
# From project root
python scripts/benchmark_embeddings.py --samples 1000 --output benchmark_report.json
```

**Expected Runtime**: 30-45 minutes

**What it does**:
- Loads 1000 balanced samples from your PostgreSQL database
- Tests 4 model configurations
- Measures speed, memory, retrieval quality
- Generates `benchmark_report.json` with GO/NO-GO recommendation

#### Step 3: Review Results

```bash
# View the report
cat benchmark_report.json
```

Look for:
- **decision**: "✅ GO" or "❌ NO-GO"
- **precision_delta_pct**: % change in Precision@3
- **recall_delta_pct**: % change in Recall@3
- **speed_delta_pct**: % change in throughput

#### Step 4: Make Decision

**If GO** ✅:
- Proceed with Jina-v3 migration (Phase 1C, 1D)
- Continue with Component 2 (Features)

**If NO-GO** ❌:
- Skip Jina-v3 migration
- Focus on Component 2 only (still valuable!)

---

### **Option 2: Skip Benchmark, Continue with Feature Integration**

If you want to focus on static features first (lower risk, independent of embedding model):

#### Step 1: Implement Payload Indexes (Stage 2C)

**File to modify**: `src/scriptguard/rag/code_similarity_store.py`

Add this method after the `__init__` method (~line 371):

```python
def create_feature_indexes(self):
    """Create indexes for feature-based filtering (Component 2 - Stage 2C)."""
    from qdrant_client import models

    logger.info("Creating feature payload indexes...")

    # Scalar indexes
    try:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="features.entropy",
            field_schema=models.PayloadSchemaType.FLOAT
        )
        logger.info("✓ Created index: features.entropy")
    except Exception as e:
        logger.warning(f"Index features.entropy may already exist: {e}")

    try:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="features.complexity_score",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        logger.info("✓ Created index: features.complexity_score")
    except Exception as e:
        logger.warning(f"Index features.complexity_score may already exist: {e}")

    # Boolean indexes
    for flag in ["has_network_api", "has_file_api", "has_process_api", "has_crypto_api"]:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=f"features.{flag}",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index: features.{flag}")
        except Exception as e:
            logger.warning(f"Index features.{flag} may already exist: {e}")

    # Array indexes
    for field in ["dangerous_api_calls", "suspicious_combinations"]:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=f"features.{field}",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"✓ Created index: features.{field}")
        except Exception as e:
            logger.warning(f"Index features.{field} may already exist: {e}")

    logger.info("✓ Feature indexes created successfully")
```

#### Step 2: Test Index Creation

Create `scripts/create_feature_indexes.py`:

```python
#!/usr/bin/env python3
"""Create feature indexes on existing Qdrant collection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.utils.config import load_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore

def main():
    config = load_config("config.yaml")
    qdrant_config = config.get("qdrant", {})

    store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        api_key=qdrant_config.get("api_key")
    )

    print("Creating feature indexes...")
    store.create_feature_indexes()
    print("✓ Done!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python scripts/create_feature_indexes.py
```

#### Step 3: Re-Vectorize to Add Features

If your existing collection doesn't have features yet, you need to re-vectorize:

```bash
# Option A: Run full pipeline (if you have ZenML setup)
python -m scriptguard.pipelines.train_pipeline

# Option B: Just re-vectorize existing samples (faster)
python scripts/re_vectorize_with_features.py
```

Create `scripts/re_vectorize_with_features.py`:

```python
#!/usr/bin/env python3
"""Re-vectorize existing samples to add features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.utils.config import load_config
from scriptguard.database.postgres_manager import PostgresManager
from scriptguard.steps.vectorize_samples import vectorize_samples

def main():
    config = load_config("config.yaml")
    db = PostgresManager(config)

    # Load all samples from PostgreSQL
    print("Loading samples from PostgreSQL...")
    with db.get_connection() as conn:
        result = conn.execute("""
            SELECT id, content, label, source
            FROM code_samples
            WHERE content IS NOT NULL
        """).fetchall()

    samples = [
        {
            "id": r[0],
            "content": r[1],
            "label": r[2],
            "source": r[3]
        }
        for r in result
    ]

    print(f"Loaded {len(samples)} samples")

    # Re-vectorize with features (clear existing to avoid duplicates)
    print("Re-vectorizing with features...")
    vectorize_samples(data=samples, config=config, clear_existing=True)

    print("✓ Done!")

if __name__ == "__main__":
    main()
```

---

### **Option 3: Do Both in Parallel**

Run benchmark in background while working on features:

**Terminal 1**:
```bash
# Run benchmark (takes 30-45 minutes)
python scripts/benchmark_embeddings.py --samples 1000 --output benchmark_report.json
```

**Terminal 2** (while benchmark runs):
```bash
# Implement Stage 2C (indexes)
# Add create_feature_indexes() method to code_similarity_store.py
# Test index creation
python scripts/create_feature_indexes.py

# Then implement Stage 2D (hybrid search)
# Modify search_similar_code() to accept feature_filters
```

---

## 📋 Implementation Checklist

### Phase 1A: Benchmark ⏳
- [ ] Install dependencies
- [ ] Run benchmark script
- [ ] Review results
- [ ] Make GO/NO-GO decision

### Phase 1B: Configuration ✅
- [x] Update config.yaml with Jina-v3 settings
- [x] Update embedding_service.py with Jina-v3 support

### Component 2 - Stage 2A: Schema ✅
- [x] Document feature schema

### Component 2 - Stage 2B: Pipeline ✅
- [x] Integrate features into vectorize_samples.py
- [x] Store features in code_similarity_store.py

### Component 2 - Stage 2C: Indexes ⏳
- [ ] Add create_feature_indexes() method
- [ ] Test index creation
- [ ] Verify indexes in Qdrant UI

### Component 2 - Stage 2D: Hybrid Search ⏳
- [ ] Add feature_filters parameter to search_similar_code()
- [ ] Implement _build_hybrid_filter() method
- [ ] Implement _extract_query_features() method
- [ ] Implement _rerank_by_features() method
- [ ] Test hybrid search with feature filters

### Component 2 - Stage 2E: API Integration ⏳
- [ ] Extract features in main.py
- [ ] Build feature filters based on query
- [ ] Add feature_analysis to response schema
- [ ] Test API with feature analysis

---

## 🧪 Testing Commands

### Test Feature Extraction
```python
from scriptguard.steps.feature_extraction import (
    extract_ast_features, calculate_entropy,
    extract_api_patterns, extract_string_features
)

code = """
import socket
s = socket.socket()
s.connect(('192.168.1.1', 4444))
"""

ast_features = extract_ast_features(code)
entropy = calculate_entropy(code)
api_patterns = extract_api_patterns(code)
string_features = extract_string_features(code)

print(f"Entropy: {entropy:.2f}")
print(f"Has network API: {len(api_patterns['network_apis']) > 0}")
print(f"Dangerous patterns: {ast_features['dangerous_patterns']}")
```

### Test Qdrant Features
```python
from scriptguard.rag.code_similarity_store import CodeSimilarityStore

store = CodeSimilarityStore(
    host="localhost",
    port=6333,
    collection_name="code_samples"
)

# Check if features exist
results = store.client.scroll(
    collection_name="code_samples",
    limit=1
)

if results[0]:
    sample = results[0][0]
    print("Sample payload:", sample.payload)
    print("Has features:", "features" in sample.payload)
```

### Test Hybrid Search (after Stage 2D)
```python
# Find obfuscated malware
results = store.search_similar_code(
    query_code="import socket; ...",
    k=5,
    feature_filters={
        "min_entropy": 6.0,
        "required_apis": ["has_network_api"]
    }
)

for r in results:
    print(f"Score: {r['score']:.4f}, Entropy: {r['features']['entropy']:.2f}")
```

---

## 📊 Expected Outcomes

### After Benchmark (Phase 1A)
- JSON report with metrics comparison
- GO/NO-GO decision for Jina-v3
- If GO: Confidence to proceed with migration

### After Feature Integration (Component 2)
- All samples in Qdrant have `features` field
- Feature indexes enable fast filtering
- Hybrid search works with vector + features
- API returns feature analysis for better explainability

### Combined (Both Components)
- Better embedding model (if Jina-v3 approved)
- Hybrid vector + feature search
- Improved malware detection accuracy
- Better handling of obfuscated code
- Explainable results (feature analysis)

---

## ⚠️ Troubleshooting

### Benchmark Fails
**Error**: "Failed to load Jina-v3 model"
**Solution**: Install Jina dependencies: `pip install jina-embeddings-v3`

**Error**: "PostgreSQL connection error"
**Solution**: Check database credentials in config.yaml

### Feature Extraction Slow
**Issue**: Vectorization takes 2x longer with features
**Solution**: This is expected (~10-20% overhead is normal). If >50%, check for syntax errors in code samples.

### Qdrant Index Creation Fails
**Error**: "Index already exists"
**Solution**: This is OK - indexes are idempotent. Ignore the warning.

**Error**: "Collection not found"
**Solution**: Create collection first: Run vectorization pipeline

### Features Not Showing in Qdrant
**Issue**: Querying samples shows no `features` field
**Solution**: Re-vectorize samples with updated pipeline. Existing samples don't have features until re-vectorized.

---

## 🎯 Success Indicators

### You're On Track If...
- ✅ Benchmark completes successfully
- ✅ Features appear in Qdrant payload
- ✅ Indexes created without errors
- ✅ Hybrid search returns filtered results
- ✅ API response includes feature_analysis

### You Need Help If...
- ❌ Benchmark crashes repeatedly
- ❌ Features are always empty dict
- ❌ Index creation fails with permissions error
- ❌ Hybrid search returns 0 results with filters
- ❌ API returns 500 errors after changes

---

## 📞 Support

If you encounter issues, check:
1. `RAG_REDESIGN_IMPLEMENTATION_STATUS.md` - Overall progress
2. `docs/FEATURE_SCHEMA.md` - Feature schema reference
3. Logs in `logs/` directory
4. Qdrant UI at http://localhost:6333/dashboard

---

**Ready to proceed?** Pick an option above and start implementing! 🚀
