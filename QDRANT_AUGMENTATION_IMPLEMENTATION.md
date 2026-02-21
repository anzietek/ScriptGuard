# Qdrant Augmentation Implementation Summary

## Problem Identified

**Qdrant augmentation was configured but never executing** during training pipeline runs despite:
- ✅ Config correctly set: `use_qdrant_patterns: true`
- ✅ Pipeline code exists with proper implementation
- ✅ Collections exist with data (malware_knowledge: 588 points, code_samples: 102,765 points)

**Root Cause:** ZenML pipeline design issue - runtime conditionals preventing step execution.

---

## Root Cause Analysis

### Issue: Runtime Conditionals in Pipeline Definition

**Original Code (training_pipeline.py lines 217-228):**
```python
# ❌ BROKEN: Conditional evaluated at pipeline DEFINITION time, not RUNTIME
if config.get("augmentation", {}).get("use_qdrant_patterns", False):
    qdrant_augmented_data = augment_with_qdrant_patterns(
        data=balanced_data,
        config=config
    )
else:
    qdrant_augmented_data = balanced_data
```

**Why This Failed:**
- ZenML pipelines are **compiled at definition time**
- Runtime conditionals (`if config.get(...)`) are evaluated when pipeline is **defined**, not when it **runs**
- At definition time, config values may not be correctly evaluated
- Result: Step never added to pipeline execution graph

---

## Fixes Implemented

### 1. **Pipeline Structure Fix** (training_pipeline.py)

**Changed:** Always call augmentation steps, let internal logic decide whether to augment

```python
# ✅ FIXED: Step always called, internal logic checks config
logger.info(f"Calling augment_with_qdrant_patterns step (enabled={config.get('augmentation', {}).get('use_qdrant_patterns', False)})...")
qdrant_augmented_data = augment_with_qdrant_patterns(
    data=balanced_data,
    config=config
)
```

**Benefits:**
- Step is always in pipeline execution graph
- ZenML can track and execute the step
- Internal logic (lines 34-37 in qdrant_augmentation.py) handles conditional behavior
- Diagnostic logging shows step invocation

### 2. **Balance Dataset Step Update** (advanced_augmentation.py)

**Changed:** Accept `config` parameter and check internally if balancing is enabled

```python
# ✅ FIXED: Accepts config, checks balance_dataset setting internally
@step
def balance_dataset(
    data: List[Dict],
    config: Dict = None
) -> List[Dict]:
    # Check if balancing is enabled
    if config:
        augmentation_config = config.get("augmentation", {})
        if not augmentation_config.get("balance_dataset", True):
            logger.info("Dataset balancing disabled in config. Skipping.")
            return data
```

**Benefits:**
- Consistent with augmentation step pattern
- Config-driven behavior without runtime conditionals in pipeline
- Clean skip logic with logging

### 3. **Qdrant Connection Fix** (config.yaml)

**Added:** HTTPS and API key configuration

```yaml
qdrant:
  host: ${QDRANT_HOST:-localhost}
  port: ${QDRANT_PORT:-6333}
  api_key: ${QDRANT_API_KEY:-}  # NEW: API key from .env
  use_https: false               # NEW: Disable SSL for local instance
  collection_name: "malware_knowledge"
  embedding_model: "all-MiniLM-L6-v2"
  prefer_grpc: true
```

**Why This Was Needed:**
- Qdrant client tries to use HTTPS when API key is present
- Local Qdrant doesn't have SSL certificates
- `use_https: false` forces HTTP connection
- Prevents SSL connection errors during augmentation

### 4. **Diagnostic Logging** (training_pipeline.py)

**Added:** Pre-step logging to track execution

```python
logger.info(f"Calling balance_dataset step (enabled={config.get('augmentation', {}).get('balance_dataset', True)})...")
logger.info(f"Calling augment_with_qdrant_patterns step (enabled={config.get('augmentation', {}).get('use_qdrant_patterns', False)})...")
logger.info("Calling validate_qdrant_augmentation step...")
```

**Benefits:**
- Confirms steps are being invoked by pipeline
- Shows config values at runtime
- Easy debugging if issues occur

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/scriptguard/pipelines/training_pipeline.py` | Removed runtime conditionals, always call steps, added diagnostic logging | 205-227 |
| `src/scriptguard/steps/advanced_augmentation.py` | Updated `balance_dataset` to accept `config` parameter | 296-325 |
| `src/scriptguard/steps/qdrant_augmentation.py` | Updated `validate_qdrant_augmentation` to accept optional `config` | 524-528 |
| `config.yaml` | Added `api_key` and `use_https` to qdrant section | 200-206 |

---

## Verification

### Collections Verified
```bash
$ python -c "from qdrant_client import QdrantClient; ..."
Collections: ['malware_knowledge', 'code_samples']
  malware_knowledge: 588 points
  code_samples: 102765 points
```

### Config Settings Verified
```bash
$ python -c "import yaml; config = yaml.safe_load(open('config.yaml', encoding='utf-8')); ..."
use_qdrant_patterns: True    ✓
balance_dataset: False       ✓
use_https: False            ✓
api_key set: True           ✓
```

---

## Expected Pipeline Output

When you run the training pipeline, you should now see:

```log
# NEW: Step invocation logs
2026-02-12 XX:XX:XX | INFO | Calling balance_dataset step (enabled=False)...
2026-02-12 XX:XX:XX | INFO | Dataset balancing disabled in config. Skipping.

2026-02-12 XX:XX:XX | INFO | Calling augment_with_qdrant_patterns step (enabled=True)...
2026-02-12 XX:XX:XX | INFO | Starting Qdrant augmentation with N existing samples

# Collection 1: CVE patterns from malware_knowledge
2026-02-12 XX:XX:XX | INFO | Fetching CVE patterns from 'malware_knowledge' collection...
2026-02-12 XX:XX:XX | INFO | Found 588 CVE patterns in 'malware_knowledge'
2026-02-12 XX:XX:XX | INFO | ✓ Added X CVE patterns to training data

# Collection 2: Code samples from code_samples
2026-02-12 XX:XX:XX | INFO | Fetching code samples from 'code_samples' collection...
2026-02-12 XX:XX:XX | INFO | Scrolled Y points from code_samples collection
2026-02-12 XX:XX:XX | INFO | Chunk index distribution (top 10): {0: W, 1: Z, ...}
2026-02-12 XX:XX:XX | INFO | ✓ Found W unique documents to augment (chunk_index==0)
2026-02-12 XX:XX:XX | INFO | ✓ Added W code samples with full content from PostgreSQL
2026-02-12 XX:XX:XX | INFO | ✓ Total augmented: X+W samples from Qdrant

# Validation statistics
2026-02-12 XX:XX:XX | INFO | Calling validate_qdrant_augmentation step...
╔════════════════════════════════════════════════════════╗
║         QDRANT AUGMENTATION STATISTICS                 ║
╠════════════════════════════════════════════════════════╣
║  Total Samples:        12000                           ║
║  From Qdrant:          1000 ( 8.3%)                    ║
║    - CVE patterns:      200                            ║
║    - Code samples:      800                            ║
╚════════════════════════════════════════════════════════╝
```

---

## Success Indicators

**Must Have:**
- ✅ "Calling augment_with_qdrant_patterns step" appears in logs
- ✅ "Starting Qdrant augmentation with N samples" appears
- ✅ CVE patterns retrieved from `malware_knowledge` collection
- ✅ Code samples retrieved from `code_samples` collection
- ✅ "✓ Total augmented: X samples from Qdrant" appears
- ✅ Validation statistics show non-zero Qdrant samples

**Warning Signs (if still broken):**
- ❌ No "Calling augment_with_qdrant_patterns" log → Pipeline still not calling step
- ❌ "Qdrant augmentation disabled in config" → Config not loading correctly
- ❌ SSL/HTTPS errors → `use_https` setting not applied
- ❌ "No documents with chunk_index==0 found" → Data integrity issue in Qdrant

---

## Running the Pipeline

```bash
cd C:\Users\anzie\workspace\ScriptGuard
python src/main.py --config config.yaml
```

**Monitor logs for augmentation phase:**
```bash
# In another terminal:
tail -f logs/scriptguard_$(date +%Y-%m-%d).log | grep -E "Qdrant|augment|CVE|code_samples"
```

---

## Configuration Reference

### Current Settings (config.yaml)

```yaml
# Augmentation Settings
augmentation:
  enabled: true
  variants_per_sample: 3
  use_qdrant_patterns: true     # ← ENABLED for few-shot augmentation
  max_qdrant_samples: 1000      # Fetch up to 1000 samples
  qdrant_score_threshold: 0.45  # Minimum similarity score
  balance_dataset: false        # DISABLED (using weighted loss instead)
  augment_after_split: true     # Prevents data leakage

# Qdrant Connection
qdrant:
  host: localhost
  port: 6333
  api_key: ${QDRANT_API_KEY}
  use_https: false              # ← Local instance without SSL
  collection_name: "malware_knowledge"
  embedding_model: "all-MiniLM-L6-v2"
```

---

## What Was Learned

### ZenML Pipeline Best Practices

**❌ Don't Do This:**
```python
# Bad: Runtime conditional in pipeline body
if config.get("feature_enabled"):
    result = my_step(data)
else:
    result = data
```

**✅ Do This Instead:**
```python
# Good: Always call step, let internal logic decide
result = my_step(data, config)

# Inside my_step:
@step
def my_step(data, config):
    if not config.get("feature_enabled"):
        return data  # Skip gracefully
    # ... actual logic
```

### Key Insight
ZenML pipelines compile at **definition time**, not **execution time**. All steps must be unconditionally added to the pipeline graph. Conditional behavior should be implemented **inside** step functions, not in pipeline orchestration code.

---

## Next Steps

1. **Run the pipeline** to verify augmentation executes
2. **Check training metrics** to see if few-shot examples improve performance
3. **Monitor dataset statistics** in validation logs
4. **Adjust `max_qdrant_samples`** if you want more/fewer augmented samples
5. **Tune `qdrant_score_threshold`** to control similarity matching (lower = more matches)

---

## Rollback Plan

If augmentation causes issues, disable in config:

```yaml
augmentation:
  use_qdrant_patterns: false  # Revert to standard augmentation only
```

Pipeline will automatically skip augmentation (step still executes but returns data unchanged).

No data corruption risk - augmentation only adds to dataset, doesn't modify existing samples.

---

## Questions or Issues?

If augmentation still doesn't work:

1. Check logs for "Calling augment_with_qdrant_patterns" - if missing, step still not executing
2. Verify Qdrant collections have data: `python -c "from qdrant_client import QdrantClient; ..."`
3. Check .env has `QDRANT_API_KEY` set
4. Ensure config.yaml has `use_https: false` for local Qdrant
5. Verify Python is reading config with UTF-8 encoding (Windows cp1250 issue)
