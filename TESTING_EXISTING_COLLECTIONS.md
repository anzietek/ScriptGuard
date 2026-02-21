# Testing with Existing Collections

This guide shows how to test model performance using existing Qdrant collections without re-vectorizing data.

## Overview

The test scripts now support a `--use-existing` flag that allows you to:
1. **Skip collection creation** - Use existing vectorized data
2. **Run multiple tests** - Test repeatedly after migration/changes
3. **Save time** - No need to re-vectorize 40 samples each time

## Prerequisites

Collections must exist in Qdrant before using `--use-existing`:

### UniXcoder Collection
- **Name:** `code_samples_balanced`
- **Dimensions:** 768
- **Points:** 40 (20 benign + 20 malicious)
- **Create with:** `python scripts/test_hybrid_balanced.py`

### Jina-v3 Collection
- **Name:** `code_samples_jina_v3`
- **Dimensions:** 1024
- **Points:** 40 (20 benign + 20 malicious)
- **Create with:** `python scripts/test_jina_v3_comparison.py` (without --use-existing)

## Usage

### Test UniXcoder with Existing Collection

```bash
# Full evaluation with existing collection
python scripts/test_hybrid_full_eval.py --use-existing

# Output:
# [MODE] Using existing collection (no re-vectorization)
# [OK] Collection 'code_samples_balanced' has 40 points
# [OK] UniXcoder ready (device: cuda)
# ...
# Accuracy: 90.00%, F1: 90.48%
```

### Test Jina-v3 with Existing Collection

```bash
# Comparison test with existing collection
python scripts/test_jina_v3_comparison.py --use-existing

# Output:
# [MODE] Using existing collection (no re-vectorization)
# [OK] Collection 'code_samples_jina_v3' has 40 points
# [OK] Jina-v3 model ready
# ...
# Accuracy: 82.50%, F1: 84.44%
```

### Test Both Models at Once

```bash
# Windows (PowerShell)
powershell scripts/test_models_existing.ps1

# Linux/Mac (Bash)
bash scripts/test_models_existing.sh
```

## Creating Collections (First Time)

If collections don't exist, create them first:

### Create UniXcoder Collection

```bash
# Create code_samples_balanced collection
python scripts/test_hybrid_balanced.py

# This will:
# 1. Create collection with 768 dimensions
# 2. Vectorize 40 samples with UniXcoder
# 3. Store with features for hybrid search
```

### Create Jina-v3 Collection

```bash
# Create code_samples_jina_v3 collection
python scripts/test_jina_v3_comparison.py

# This will:
# 1. Download Jina-v3 model (if needed)
# 2. Create collection with 1024 dimensions
# 3. Vectorize 40 samples with Jina-v3
# 4. Run full comparison vs UniXcoder
```

## Workflow Example

### Scenario: Testing After Migrating to Production Collection

```bash
# 1. Create collections once
python scripts/test_hybrid_balanced.py       # UniXcoder
python scripts/test_jina_v3_comparison.py    # Jina-v3

# 2. Now test repeatedly without re-vectorizing
python scripts/test_hybrid_full_eval.py --use-existing       # Test UniXcoder
python scripts/test_jina_v3_comparison.py --use-existing     # Test Jina-v3

# 3. After making changes to search logic, test again
python scripts/test_hybrid_full_eval.py --use-existing       # Quick re-test
```

## Error Handling

### Collection Not Found

```bash
$ python scripts/test_hybrid_full_eval.py --use-existing

[ERROR] Collection not found!
        Run without --use-existing to create it first
        Or run: python scripts/test_hybrid_balanced.py
```

**Solution:** Create the collection first without `--use-existing`

### Wrong Collection Name

If you migrated to a different collection name, you need to either:
1. Update the script's `collection_name` variable, or
2. Recreate the expected collection name

## Performance Comparison

### Without --use-existing (Full Vectorization)
- **Time:** ~2-3 minutes (UniXcoder), ~10-15 minutes (Jina-v3)
- **Use when:** First time setup, or after changing samples

### With --use-existing (Skip Vectorization)
- **Time:** ~30 seconds (both models)
- **Use when:** Testing search logic, comparing models, debugging

## Advanced: Testing Custom Collections

To test your production collection:

```python
# Edit the script and change collection name
collection_name = "your_production_collection"  # Instead of "code_samples_balanced"

# Then run with --use-existing
python scripts/test_hybrid_full_eval.py --use-existing
```

## Summary

| Command | When to Use | Time |
|---------|-------------|------|
| Without `--use-existing` | First time, or data changed | 2-15 min |
| With `--use-existing` | Quick tests, comparing models | ~30 sec |

**Best Practice:** Create collections once, then use `--use-existing` for all subsequent tests.
