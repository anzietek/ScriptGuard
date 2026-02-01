# ScriptGuard Metrics Guide

Comprehensive guide to understanding and interpreting ScriptGuard's evaluation metrics.

## Table of Contents
- [Overview](#overview)
- [Key Performance Indicators](#key-performance-indicators)
- [Metric Definitions](#metric-definitions)
- [Expected Values](#expected-values)
- [Interpreting Results](#interpreting-results)
- [Improvement Strategies](#improvement-strategies)

## Overview

ScriptGuard uses standard classification metrics to evaluate malware detection performance. Understanding these metrics is crucial for assessing model quality and making informed decisions about deployment.

### Core Metrics

The model evaluation produces four primary metrics:
- **Accuracy** - Overall correctness
- **Precision** - Reliability of malicious predictions
- **Recall** - Coverage of actual malicious scripts
- **F1 Score** - Harmonic mean of precision and recall

## Key Performance Indicators

### Production Readiness Criteria

For production deployment, ScriptGuard should meet these minimum thresholds:

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| **Accuracy** | 90% | 95% | 98% |
| **Precision** | 85% | 90% | 95% |
| **Recall** | 90% | 95% | 98% |
| **F1 Score** | 88% | 92% | 96% |

### Metric Priority by Use Case

Different deployments may prioritize different metrics:

**Security-Critical Systems** (e.g., production servers):
- **Recall > 95%** (catch all malware, accept some false positives)
- Precision > 85% (minimize but tolerate some false alarms)

**Developer Tools** (e.g., code review):
- **Precision > 90%** (minimize false positives to reduce alert fatigue)
- Recall > 90% (still catch most threats)

**Balanced** (general use):
- **F1 Score > 92%** (optimize balance)
- Accuracy > 95%

## Metric Definitions

### Accuracy

**Definition:** Proportion of correct predictions (both malicious and benign).

**Formula:**
```
Accuracy = (True Positives + True Negatives) / Total Samples
         = (TP + TN) / (TP + TN + FP + FN)
```

**Example:**
- TP (correctly identified malware): 950
- TN (correctly identified benign): 940
- FP (benign marked as malicious): 60
- FN (malware marked as benign): 50
- **Accuracy = (950 + 940) / 2000 = 0.945 = 94.5%**

**Interpretation:**
- **>95%**: Good overall performance
- **90-95%**: Acceptable, room for improvement
- **<90%**: Requires significant tuning

### Precision (Malicious Class)

**Definition:** When the model predicts "malicious", how often is it correct?

**Formula:**
```
Precision = True Positives / (True Positives + False Positives)
          = TP / (TP + FP)
```

**Example:**
- Model predicts 1010 scripts as malicious
- Of these, 950 are actually malicious (TP)
- 60 are actually benign (FP)
- **Precision = 950 / 1010 = 0.941 = 94.1%**

**Interpretation:**
- **>90%**: Low false positive rate, good for production
- **85-90%**: Acceptable, monitor false alarms
- **<85%**: High false positive rate, may cause alert fatigue

**Impact of Low Precision:**
- Users lose trust due to false alarms
- Increased manual review workload
- Legitimate scripts blocked unnecessarily

### Recall (Malicious Class)

**Definition:** Of all actual malicious scripts, how many does the model catch?

**Formula:**
```
Recall = True Positives / (True Positives + False Negatives)
       = TP / (TP + FN)
```

**Example:**
- 1000 actually malicious scripts in test set
- Model correctly identifies 950 (TP)
- Model misses 50 (FN)
- **Recall = 950 / 1000 = 0.95 = 95%**

**Interpretation:**
- **>95%**: Excellent detection coverage
- **90-95%**: Good, acceptable for most use cases
- **<90%**: Too many threats slip through, security risk

**Impact of Low Recall:**
- Malware goes undetected
- Security breaches possible
- False sense of security

### F1 Score

**Definition:** Harmonic mean of precision and recall. Balances both metrics.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
- Precision = 94.1%
- Recall = 95.0%
- **F1 = 2 × (0.941 × 0.950) / (0.941 + 0.950) = 0.945 = 94.5%**

**Interpretation:**
- **>92%**: Excellent balance
- **88-92%**: Good balance
- **<88%**: Poor balance, optimize training

**Why F1 Matters:**
- Accuracy can be misleading with imbalanced datasets
- F1 ensures both precision and recall are strong
- Better indicator of real-world performance

## Expected Values

### By Training Configuration

#### Quick Training (2 epochs, lora_r=8)
```yaml
Expected Metrics:
  Accuracy: 88-92%
  Precision: 83-88%
  Recall: 87-92%
  F1 Score: 85-90%

Use Case: Development testing, proof of concept
```

#### Standard Training (3 epochs, lora_r=16)
```yaml
Expected Metrics:
  Accuracy: 93-96%
  Precision: 89-93%
  Recall: 93-96%
  F1 Score: 91-94%

Use Case: Production deployment (recommended)
```

#### High-Quality Training (5 epochs, lora_r=32, large model)
```yaml
Expected Metrics:
  Accuracy: 96-98%
  Precision: 93-96%
  Recall: 95-98%
  F1 Score: 94-97%

Use Case: Security-critical systems, research
```

### By Dataset Size

| Dataset Size | Expected Accuracy | Expected F1 |
|--------------|-------------------|-------------|
| Small (<5k) | 85-90% | 83-88% |
| Medium (5k-10k) | 90-94% | 88-92% |
| Large (10k-20k) | 94-96% | 92-95% |
| Very Large (>20k) | 96-98% | 94-97% |

### By Base Model

| Model | Size | Expected Accuracy | Expected F1 |
|-------|------|-------------------|-------------|
| starcoder2-3b | 3B | 93-96% | 91-94% |
| codellama-7b | 7B | 95-97% | 93-95% |
| deepseek-coder-6.7b | 6.7B | 95-98% | 94-96% |
| starcoder2-7b | 7B | 95-97% | 93-96% |

## Interpreting Results

### Confusion Matrix

The evaluation produces a confusion matrix showing all prediction outcomes:

```
                    Predicted
                Benign    Malicious
Actual  Benign    940         60      (TN=940, FP=60)
        Malicious  50        950      (FN=50, TP=950)
```

**Analysis:**
- **True Negatives (TN) = 940**: Correctly identified benign (94%)
- **False Positives (FP) = 60**: Benign wrongly marked malicious (6%)
- **False Negatives (FN) = 50**: Malicious missed (5%)
- **True Positives (TP) = 950**: Correctly identified malicious (95%)

### Common Patterns

#### Pattern 1: High Precision, Low Recall
```
Precision: 96%
Recall: 82%
F1: 88%
```

**Diagnosis:** Model is too conservative
- Only flags very obvious malware
- Misses subtle or obfuscated threats

**Solution:**
- Lower classification threshold
- Add more diverse malicious samples
- Increase augmentation variants

#### Pattern 2: High Recall, Low Precision
```
Precision: 78%
Recall: 97%
F1: 86%
```

**Diagnosis:** Model is too aggressive
- Flags many benign scripts as malicious
- High false alarm rate

**Solution:**
- Raise classification threshold
- Add more diverse benign samples
- Improve data quality filtering

#### Pattern 3: Low Both Metrics
```
Precision: 75%
Recall: 73%
F1: 74%
```

**Diagnosis:** Model needs improvement
- Insufficient training data or epochs
- Poor model configuration

**Solution:**
- Increase dataset size (target 10k+ samples)
- Train longer (5 epochs)
- Increase LoRA rank to 32
- Try larger base model

#### Pattern 4: Balanced High Metrics (Ideal)
```
Precision: 94%
Recall: 95%
F1: 94.5%
Accuracy: 95%
```

**Diagnosis:** Model is production-ready
- Good balance of precision and recall
- Strong overall performance

## Improvement Strategies

### To Improve Precision (Reduce False Positives)

1. **Add More Benign Data**
   ```yaml
   data_sources:
     huggingface:
       max_samples: 15000  # Increase benign samples
   ```

2. **Improve Benign Data Quality**
   ```yaml
   validation:
     min_code_lines: 10  # More substantial code
     max_comment_ratio: 0.3  # Less commented-out code
   ```

3. **Reduce Augmentation**
   ```yaml
   augmentation:
     variants_per_sample: 1  # Less aggressive augmentation
   ```

4. **Adjust Classification Threshold**
   ```python
   # In inference
   threshold = 0.7  # Increase from 0.5 (default)
   ```

### To Improve Recall (Reduce False Negatives)

1. **Add More Malicious Data**
   ```yaml
   data_sources:
     malwarebazaar:
       max_samples: 200  # More malware samples
   ```

2. **Increase Augmentation**
   ```yaml
   augmentation:
     enabled: true
     variants_per_sample: 3  # More obfuscated variants
   ```

3. **Add Diverse Malware Types**
   - Obfuscated scripts
   - Encoded payloads
   - Polymorphic variants
   - Recent/novel attack patterns

4. **Lower Classification Threshold**
   ```python
   # In inference
   threshold = 0.3  # Decrease from 0.5 (default)
   ```

### To Improve Overall Accuracy

1. **Increase Training Data**
   - Target: 10,000+ samples (balanced)
   - Diverse sources (GitHub, MalwareBazaar, etc.)

2. **Balance Dataset**
   ```yaml
   augmentation:
     balance_dataset: true
     target_balance_ratio: 1.0  # Equal malicious/benign
   ```

3. **Optimize Training**
   ```yaml
   training:
     num_epochs: 5
     lora_r: 32
     learning_rate: 0.0001
   ```

4. **Use Larger Model**
   ```yaml
   training:
     model_id: "deepseek-ai/deepseek-coder-6.7b-instruct"
   ```

### To Improve F1 Score

F1 score improves when both precision and recall improve together:

1. **Ensure Balanced Dataset**
   ```python
   # Check balance
   malicious_count = len([s for s in data if s['label'] == 'malicious'])
   benign_count = len([s for s in data if s['label'] == 'benign'])
   ratio = malicious_count / benign_count
   print(f"Balance ratio: {ratio:.2f}")  # Target: 0.8-1.2
   ```

2. **Add High-Quality Data from Both Classes**
   - Diverse malware families
   - Real-world benign code patterns

3. **Optimize Training Configuration**
   - Use recommended settings from TUNING_GUIDE.md
   - Monitor validation metrics during training

## Monitoring Metrics

### During Training

Monitor these metrics to detect issues early:

```python
# Training logs show:
Epoch 1/3: train_loss=0.45, train_acc=0.87
Epoch 2/3: train_loss=0.32, train_acc=0.92
Epoch 3/3: train_loss=0.25, train_acc=0.95
```

**Good Pattern:** Loss decreases, accuracy increases
**Bad Pattern (Overfitting):** Train acc very high (>98%), validation acc low

### After Training

Always evaluate on a separate test set:

```python
from scriptguard.steps.model_evaluation import evaluate_model

metrics = evaluate_model(
    adapter_path="./model_checkpoints/final_adapter",
    test_dataset=test_data
)

print(f"""
Model Evaluation Results:
========================
Accuracy:  {metrics['accuracy']:.1%}
Precision: {metrics['precision']:.1%}
Recall:    {metrics['recall']:.1%}
F1 Score:  {metrics['f1']:.1%}

Production Ready: {metrics['accuracy'] >= 0.95 and metrics['f1'] >= 0.92}
""")
```

## Benchmark Comparisons

### Industry Standards

ScriptGuard metrics compared to similar systems:

| System Type | Typical Accuracy | Typical F1 |
|-------------|------------------|------------|
| Traditional AV (signature-based) | 85-92% | 82-88% |
| ML-based Static Analysis | 90-95% | 88-93% |
| **ScriptGuard (LLM-based)** | **93-96%** | **91-94%** |
| Human Security Experts | 96-99% | 95-98% |

### ScriptGuard Goals

- **Match or exceed** traditional ML approaches (>93% accuracy)
- **Approach** human expert performance on common malware patterns
- **Minimize** false positives to reduce alert fatigue (<10% FP rate)
- **Maximize** recall to prevent security breaches (>95% recall)

## Next Steps

- Review [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for training instructions
- Review [TUNING_GUIDE.md](./TUNING_GUIDE.md) for optimization strategies
- Monitor metrics during training with ZenML dashboard
- Set up alerts for metric degradation in production
