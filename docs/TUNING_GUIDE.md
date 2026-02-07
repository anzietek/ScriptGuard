# ScriptGuard Tuning Guide

Comprehensive guide for optimizing ScriptGuard's performance and accuracy.

## Table of Contents
- [QLoRA Hyperparameters](#qlora-hyperparameters)
- [Training Hyperparameters](#training-hyperparameters)
- [Model Selection](#model-selection)
- [Data Optimization](#data-optimization)
- [Inference Optimization](#inference-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)

## QLoRA Hyperparameters

### LoRA Rank (r)

Controls the rank of low-rank adaptation matrices.

**Default:** `16`

**Options:**
- `r=8`: Faster training, smaller model, lower accuracy
- `r=16`: **Recommended** - Good balance
- `r=32`: Slower training, better accuracy
- `r=64`: Diminishing returns, much slower

**Example:**
```yaml
training:
  lora_r: 16
```

**When to adjust:**
- Increase if model underfits (low training accuracy)
- Decrease if training is too slow or overfits

### LoRA Alpha

Scaling factor for LoRA updates.

**Default:** `32` (typically 2× rank)

**Formula:** `alpha = 2 * r`

**Options:**
- `alpha=16` (for r=8)
- `alpha=32` (for r=16)
- `alpha=64` (for r=32)

**Example:**
```yaml
training:
  lora_r: 16
  lora_alpha: 32
```

### LoRA Dropout

Dropout probability for LoRA layers.

**Default:** `0.05`

**Options:**
- `0.0`: No dropout, faster convergence
- `0.05`: **Recommended** - Good regularization
- `0.1`: More regularization, slower convergence

**When to adjust:**
- Increase if overfitting (high train accuracy, low validation)
- Decrease if underfitting

### Target Modules

Which model layers to apply LoRA.

**Default:**
```yaml
training:
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**Options:**
- Query/Value only: `["q_proj", "v_proj"]` (faster)
- All attention: `["q_proj", "v_proj", "k_proj", "o_proj"]` (recommended)
- Full: Add `"mlp"` or gate/up/down projections (slower, more accurate)

## Training Hyperparameters

### Learning Rate

**Default:** `0.0002` (2e-4)

**Recommended ranges:**
- **Conservative:** `1e-4` - Safe, slower convergence
- **Standard:** `2e-4` - **Recommended**
- **Aggressive:** `5e-4` - Faster, risk of instability

**Finding optimal LR:**
```python
# LR range test
learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

for lr in learning_rates:
    config["training"]["learning_rate"] = lr
    metrics = train_model(config)
    print(f"LR {lr}: Loss={metrics['loss']}, Acc={metrics['accuracy']}")
```

### Batch Size

**Default:** `4`

**GPU Memory Guidelines:**
- **8GB VRAM:** batch_size = 1-2
- **16GB VRAM:** batch_size = 2-4
- **24GB VRAM:** batch_size = 4-8
- **40GB+ VRAM:** batch_size = 8-16

**Effective batch size:**
```
effective_batch = batch_size × gradient_accumulation_steps
```

**Example:**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective = 16
```

### Gradient Accumulation

Accumulate gradients over multiple steps.

**Default:** `4`

**When to use:**
- Limited GPU memory
- Want larger effective batch size
- Stabilize training

**Trade-off:** Slower training speed

### Number of Epochs

**Default:** `3`

**Guidelines:**
- **1 epoch:** Quick test, likely underfit
- **3 epochs:** **Recommended** for most cases
- **5 epochs:** If dataset is small (<5k samples)
- **10+ epochs:** Risk of overfitting

**Monitor:** Stop if validation loss stops decreasing

### Warmup Steps

Number of steps to linearly increase learning rate.

**Default:** `100`

**Formula:** `warmup_steps = 0.1 * total_steps`

**Example:**
```python
total_steps = (len(dataset) // batch_size) * num_epochs
warmup_steps = int(0.1 * total_steps)
```

### Weight Decay

L2 regularization strength.

**Default:** `0.01`

**Options:**
- `0.0`: No regularization
- `0.01`: **Recommended**
- `0.1`: Strong regularization

## Model Selection

### Base Models Comparison

| Model | Size | Speed | Accuracy | GPU Memory | Recommended For |
|-------|------|-------|----------|------------|-----------------|
| **starcoder2-3b** | 3B | Fast | Good | 16GB | **Production** |
| **codellama-7b** | 7B | Medium | Better | 24GB | High accuracy |
| **deepseek-coder-6.7b** | 6.7B | Medium | Best | 24GB | Code-specialized |
| **starcoder2-7b** | 7B | Medium | Better | 24GB | Balanced |
| **codellama-13b** | 13B | Slow | Best | 40GB+ | Research |

### Choosing a Model

**For production (speed priority):**
```yaml
training:
  model_id: "bigcode/starcoder2-3b"
```

**For accuracy (quality priority):**
```yaml
training:
  model_id: "deepseek-ai/deepseek-coder-6.7b-instruct"
```

**For balance:**
```yaml
training:
  model_id: "codellama/CodeLlama-7b-hf"
```

## Data Optimization

### Dataset Size

**Recommended minimum:**
- Malicious: 5,000 samples
- Benign: 5,000 samples
- Total: 10,000+ samples

**Optimal:**
- Malicious: 10,000+ samples
- Benign: 10,000+ samples

### Class Balance

**Target ratio:** `1.0` (equal malicious/benign)

**Acceptable range:** `0.8 - 1.25`

**Check balance:**
```python
from scriptguard.monitoring import check_balance

is_balanced = check_balance(samples, target_ratio=1.0, tolerance=0.2)
```

**Fix imbalance:**
```yaml
augmentation:
  balance_dataset: true
  target_balance_ratio: 1.0
  balance_method: "undersample"  # or "oversample"
```

### Data Quality

**Validation settings:**
```yaml
validation:
  validate_syntax: true  # Remove invalid Python
  skip_syntax_errors: true
  min_length: 50  # Min characters
  max_length: 50000  # Max characters
  min_code_lines: 5  # Min lines of actual code
  max_comment_ratio: 0.5  # Max 50% comments
```

### Augmentation

**Recommended settings:**
```yaml
augmentation:
  enabled: true
  variants_per_sample: 2  # Generate 2 variants per malicious sample
  techniques:
    - "base64"
    - "hex"
    - "rename_vars"
```

**When to increase variants:**
- Small malicious dataset (<3k samples)
- High false negatives in evaluation

**When to disable:**
- Large dataset (>20k samples)
- Overfitting issues

## Inference Optimization

### Model Quantization

**4-bit quantization (recommended):**
```python
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

**Performance:**
- **Speed:** 2-3× faster
- **Memory:** 4× less
- **Accuracy:** ~1-2% drop

### Batch Inference

```python
# Analyze multiple scripts at once
scripts = [script1, script2, script3, ...]
results = analyzer.analyze_batch(scripts, batch_size=8)
```

**Speed:** 5-10× faster than individual analysis

### Caching

```python
analyzer = ScriptGuardInference(
    model_path="./models/scriptguard-model",
    cache_size=1000  # Cache last 1000 results
)
```

### GPU Optimization

**Enable TF32 (on Ampere+ GPUs):**
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Flash Attention 2:**
```yaml
training:
  use_flash_attention_2: true
```

**Requires:** `pip install flash-attn`

## Evaluation Metrics

### Primary Metrics

**Accuracy:**
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Target:** >95%

**Precision (malicious class):**
```
precision = TP / (TP + FP)
```

**Target:** >90% (minimize false positives)

**Recall (malicious class):**
```
recall = TP / (TP + FN)
```

**Target:** >95% (minimize false negatives)

**F1 Score:**
```
f1 = 2 * (precision * recall) / (precision + recall)
```

**Target:** >92%

### Evaluation Code

```python
from scriptguard.steps.model_evaluation import evaluate_model

metrics = evaluate_model(adapter_path="./model_checkpoints/final_adapter")

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")
```

### Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
y_true = [...]
y_pred = [...]

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Troubleshooting

### Problem: Low Accuracy (<80%)

**Solutions:**
1. Increase training data (target 10k+ samples)
2. Increase LoRA rank: `lora_r: 32`
3. Train for more epochs: `num_epochs: 5`
4. Use larger base model: `codellama-7b`

### Problem: Overfitting

**Symptoms:** High train accuracy, low validation accuracy

**Solutions:**
1. Increase dropout: `lora_dropout: 0.1`
2. Add weight decay: `weight_decay: 0.1`
3. Reduce epochs: `num_epochs: 2`
4. Increase dataset size
5. Enable augmentation

### Problem: High False Positives

**Solutions:**
1. Increase precision threshold
2. Balance dataset (more benign samples)
3. Improve benign data quality
4. Reduce augmentation variants

### Problem: High False Negatives

**Solutions:**
1. Increase recall threshold
2. Add more malicious samples
3. Increase augmentation: `variants_per_sample: 3`
4. Add obfuscated malware samples

### Problem: Slow Training

**Solutions:**
1. Reduce batch size + increase gradient accumulation
2. Use smaller base model: `starcoder2-3b`
3. Reduce LoRA rank: `lora_r: 8`
4. Enable mixed precision: `bf16: true`
5. Use Flash Attention 2
6. **Use Unsloth** (enabled by default)

### Problem: Out of Memory

**Solutions:**
1. Reduce batch size: `batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Reduce max sequence length: `max_seq_length: 1024`
4. Enable gradient checkpointing
5. Use smaller model

## Hyperparameter Search

### Grid Search Example

```python
import yaml

param_grid = {
    "lora_r": [8, 16, 32],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "num_epochs": [3, 5]
}

best_f1 = 0
best_params = None

for r in param_grid["lora_r"]:
    for lr in param_grid["learning_rate"]:
        for epochs in param_grid["num_epochs"]:
            # Update config
            config["training"]["lora_r"] = r
            config["training"]["learning_rate"] = lr
            config["training"]["num_epochs"] = epochs

            # Train and evaluate
            metrics = train_and_evaluate(config)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_params = {"r": r, "lr": lr, "epochs": epochs}

print(f"Best params: {best_params}, F1: {best_f1:.2%}")
```

## Best Practices

1. **Start with defaults** - Only tune if needed
2. **Monitor validation metrics** - Watch for overfitting
3. **Use early stopping** - Save time and prevent overfitting
4. **Test on real data** - Evaluate on production-like samples
5. **Version control** - Track config changes with results
6. **Document experiments** - Note what works and what doesn't

## Recommended Configurations

### Quick Training (Fast)
```yaml
training:
  model_id: "bigcode/starcoder2-3b"
  lora_r: 8
  batch_size: 4
  num_epochs: 2
  learning_rate: 0.0002
```

### Balanced (Recommended)
```yaml
training:
  model_id: "bigcode/starcoder2-3b"
  lora_r: 16
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 0.0002
```

### High Accuracy (Slow)
```yaml
training:
  model_id: "deepseek-ai/deepseek-coder-6.7b-instruct"
  lora_r: 32
  batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 5
  learning_rate: 0.0001
```

## Next Steps

- Review [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for training instructions
- Review [USAGE_GUIDE.md](./USAGE_GUIDE.md) for deployment
- Monitor metrics with WandB or TensorBoard
- Experiment with different configurations
