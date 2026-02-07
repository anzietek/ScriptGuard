# ScriptGuard Training Guide

Complete guide for training the ScriptGuard malware detection model from scratch.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Data Collection](#data-collection)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Monitoring Training](#monitoring-training)
- [Model Export](#model-export)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (recommended: RTX 3090, RTX 4090, A100, A6000)
- **RAM**: Minimum 32GB system RAM
- **Storage**: 100GB+ free space for datasets and models
- **OS**: Linux (Ubuntu 22.04+), Windows 10/11 (WSL2 recommended), or macOS

### Software Requirements
- **Python 3.12**
- **CUDA 12.4** (for GPU training)
- **uv** package manager (recommended)
- Docker (optional, for containerized deployment)

### API Keys (Optional but Recommended)
- **GitHub Personal Access Token**: Increases API rate limits
- **NVD API Key**: Faster CVE data fetching
- **Hugging Face Token**: For private dataset access and model uploads
- **WandB API Key**: For experiment tracking

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
```

### 2. Install Dependencies
We recommend using `uv` for fast and reliable dependency management.

```bash
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates virtual environment automatically)
uv sync
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GITHUB_API_TOKEN=ghp_your_token_here
NVD_API_KEY=your_nvd_key_here
HUGGINGFACE_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
```

## Data Collection

### Understanding Data Sources

ScriptGuard collects data from multiple sources:

1. **GitHub** - Malicious and benign code samples
2. **MalwareBazaar** - Fresh malware samples
3. **Hugging Face Datasets** - Large-scale benign code and malware collections
4. **CVE Feeds** - Exploit patterns from NVD

### Configure Data Sources

Edit `config.yaml`:

```yaml
data_sources:
  github:
    enabled: true
    fetch_malicious: true
    fetch_benign: true
    max_samples_per_keyword: 200
    max_files_per_repo: 500

  malwarebazaar:
    enabled: true
    max_samples: 1000

  huggingface:
    enabled: false  # Gated dataset example
    datasets:
      - "codeparrot/github-code"
    max_samples: 20000

  cve_feeds:
    enabled: true
    days_back: 119
```

## Configuration

### Training Configuration

Edit `config.yaml` training section. ScriptGuard uses **Unsloth** for optimized training.

```yaml
training:
  model_id: "bigcode/starcoder2-3b"  # Base model
  output_dir: "${MODEL_OUTPUT_DIR:-/workspace/models/scriptguard-model}"

  # QLoRA settings
  use_qlora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

  # Hyperparameters
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 2e-4
  max_seq_length: 4096

  # Optimization
  bf16: true
  optim: "adamw_8bit"
  use_flash_attention_2: true
```

### Validation Settings

```yaml
validation:
  validate_syntax: true
  skip_syntax_errors: true
  min_length: 50
  max_length: 100000
  min_code_lines: 5
```

### Augmentation Settings

```yaml
augmentation:
  enabled: true
  variants_per_sample: 3
  balance_dataset: true
  target_balance_ratio: 1.0
  balance_method: "oversample"
  
  # Qdrant CVE Pattern Augmentation
  use_qdrant_patterns: true
```

## Training Pipeline

### Quick Start

Run the training pipeline with default configuration:

```bash
uv run python src/main.py
```

### Step-by-Step Pipeline

The training pipeline executes the following steps:

#### 1. Data Ingestion
Fetches data from all configured sources (GitHub, MalwareBazaar, HF, CVE).

#### 2. Data Validation
Validates syntax, length, and encoding of collected samples.

#### 3. Quality Filtering
Filters out low-quality samples based on code/comment ratio and line counts.

#### 4. Feature Extraction
Extracts AST features, entropy, and API patterns.

#### 5. Data Augmentation
Generates polymorphic variants and augments with CVE patterns from Qdrant.

#### 6. Dataset Balancing
Balances the malicious/benign ratio to prevent model bias.

#### 7. Preprocessing
Tokenizes and formats data for the model.

#### 8. Model Training
Trains the model using QLoRA with Unsloth optimizations and Flash Attention 2.

#### 9. Evaluation
Evaluates model performance on a hold-out test set.

## Monitoring Training

### ZenML Dashboard

Start ZenML server:
```bash
uv run zenml up
```

Access dashboard at `http://localhost:8237` (default port).

### WandB Integration

Configure WandB in `.env`:
```env
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=scriptguard
```

View experiments at [wandb.ai](https://wandb.ai).

### Training Logs

Check logs in the `logs/` directory or console output.

## Model Export

### Export Trained Model

After training completes, the adapter is saved to the output directory. You can load it using `PeftModel` or merge it.

### Merge LoRA Adapter with Base Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoder2-3b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./model_checkpoints/final_adapter")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/scriptguard-merged")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
tokenizer.save_pretrained("./models/scriptguard-merged")
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
**Solution**: Reduce batch size or increase gradient accumulation.
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

#### CUDA Version Mismatch
**Solution**: Ensure you have CUDA 12.4 installed and `uv sync` installed the correct PyTorch version.
```bash
uv run python -c "import torch; print(torch.version.cuda)"
```

#### Unsloth Import Error
**Solution**: Unsloth requires specific GPU capabilities (Ampere+ recommended). If on older GPU, you might need to disable Unsloth in code or upgrade hardware.

#### GitHub Rate Limit
**Solution**: Add GitHub API token in `.env`.

### Performance Optimization

1. **Use BF16** (requires Ampere+ GPU):
```yaml
training:
  bf16: true
  fp16: false
```

2. **Enable Flash Attention 2**:
```yaml
training:
  use_flash_attention_2: true
```

3. **Use Unsloth**:
ScriptGuard uses Unsloth by default for up to 2x faster training and 60% less memory usage.

### Getting Help

- **GitHub Issues**: https://github.com/yourusername/ScriptGuard/issues
- **Documentation**: [docs/](docs/)
