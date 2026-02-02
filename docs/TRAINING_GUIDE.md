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
- **GPU**: NVIDIA GPU with at least 16GB VRAM (recommended: RTX 3090, RTX 4090, A100)
- **RAM**: Minimum 32GB system RAM
- **Storage**: 100GB+ free space for datasets and models
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS

### Software Requirements
- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)
- Docker (optional, for containerized deployment)

### API Keys (Optional but Recommended)
- **GitHub Personal Access Token**: Increases API rate limits
- **NVD API Key**: Faster CVE data fetching
- **Hugging Face Token**: For private dataset access

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GITHUB_API_TOKEN=ghp_your_token_here
NVD_API_KEY=your_nvd_key_here
HUGGINGFACE_TOKEN=hf_your_token_here
```

## Data Collection

### Understanding Data Sources

ScriptGuard collects data from multiple sources:

1. **GitHub** - Malicious and benign code samples
2. **MalwareBazaar** - Fresh malware samples
3. **Hugging Face Datasets** - Large-scale benign code
4. **CVE Feeds** - Exploit patterns from NVD

### Configure Data Sources

Edit `config.yaml`:

```yaml
data_sources:
  github:
    enabled: true
    fetch_malicious: true
    fetch_benign: true
    max_samples_per_keyword: 20
    max_files_per_repo: 50

  malwarebazaar:
    enabled: true
    max_samples: 100

  huggingface:
    enabled: true
    max_samples: 10000

  cve_feeds:
    enabled: true
    days_back: 30
```

### Manual Data Collection

You can manually collect data before training:

```python
from scriptguard.database import DatasetManager
from scriptguard.data_sources import GitHubDataSource, MalwareBazaarDataSource

# Initialize database (automatically uses config.yaml)
db = DatasetManager()

# Fetch from GitHub
github = GitHubDataSource(api_token="your_token")
malicious = github.fetch_malicious_samples(max_per_keyword=20)
benign = github.fetch_benign_samples(max_files_per_repo=50)

# Add to database
db.add_samples_batch(malicious)
db.add_samples_batch(benign)

# Check statistics
stats = db.get_dataset_stats()
print(stats)
```

## Configuration

### Training Configuration

Edit `config.yaml` training section:

```yaml
training:
  model_id: "bigcode/starcoder2-3b"  # Base model
  output_dir: "./models/scriptguard-model"

  # QLoRA settings
  use_qlora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

  # Hyperparameters
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 0.0002
  max_seq_length: 2048

  # Optimization
  bf16: true
  optim: "paged_adamw_8bit"
```

### Validation Settings

```yaml
validation:
  validate_syntax: true
  skip_syntax_errors: true
  min_length: 50
  max_length: 50000
  min_code_lines: 5
```

### Augmentation Settings

```yaml
augmentation:
  enabled: true
  variants_per_sample: 2
  balance_dataset: true
  target_balance_ratio: 1.0
  balance_method: "undersample"
```

## Training Pipeline

### Quick Start

Run the training pipeline with default configuration:

```bash
python src/main.py
```

### Step-by-Step Pipeline

The training pipeline executes the following steps:

#### 1. Data Ingestion
```python
from scriptguard.steps.advanced_ingestion import advanced_data_ingestion
# Fetches data from all configured sources
# config is loaded from config.yaml
raw_data = advanced_data_ingestion(config=config)
```

#### 2. Data Validation
```python
from scriptguard.steps.data_validation import validate_samples
# Validates syntax, length, encoding
validated_data = validate_samples(data=raw_data)
```

#### 3. Quality Filtering
```python
from scriptguard.steps.data_validation import filter_by_quality
# Filters by code quality metrics
quality_data = filter_by_quality(data=validated_data)
```

#### 4. Feature Extraction
```python
from scriptguard.steps.feature_extraction import extract_features
# Extracts AST, entropy, API patterns
featured_data = extract_features(data=quality_data)
```

#### 5. Data Augmentation
```python
from scriptguard.steps.advanced_augmentation import augment_malicious_samples
# Generates polymorphic variants
augmented_data = augment_malicious_samples(data=featured_data)
```

#### 6. Dataset Balancing
```python
from scriptguard.steps.advanced_augmentation import balance_dataset
# Balances malicious/benign ratio
balanced_data = balance_dataset(data=augmented_data)
```

#### 7. Preprocessing
```python
from scriptguard.steps.data_preprocessing import preprocess_data
# Tokenizes and formats for training
processed_dataset = preprocess_data(data=balanced_data)
```

#### 8. Model Training
```python
from scriptguard.steps.model_training import train_model
# Trains model with QLoRA
adapter_path = train_model(dataset=processed_dataset)
```

#### 9. Evaluation
```python
from scriptguard.steps.model_evaluation import evaluate_model
# Evaluates model performance
metrics = evaluate_model(adapter_path=adapter_path)
```

### Custom Pipeline Execution

```python
from scriptguard.pipelines.training_pipeline import advanced_training_pipeline
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Run pipeline
metrics = advanced_training_pipeline(
    config=config,
    model_id="bigcode/starcoder2-3b"
)

print(f"Training completed! Metrics: {metrics}")
```

## Monitoring Training

### ZenML Dashboard

Start ZenML server:
```bash
zenml up
```

Access dashboard at `http://localhost:8080`

### Comet.ml Integration

Configure Comet.ml in `.env`:
```env
COMET_API_KEY=your_comet_key
COMET_PROJECT_NAME=scriptguard
```

View experiments at [comet.ml](https://www.comet.ml)

### Training Logs

Check logs:
```bash
tail -f logs/scriptguard.log
```

### Dataset Statistics

View dataset statistics in logs:
```
====================================================
DATASET STATISTICS REPORT
====================================================
Total Samples: 15000
Label Distribution:
  malicious: 7500 (50.0%)
  benign: 7500 (50.0%)
Source Distribution:
  github: 5000 (33.3%)
  malwarebazaar: 2500 (16.7%)
  huggingface: 7500 (50.0%)
Balance Ratio: 1.00
Balance Quality: EXCELLENT - Well balanced
====================================================
```

## Model Export

### Export Trained Model

After training completes, the adapter is saved to the output directory specified in `config.yaml`. To use it, you can load it using `PeftModel`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")

# Load adapter
model = PeftModel.from_pretrained(base_model, "./model_checkpoints/final_adapter")
```

### Merge LoRA Adapter with Base Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")

# Load adapter
model = PeftModel.from_pretrained(base_model, "./model_checkpoints/final_adapter")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/scriptguard-merged")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
tokenizer.save_pretrained("./models/scriptguard-merged")
```

### Quantize for Deployment

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/scriptguard-merged",
    quantization_config=quantization_config
)

model.save_pretrained("./models/scriptguard-4bit")
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
**Solution**: Reduce batch size or use gradient accumulation
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
```

#### Slow Data Collection
**Solution**: Reduce sample limits
```yaml
data_sources:
  github:
    max_samples_per_keyword: 10  # Reduce from 20
  huggingface:
    max_samples: 5000  # Reduce from 10000
```

#### Invalid Syntax Errors
**Solution**: Enable syntax error skipping
```yaml
validation:
  skip_syntax_errors: true
```

#### Imbalanced Dataset
**Solution**: Adjust balancing method
```yaml
augmentation:
  balance_dataset: true
  target_balance_ratio: 1.0
  balance_method: "oversample"  # Try oversample instead
```

#### GitHub Rate Limit
**Solution**: Add GitHub API token or reduce requests
```env
GITHUB_API_TOKEN=your_token_here
```

### Performance Optimization

1. **Use BF16 instead of FP16** (better for modern GPUs):
```yaml
training:
  bf16: true
  fp16: false
```

2. **Enable gradient checkpointing** (saves memory):
```python
# During model initialization
model.gradient_checkpointing_enable()
```

3. **Use Flash Attention 2** (faster attention):
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoder2-3b",
    attn_implementation="flash_attention_2"
)
```

### Getting Help

- **GitHub Issues**: https://github.com/yourusername/ScriptGuard/issues
- **Documentation**: https://scriptguard.readthedocs.io
- **Community**: Discord or Slack channel

## Next Steps

After training:
1. Review [USAGE_GUIDE.md](./USAGE_GUIDE.md) for deployment
2. Review [TUNING_GUIDE.md](./TUNING_GUIDE.md) for optimization
3. Set up inference API for production use
