# ScriptGuard Architecture Documentation

Complete architectural overview of ScriptGuard v2.1 - a production-ready malware detection system for scripts.

## Table of Contents
- [System Overview](#system-overview)
- [Architecture Layers](#architecture-layers)
- [Component Details](#component-details)
- [ZenML Pipeline Architecture](#zenml-pipeline-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Deployment Architecture](#deployment-architecture)
- [Security Architecture](#security-architecture)

---

## System Overview

ScriptGuard is a microservices-based system that uses machine learning to detect malicious scripts. It combines:
- **Multiple data sources** for comprehensive training data
- **Advanced preprocessing** with validation and feature extraction
- **QLoRA fine-tuning** with **Unsloth** optimization for efficient model training
- **RAG (Retrieval-Augmented Generation)** for contextual analysis (CVE patterns & Code Similarity)
- **Production-ready inference API** with PostgreSQL and vector storage

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (CLI, Web UI, API Clients, CI/CD Integrations)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway / Load Balancer                   │
│                    (Nginx, Optional SSL/TLS)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ScriptGuard API Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   FastAPI    │  │  Inference   │  │     RAG      │          │
│  │   Endpoint   │→ │    Engine    │→ │   Context    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────┬───────────────────┬───────────────────┬──────────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌──────────┐      ┌──────────────┐    ┌──────────────┐
│PostgreSQL│      │    Model     │    │   Qdrant     │
│ Database │      │   Storage    │    │  Vector DB   │
│          │      │  (LoRA/Base) │    │ (CVE/Malware)│
└──────────┘      └──────────────┘    └──────────────┘
      ▲                                        ▲
      │                                        │
┌─────┴────────────────────────────────────────┴──────────────────┐
│               Training Pipeline (ZenML Orchestration)            │
│  Data Ingestion → Validation → Augmentation → Training          │
└──────────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌────────────────────────┴─────────────────────────────────────────┐
│                    Data Sources Layer                            │
│  GitHub | MalwareBazaar | Hugging Face | CVE Feeds | Manual     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### 1. **Data Sources Layer**
Responsible for collecting training data from multiple sources.

**Components:**
- `GitHubDataSource` - Searches GitHub for malicious/benign code
- `MalwareBazaarDataSource` - Fetches fresh malware samples
- `HuggingFaceDataSource` - Loads large-scale benign datasets
- `CVEFeedSource` - Retrieves vulnerability patterns from NVD

**Key Features:**
- Rate limiting and retry logic
- Deduplication at source level
- Metadata preservation
- Automatic error handling

---

### 2. **Data Processing Layer**
Handles validation, transformation, and augmentation.

**Components:**

#### **Validation Module** (`data_validation.py`)
- **Purpose:** Ensures data quality before training
- **Functions:**
  - AST syntax validation
  - Encoding verification (UTF-8)
  - Length filtering (min/max)
  - Comment ratio checking
  - Minimum code content verification

#### **Augmentation Module** (`advanced_augmentation.py`)
- **Purpose:** Generates synthetic variants to improve model robustness
- **Techniques:**
  - Base64/Hex encoding obfuscation
  - ROT13 cipher
  - Variable renaming
  - String splitting
  - Code reordering
  - Polymorphic variant generation
  - **Qdrant CVE Pattern Augmentation**

#### **Feature Extraction Module** (`feature_extraction.py`)
- **Purpose:** Extracts meaningful features for analysis
- **Features:**
  - AST-based features (function calls, imports, node counts)
  - Shannon entropy calculation
  - API pattern detection (network, file, process, crypto)
  - Suspicious string identification
  - Dangerous pattern recognition

---

### 3. **Storage Layer**
Manages persistent data storage.

#### **PostgreSQL Database**
- **Purpose:** Primary data store for code samples and metadata
- **Connection:** Uses connection pooling (1-10 connections)
- **Features:**
  - JSONB storage for flexible metadata
  - Materialized views for fast statistics
  - Dataset versioning support

#### **Qdrant Vector Database**
- **Purpose:** RAG knowledge base for CVEs and malware patterns
- **Collections:**
  - `malware_knowledge` - CVE data, exploit patterns, signatures
  - `code_samples` - Embeddings of training data for Few-Shot RAG
- **Features:**
  - HNSW indexing for fast similarity search
  - Payload indexes (cve_id, severity, type)
  - Score-based filtering
  - Automatic embedding generation

---

### 4. **Machine Learning Layer**

#### **Model Training**
- **Base Model:** `bigcode/starcoder2-3b` (or configurable)
- **Fine-tuning Method:** QLoRA (4-bit quantization) with **Unsloth**
- **Architecture:**
  ```
  Base Model (Frozen)
    ↓
  LoRA Adapters (Trainable)
    ↓
  Classification Head
    ↓
  Binary Output (malicious/benign)
  ```

**QLoRA Configuration:**
- Rank (r): 16 (low-rank matrices)
- Alpha: 32 (scaling factor)
- Dropout: 0.05 (regularization)
- Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

**Training Hyperparameters:**
- Batch size: 4
- Gradient accumulation: 4 (effective batch: 16)
- Learning rate: 2e-4
- Epochs: 3
- Optimizer: adamw_8bit
- **Flash Attention 2**: Enabled

#### **Inference Engine**
- **Purpose:** Real-time script analysis
- **Components:**
  1. **Preprocessor** - Tokenizes input code
  2. **Model** - Runs inference with LoRA adapters
  3. **RAG Context** - Retrieves relevant CVE patterns & similar code
  4. **Postprocessor** - Generates risk scores and explanations

**Inference Flow:**
```
Input Code
   ↓
Tokenization
   ↓
RAG Context Retrieval (Qdrant)
   ↓
Model Inference (Base + LoRA)
   ↓
Classification + Confidence
   ↓
Risk Score Calculation
   ↓
Output (JSON)
```

---

### 5. **API Layer**

#### **FastAPI Service**
- **Endpoints:**
  - `GET /health` - Health check
  - `POST /analyze` - Analyze single script
  - `POST /analyze/batch` - Batch analysis
  - `GET /model/info` - Model metadata
  - `GET /metrics` - Prometheus metrics

**Request Flow:**
```
Client Request
   ↓
FastAPI Validation (Pydantic)
   ↓
Rate Limiting Check
   ↓
Inference Engine
   ↓
Result Caching (Optional)
   ↓
JSON Response
```

**Response Schema:**
```json
{
  "label": "malicious",
  "confidence": 0.98,
  "risk_score": 9.5,
  "dangerous_patterns": ["os.system", "socket"],
  "features": {
    "entropy": 4.2,
    "suspicious_imports": ["os", "socket"],
    "api_patterns": ["network_and_process"]
  },
  "explanation": "Code uses os.system with network operations"
}
```

---

## ZenML Pipeline Architecture

ScriptGuard uses **ZenML** for ML pipeline orchestration. The pipeline is modular, reproducible, and production-ready.

### Pipeline Overview

```
┌───────────────────────────────────────────────────────────────┐
│                  advanced_training_pipeline                    │
└───────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Step 1:    │    │   Step 2:    │    │   Step 3:    │
│   Data       │───▶│  Validation  │───▶│   Quality    │
│  Ingestion   │    │              │    │   Filter     │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Step 4:    │    │   Step 5:    │    │   Step 6:    │
│   Feature    │───▶│  Feature     │───▶│   Data       │
│  Extraction  │    │  Analysis    │    │ Augmentation │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Step 7:    │    │   Step 8:    │    │   Step 9:    │
│   Dataset    │───▶│   Preprocess │───▶│    Model     │
│   Balance    │    │              │    │   Training   │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌──────────────┐
│   Step 10:   │
│    Model     │
│  Evaluation  │
└──────────────┘
```

### Pipeline Steps Detailed

#### **Step 1: Advanced Data Ingestion** (`advanced_ingestion.py`)

**Purpose:** Collect data from all configured sources

**Responsibilities:**
- Initialize data source connectors (GitHub, MalwareBazaar, HF, CVE)
- Fetch malicious and benign samples
- Merge data from multiple sources
- Deduplicate based on SHA256 hashes
- Store in PostgreSQL database
- Generate ingestion statistics

**Input:** Configuration dictionary
**Output:** List of unique samples

---

#### **Step 2: Sample Validation** (`data_validation.py`)

**Purpose:** Validate code syntax and quality

**Responsibilities:**
- AST syntax parsing
- Encoding validation (UTF-8)
- Length checks (50-50000 chars)
- Label verification (malicious/benign)
- Comment ratio analysis

**Input:** Raw samples
**Output:** Validated samples

---

#### **Step 3: Quality Filtering** (`data_validation.py`)

**Purpose:** Filter low-quality samples

**Responsibilities:**
- Count code vs comment lines
- Check minimum code lines (default: 5)
- Verify comment ratio (default: < 50%)
- Remove mostly-comment files

**Input:** Validated samples
**Output:** High-quality samples

---

#### **Step 4: Feature Extraction** (`feature_extraction.py`)

**Purpose:** Extract features for analysis

**Responsibilities:**
- **AST Features:**
  - Function calls (eval, exec, os.system, etc.)
  - Import statements
  - Dangerous patterns
  - Node type counts
  - Complexity score

- **Entropy Analysis:**
  - Shannon entropy calculation
  - Randomness detection

- **API Patterns:**
  - Network APIs (socket, requests, urllib)
  - File APIs (open, read, write)
  - Process APIs (subprocess, os.system, exec)
  - Crypto APIs (base64, hashlib)

- **String Features:**
  - URL detection
  - IP address detection
  - Base64-like strings
  - Hex strings
  - Suspicious keywords

**Input:** Quality-filtered samples
**Output:** Samples with extracted features

---

#### **Step 5: Feature Analysis** (`feature_extraction.py`)

**Purpose:** Analyze feature importance

**Responsibilities:**
- Calculate average entropy (malicious vs benign)
- Count dangerous patterns per class
- Identify suspicious API combinations
- Generate feature importance report

**Input:** Featured samples
**Output:** Feature analysis report

---

#### **Step 6: Data Augmentation** (`advanced_augmentation.py`)

**Purpose:** Generate polymorphic variants

**Responsibilities:**
- Generate N variants per malicious sample
- Apply obfuscation techniques:
  - Base64 encoding: `exec(base64.b64decode(...))`
  - Hex encoding: `exec(bytes.fromhex(...))`
  - ROT13 cipher
  - Variable renaming
  - String splitting
  - Code reordering
- **Qdrant Augmentation**: Inject known CVE patterns

**Input:** Featured samples
**Output:** Original + augmented samples

---

#### **Step 7: Dataset Balancing** (`advanced_augmentation.py`)

**Purpose:** Balance malicious/benign ratio

**Responsibilities:**
- Calculate current balance ratio
- Apply balancing strategy:
  - **Undersample:** Reduce majority class
  - **Oversample:** Duplicate minority class
- Target ratio: 1.0 (equal classes)

**Input:** Augmented samples
**Output:** Balanced dataset

---

#### **Step 8: Preprocessing** (`data_preprocessing.py`)

**Purpose:** Prepare data for training

**Responsibilities:**
- Tokenize code using model tokenizer
- Create attention masks
- Generate labels (0=benign, 1=malicious)
- Create train/validation splits
- Convert to HuggingFace Dataset format

**Input:** Balanced samples
**Output:** Tokenized Dataset

---

#### **Step 9: Model Training** (`model_training.py`)

**Purpose:** Fine-tune model with QLoRA

**Responsibilities:**
- Load base model (starcoder2-3b)
- Configure QLoRA:
  ```python
  LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  )
  ```
- Train with:
  - Batch size: 4
  - Gradient accumulation: 4
  - Learning rate: 2e-4
  - Epochs: 3
  - BF16 precision
  - **Unsloth Optimization**
  - **Flash Attention 2**
- Save LoRA adapters
- Log to WandB / Comet.ml

**Input:** Tokenized dataset
**Output:** Trained model path

---

#### **Step 10: Model Evaluation** (`model_evaluation.py`)

**Purpose:** Evaluate model performance

**Responsibilities:**
- Load validation dataset
- Run inference on test set
- Calculate metrics:
  - Accuracy
  - Precision (malicious class)
  - Recall (malicious class)
  - F1 Score
  - Confusion matrix
  - ROC-AUC
- Generate evaluation report
- Save results

**Input:** Trained model path
**Output:** Evaluation metrics

---

## Component Responsibilities

### Data Sources Components

| Component | Responsibility | Output |
|-----------|---------------|--------|
| **GitHubDataSource** | Search GitHub for code samples | Malicious/benign code with metadata |
| **MalwareBazaarDataSource** | Fetch fresh malware samples | Recent malware scripts |
| **HuggingFaceDataSource** | Load large-scale benign code | Clean code from popular repos |
| **CVEFeedSource** | Retrieve CVE patterns | Exploit patterns and CVE data |

### Database Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **DatabasePool** | Manage PostgreSQL connections | psycopg2 connection pool |
| **DatasetManager** | CRUD operations on samples | PostgreSQL with JSONB |
| **Deduplication** | Remove duplicate samples | SHA256 hashing |

### Monitoring Components

| Component | Responsibility | Metrics |
|-----------|---------------|---------|
| **DatasetStatistics** | Analyze dataset quality | Balance ratio, distribution, length stats |
| **check_balance()** | Verify class balance | Boolean (balanced or not) |
| **analyze_sources()** | Per-source statistics | Source-level metrics |

### RAG Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **QdrantStore** | Vector similarity search | Qdrant with HNSW |
| **bootstrap_cve_data()** | Initialize CVE knowledge | Pre-loaded patterns |
| **CodeSimilarityStore** | Few-Shot RAG | Embeddings of code samples |
| **SentenceTransformer** | Generate embeddings | all-MiniLM-L6-v2 / unixcoder-base |

---

## Data Flow

### Training Data Flow

```
External Sources (GitHub, MalwareBazaar, etc.)
   ↓
[Data Ingestion] - API calls, rate limiting
   ↓
Raw Samples (JSON with metadata)
   ↓
[Validation] - AST parsing, encoding checks
   ↓
Valid Samples
   ↓
[Quality Filter] - Comment ratio, min lines
   ↓
Quality Samples
   ↓
[Feature Extraction] - AST, entropy, patterns
   ↓
Featured Samples
   ↓
[Augmentation] - Obfuscation, variants, Qdrant CVEs
   ↓
Augmented Samples
   ↓
[Balancing] - Undersample/oversample
   ↓
Balanced Dataset
   ↓
[Preprocessing] - Tokenization
   ↓
Training-ready Dataset
   ↓
[Model Training] - QLoRA fine-tuning (Unsloth)
   ↓
Trained Model (LoRA adapters)
   ↓
[Storage] - Save to disk
```

### Inference Data Flow

```
Client Request (code string)
   ↓
[API Gateway] - FastAPI validation
   ↓
[Tokenization] - Convert to model input
   ↓
[RAG Context Retrieval] - Query Qdrant for similar patterns & code
   ↓
[Model Inference] - Base model + LoRA adapters
   ↓
[Classification] - Malicious/Benign + confidence
   ↓
[Feature Analysis] - Extract patterns, calculate entropy
   ↓
[Risk Scoring] - Combine factors into risk score
   ↓
[Response Generation] - JSON with explanation
   ↓
Client Response
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Database** | PostgreSQL 15 | Primary data storage with JSONB |
| **Vector DB** | Qdrant | RAG knowledge base |
| **ML Framework** | PyTorch 2.6+ (CUDA 12.4) | Model training and inference |
| **Transformers** | HuggingFace Transformers | Model loading and fine-tuning |
| **Fine-tuning** | PEFT (QLoRA) + Unsloth | Parameter-efficient training |
| **API** | FastAPI + Uvicorn | HTTP inference service |
| **Orchestration** | ZenML | ML pipeline management |
| **Containerization** | Docker (multistage) | Deployment packaging |
| **Monitoring** | Prometheus + Grafana | Metrics and dashboards |

### Python Libraries

```python
# Core ML
torch>=2.6.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
unsloth[cu124-torch260]

# Data
datasets>=2.19.0
pandas
numpy

# Database
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.0

# Vector DB
qdrant-client>=1.9.0
sentence-transformers>=3.0.0

# API
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0

# Utils
pyyaml>=6.0
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.66.0
```

---

## Deployment Architecture

### Docker Services

```yaml
services:
  postgres:        # Primary database
  qdrant:          # Vector store
  training:        # One-time training job
  api:             # Inference service (always-on)
  nginx:           # Reverse proxy (optional)
  prometheus:      # Metrics collection (optional)
  grafana:         # Dashboards (optional)
```

### Container Images

**Multistage Dockerfile:**
```
Stage 1: base-builder     (build tools, CUDA)
   ↓
Stage 2: dependencies     (compiled packages)
   ↓
Stage 3: training         (3.0 GB - training image)
   ↓
Stage 4: inference        (2.5 GB - API image)
   ↓
Stage 5: development      (full dev environment)
```

### Network Architecture

```
Internet
   ↓
[Nginx:80/443] SSL termination, load balancing
   ↓
[API:8000] FastAPI service (multiple instances)
   ↓
┌──────────────┬────────────────┐
│              │                │
[PostgreSQL:5432] [Qdrant:6333] [Models]
```

---

## Security Architecture

### Authentication & Authorization
- **API Keys** (optional) - Bearer token authentication
- **Rate Limiting** - Prevent abuse
- **Input Validation** - Pydantic models

### Data Security
- **Encryption at Rest** - PostgreSQL encryption
- **TLS/SSL** - HTTPS for API communication
- **Secrets Management** - Environment variables, Docker secrets

### Network Security
- **Bridge Network** - Isolated container communication
- **Firewall Rules** - Restrict external access
- **No Root** - Containers run as non-root user

### Code Security
- **Sandboxing** - Analysis in isolated environment
- **No Code Execution** - Static analysis only (no eval/exec)
- **Input Sanitization** - Prevent injection attacks

---

## Performance Considerations

### Optimization Strategies

| Component | Strategy | Impact |
|-----------|----------|--------|
| **Database** | Connection pooling | 10x write speed |
| **Database** | JSONB indexes | Fast metadata queries |
| **Database** | Materialized views | Instant statistics |
| **Qdrant** | HNSW indexing | <50ms search |
| **Qdrant** | Payload indexes | Fast filtering |
| **API** | Result caching | Avoid duplicate inference |
| **API** | Batch inference | 5-10x throughput |
| **Model** | 4-bit quantization | 4x memory reduction |
| **Model** | BF16 training | Faster on modern GPUs |
| **Training** | Unsloth | 2x faster training |
| **Training** | Flash Attention 2 | Faster attention mechanism |

### Scalability

**Horizontal Scaling:**
- API service: `docker-compose up --scale api=N`
- Load balancer: Nginx upstream
- Stateless design

**Vertical Scaling:**
- Increase GPU memory
- More CPU cores for data processing
- Larger PostgreSQL instance

---

## Monitoring & Observability

### Metrics

**System Metrics:**
- CPU/GPU utilization
- Memory usage
- Disk I/O
- Network traffic

**Application Metrics:**
- Request latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- Model inference time

**Business Metrics:**
- Detection rate (TP, FP, TN, FN)
- Confidence distribution
- Source distribution

### Logging

**Log Levels:**
- ERROR: Critical failures
- WARNING: Potential issues
- INFO: Normal operations
- DEBUG: Detailed traces

**Log Destinations:**
- stdout/stderr (Docker logs)
- File: `/app/logs/scriptguard.log`
- Centralized: Elasticsearch (optional)

---

## Summary

ScriptGuard v2.1 is a **modular, scalable, production-ready** malware detection system built with:

✅ **Microservices architecture** - Independent, scalable components
✅ **ML pipeline orchestration** - ZenML for reproducibility
✅ **Multi-source data collection** - Comprehensive training data
✅ **Advanced preprocessing** - Validation, augmentation, features
✅ **Efficient training** - QLoRA + Unsloth for fast, low-memory fine-tuning
✅ **RAG-enhanced inference** - CVE context & Code Similarity for better detection
✅ **Production-grade storage** - PostgreSQL + Qdrant
✅ **Container-native** - Docker multistage builds
✅ **Monitoring-ready** - Prometheus + Grafana integration

**Next Steps:**
- Deploy to production: [DEPLOYMENT.md](./DEPLOYMENT.md)
- Train your model: [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)
- Use the API: [USAGE_GUIDE.md](./USAGE_GUIDE.md)
- Optimize performance: [TUNING_GUIDE.md](./TUNING_GUIDE.md)
