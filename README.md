# ScriptGuard v2.1: Production-Ready Malware Detection for Scripts

ScriptGuard is an advanced AI-powered system designed to detect malicious and dangerous scripts using state-of-the-art LLM techniques, ZenML pipelines, RAG architecture, and comprehensive data sources.

## 🎯 Key Features

- **Multi-Source Data Collection**: GitHub, MalwareBazaar, Hugging Face, CVE Feeds
- **Advanced Preprocessing**: Syntax validation, quality filtering, feature extraction
- **Intelligent Augmentation**: Code obfuscation, polymorphic variant generation
- **Few-Shot RAG**: Code similarity search for context-aware classification (NEW - EXPERIMENTAL)
- **Database Management**: PostgreSQL-based dataset versioning and deduplication
- **Production-Ready**: FastAPI inference, Docker deployment, RAG with Qdrant
- **Optimized Training**: Unsloth & Flash Attention 2 support for faster fine-tuning

## 🏗️ Architecture

### Data Pipeline
- **Sources**: GitHub API, MalwareBazaar, Hugging Face Datasets, NVD CVE Feeds
- **Validation**: AST syntax checking, encoding validation, quality metrics
- **Augmentation**: Base64/hex obfuscation, variable renaming, code mutation
- **Features**: Entropy analysis, API pattern detection, AST features

### ML Pipeline
- **Base Model:** `bigcode/starcoder2-3b` (Optimized for code analysis)
- **Fine-tuning:** Parameter-efficient fine-tuning using **QLoRA** (4-bit quantization) with **Unsloth** optimization
- **Few-Shot RAG:** Code similarity search using **microsoft/unixcoder-base** embeddings (NEW)
- **Orchestration:** **ZenML** manages the end-to-end ML lifecycle
- **RAG:** **Qdrant** stores embeddings of known CVEs and code samples
- **Tracking:** **Comet.ml** / **WandB** monitors experiments and metrics

### Deployment
- **Inference:** **FastAPI** provides high-performance REST API
- **Containerization:** **Docker Compose** orchestrates services
- **Database:** PostgreSQL for dataset management and versioning

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Database:** PostgreSQL 15 (with connection pooling)
- **Vector DB:** Qdrant (enhanced RAG)
- **Package Manager:** `uv`
- **Orchestration:** ZenML
- **Fine-tuning:** PEFT (LoRA/QLoRA), Unsloth, Flash Attention 2
- **Experiment Tracking:** WandB / Comet.ml
- **Serving:** FastAPI + Uvicorn
- **Containerization:** Docker (multistage builds)
- **Monitoring:** Prometheus + Grafana (optional)

## 📁 Project Structure

```text
├── docker/                      # Containerization configs
├── src/
│   ├── scriptguard/
│   │   ├── api/                 # FastAPI inference service
│   │   ├── data_sources/        # Multi-source data collectors
│   │   ├── database/            # Dataset management
│   │   ├── monitoring/          # Statistics & monitoring
│   │   ├── models/              # QLoRA fine-tuning logic
│   │   ├── pipelines/           # ZenML pipeline definitions
│   │   ├── rag/                 # Qdrant RAG store
│   │   └── steps/               # ZenML steps
│   └── main.py                  # Pipeline entry point
├── docs/                        # Comprehensive documentation
├── config.yaml                  # Central configuration
├── zenml_config.yaml            # ZenML step configuration
├── .env.example                 # Environment variables template
├── pyproject.toml               # Dependency management
├── podrun-setup.sh              # RunPod setup script
├── dev-setup.sh                 # Local development setup script
└── connect.sh                   # SSH tunnel script
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12**
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended for training)
- **CUDA**: 12.4
- **`uv`** installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** (optional for deployment)

### Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
```

#### Step 2: Install Dependencies

We use `uv` for fast and reliable dependency management.

```bash
# Install dependencies (including PyTorch with CUDA 12.4)
uv sync
```

#### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

#### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU) | NVIDIA RTX 3090/4090 (24GB VRAM) |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| CUDA | N/A | 12.4 |

### Configuration

Edit `config.yaml` to configure data sources, training parameters, and RAG settings. The default configuration is optimized for RunPod (RTX 3090/4090).

### Running on Podrun (RunPod)

For running training pipelines on Podrun with ZenML, use the automated setup scripts:

**Linux/macOS:**
```bash
chmod +x podrun-setup.sh
./podrun-setup.sh
```

**Windows (PowerShell):**
```powershell
.\podrun-setup.ps1
```

### Local Development Setup

For local development with Dockerized infrastructure (Postgres, Qdrant):

**Linux/macOS:**
```bash
chmod +x dev-setup.sh
./dev-setup.sh
```

**Windows:**
```cmd
dev-setup.bat
```

### Remote Connection

If you are deploying on a remote server and want to access services locally:

```bash
chmod +x connect.sh
./connect.sh
```

### Training

```bash
# Run advanced training pipeline
uv run python src/main.py
```

The pipeline will:
1. Collect data from configured sources
2. Validate and filter samples
3. Extract features and augment data
4. Train model with QLoRA (using Unsloth optimizations)
5. Evaluate performance

### Deployment

Start inference API:

```bash
# Using Docker (Recommended for Production)
docker-compose up -d api

# Or directly (Local Development)
uvicorn scriptguard.api.main:app --host 0.0.0.0 --port 8000
```

## 📖 Usage Examples

### API Request

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "import os; os.system(\"rm -rf /\")"
     }'
```

Response:
```json
{
  "label": "malicious",
  "confidence": 0.98,
  "risk_score": 9.5,
  "dangerous_patterns": ["os.system"],
  "explanation": "Uses os.system for dangerous command execution"
}
```

## 📚 Documentation

### Core Documentation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and component details
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Complete training guide
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - API usage and integration
- **[TUNING_GUIDE.md](docs/TUNING_GUIDE.md)** - Hyperparameter tuning
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guide
- **[LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md)** - Local development guide
- **[QDRANT_SETUP.md](docs/QDRANT_SETUP.md)** - Qdrant RAG setup
- **[PODRUN_README.md](PODRUN_README.md)** - Podrun specific documentation

## 🔧 Advanced Features

### Few-Shot RAG (Code Similarity Search)

ScriptGuard includes a **Code Similarity Search** system to potentially improve inference:

**How it works:**
1. **Vectorization**: Code samples from PostgreSQL are embedded using `microsoft/unixcoder-base`
2. **Storage**: Embeddings stored in Qdrant vector database
3. **Retrieval**: During inference, finds k=3 most similar code examples
4. **Context**: Similar examples added to prompt (Few-Shot Learning)

### Data Sources

ScriptGuard collects training data from multiple sources:
- **GitHub**
- **MalwareBazaar**
- **Hugging Face**
- **CVE Feeds**
- **Additional Datasets**: InQuest, dhuynh/malware-classification, malicious-urls

### Feature Extraction

Automatically extracts:
- AST-based features (function calls, imports, patterns)
- Shannon entropy
- API call patterns
- Suspicious string patterns

### Data Augmentation

Generates polymorphic variants using:
- Base64/hex encoding obfuscation
- Variable renaming
- String splitting
- Code mutation
- **Qdrant CVE Pattern Augmentation**

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🔐 Security Note

ScriptGuard is designed for **defensive security** purposes only. Do not use to create, modify, or improve malicious code.

## 📧 Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Full docs at [docs/](docs/)
