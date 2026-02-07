# 🚀 Podrun Deployment - Complete Setup Summary

## ✅ Created Files

### 1. Setup Scripts

#### **podrun-setup.sh** (Linux/macOS)
- Automated setup script for Podrun environment
- Installs uv, dependencies, ZenML
- Verifies services (Qdrant, PostgreSQL)
- Initializes ZenML and starts training
- Usage: `./podrun-setup.sh` or `./podrun-setup.sh --check`

#### **podrun-setup.ps1** (Windows/PowerShell)
- Windows equivalent of setup script
- Same functionality as bash version
- Usage: `.\podrun-setup.ps1` or `.\podrun-setup.ps1 -Check`

### 2. Test Scripts

#### **test-podrun.sh** (Linux/macOS)
- Quick environment verification
- Checks Python, uv, ZenML, PyTorch, Qdrant
- Validates configuration files
- Usage: `./test-podrun.sh`

#### **test-podrun.ps1** (Windows/PowerShell)
- Windows quick test script
- Usage: `.\test-podrun.ps1`

### 3. Check Scripts

#### **check_podrun_env.py**
- Comprehensive Python environment checker
- Validates:
  - Python version (3.12+)
  - ZenML installation
  - PyTorch and CUDA
  - Qdrant connection
  - Configuration files
  - Dependencies
  - Environment variables
  - Disk space
- Usage: `python check_podrun_env.py` or `uv run python check_podrun_env.py`

#### **check_zenml.py**
- ZenML-specific validation script
- Checks installation, initialization, status
- Tests server connection
- Provides setup instructions
- Usage: `python check_zenml.py`

### 4. Configuration Files

#### **requirements.txt**
- Alternative dependency installation method
- Lists all required packages including ZenML
- Usage: `pip install -r requirements.txt`

#### **.podrunignore**
- Specifies files to exclude when uploading to Podrun
- Excludes logs, models, cache, virtual environments

### 5. Documentation

#### **docs/PODRUN_SETUP.md**
- Complete Podrun deployment guide
- Covers:
  - Prerequisites
  - Quick start
  - Manual setup steps
  - Configuration details
  - GPU configuration
  - Resource requirements
  - Monitoring
  - Troubleshooting
  - Advanced usage

#### **docs/PODRUN_QUICKSTART.md**
- Quick reference guide
- One-line setup commands
- Prerequisites checklist
- Essential configuration
- Common issues
- Pro tips
- Quick commands reference

#### **PODRUN_SCRIPTS.md**
- Overview of all Podrun scripts
- Workflow descriptions
- Usage examples
- Configuration details
- Troubleshooting guide

## 🎯 Quick Start Commands

### Linux/macOS
```bash
# Full setup
./podrun-setup.sh

# Check only
./podrun-setup.sh --check

# Quick test
./test-podrun.sh

# Environment check
python check_podrun_env.py

# ZenML check
python check_zenml.py
```

### Windows (PowerShell)
```powershell
# Full setup
.\podrun-setup.ps1

# Check only
.\podrun-setup.ps1 -Check

# Quick test
.\test-podrun.ps1

# Environment check
python check_podrun_env.py

# ZenML check
python check_zenml.py
```

## 📋 Prerequisites

- ✅ Python 3.12 or 3.13
- ✅ CUDA GPU (optional, recommended)
- ✅ Qdrant running and accessible
- ✅ 50+ GB free disk space
- ✅ HuggingFace token
- ✅ WandB API key (optional)

## 🔧 Required Environment Variables

Create `.env` file:
```bash
# Required
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token

# Optional
WANDB_API_KEY=your_wandb_key
ZENML_SERVER_URL=http://zenml-server:8237
DATABASE_URL=postgresql://user:pass@host:5432/db
```

## 🚀 Training Pipeline

After setup:
```bash
# Start training
uv run python src/main.py

# Check ZenML status
uv run zenml status

# View pipelines
uv run zenml pipeline list

# View runs
uv run zenml pipeline runs list
```

## 📊 Monitoring

### ZenML Dashboard
```bash
# Start local server
uv run zenml up

# Access at: http://localhost:8237
```

### WandB Dashboard
- Set `WANDB_API_KEY` in `.env`
- Access at: https://wandb.ai/

### Logs
```bash
# View logs
tail -f logs/scriptguard_$(date +%Y-%m-%d).log

# View errors
tail -f logs/errors_$(date +%Y-%m-%d).log
```

## 🔍 Verification Steps

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
   irm https://astral.sh/uv/install.ps1 | iex       # Windows
   ```

2. **Run setup script**:
   ```bash
   ./podrun-setup.sh          # Linux/macOS
   .\podrun-setup.ps1         # Windows
   ```

3. **Verify environment**:
   ```bash
   python check_podrun_env.py
   python check_zenml.py
   ```

4. **Start training**:
   ```bash
   uv run python src/main.py
   ```

## 🛠️ Key Features

### Automated Setup
- ✅ Detects and installs missing dependencies
- ✅ Configures ZenML automatically
- ✅ Verifies service connectivity
- ✅ Creates necessary directories
- ✅ Installs PyTorch with CUDA support

### Comprehensive Validation
- ✅ Python version check
- ✅ ZenML installation and initialization
- ✅ PyTorch and CUDA availability
- ✅ Qdrant connectivity
- ✅ Configuration file validation
- ✅ Disk space verification

### Cross-Platform Support
- ✅ Linux scripts (.sh)
- ✅ macOS scripts (.sh)
- ✅ Windows scripts (.ps1)
- ✅ Python validation scripts (all platforms)

## 📚 Documentation Structure

```
docs/
├── PODRUN_SETUP.md          # Complete setup guide
├── PODRUN_QUICKSTART.md     # Quick reference
└── ... (other docs)

Root/
├── PODRUN_SCRIPTS.md        # Scripts overview
├── podrun-setup.sh          # Linux/macOS setup
├── podrun-setup.ps1         # Windows setup
├── test-podrun.sh           # Linux/macOS test
├── test-podrun.ps1          # Windows test
├── check_podrun_env.py      # Environment checker
├── check_zenml.py           # ZenML checker
├── requirements.txt         # Dependencies list
└── .podrunignore            # Upload exclusions
```

## 🎓 Training Pipeline with ZenML

The setup configures ZenML to orchestrate the complete ML pipeline:

1. **Data Ingestion** - Multi-source collection
2. **Validation** - Syntax and quality checks
3. **Augmentation** - Code pattern generation
4. **Feature Extraction** - Malware indicators
5. **RAG Integration** - CVE pattern matching
6. **Vectorization** - Code embeddings to Qdrant
7. **Training** - QLoRA fine-tuning
8. **Evaluation** - Few-shot RAG evaluation

All managed by ZenML with tracking in WandB.

## 🐛 Common Issues & Solutions

### ZenML Not Found
```bash
uv pip install 'zenml[server]'
uv run zenml init
```

### CUDA Out of Memory
Edit `config.yaml`:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
```

### Qdrant Connection Failed
```bash
curl http://localhost:6333/health
echo $QDRANT_URL
```

### Permission Denied (Linux/macOS)
```bash
chmod +x podrun-setup.sh test-podrun.sh
```

## 🆘 Support

1. Run checks: `python check_podrun_env.py`
2. Check ZenML: `python check_zenml.py`
3. View logs: `tail -f logs/scriptguard_*.log`
4. See documentation: `docs/PODRUN_SETUP.md`

## 🔗 Related Documentation

- [PODRUN_SETUP.md](docs/PODRUN_SETUP.md) - Full setup guide
- [PODRUN_QUICKSTART.md](docs/PODRUN_QUICKSTART.md) - Quick reference
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training details
- [FEW_SHOT_RAG_GUIDE.md](docs/FEW_SHOT_RAG_GUIDE.md) - RAG guide

## ✨ What's Included

### Scripts (8 files)
- 2 setup scripts (Linux/Windows)
- 2 test scripts (Linux/Windows)
- 2 Python check scripts
- 2 configuration files

### Documentation (3 files)
- Complete setup guide
- Quick start reference
- Scripts overview

### Total: 11 new files for Podrun deployment with ZenML

---

**Ready to deploy on Podrun!** 🚀

Just run `./podrun-setup.sh` (Linux/macOS) or `.\podrun-setup.ps1` (Windows) to get started.
