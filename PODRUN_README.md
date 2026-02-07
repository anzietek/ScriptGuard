# ⚡ Podrun Quick Reference Card

## 🚀 One Command Setup

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh && ./podrun-setup.sh

# Windows
irm https://astral.sh/uv/install.ps1 | iex; .\podrun-setup.ps1
```

## 📝 Essential Commands

```bash
# Setup
./podrun-setup.sh              # Full setup (Linux/macOS)
.\podrun-setup.ps1             # Full setup (Windows)

# Verify
./test-podrun.sh               # Quick test (Linux/macOS)
python check_podrun_env.py     # Full check

# Train
uv run python src/main.py      # Start training

# Monitor
uv run zenml status            # ZenML status
uv run zenml pipeline runs list # View runs
tail -f logs/scriptguard_*.log # View logs
```

## 🔑 Required in .env

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_key
HUGGINGFACE_TOKEN=your_token
WANDB_API_KEY=your_key  # optional
```

## 📊 Quick Config (config.yaml)

```yaml
training:
  model_id: "bigcode/starcoder2-3b"
  batch_size: 4  # Adjust for GPU
  num_epochs: 3

augmentation:
  use_qdrant_patterns: true  # Enable RAG

code_embedding:
  max_samples_to_vectorize: 1000
```

## 🐛 Common Fixes

```bash
# ZenML missing
uv pip install 'zenml[server]'

# CUDA OOM
# Edit config.yaml: batch_size: 1

# Qdrant down
curl http://localhost:6333/health

# Permissions
chmod +x podrun-setup.sh test-podrun.sh
```

## 📚 Full Docs

- [PODRUN_QUICKSTART.md](docs/PODRUN_QUICKSTART.md)
- [PODRUN_SETUP.md](docs/PODRUN_SETUP.md)
- [PODRUN_SCRIPTS.md](PODRUN_SCRIPTS.md)
- [PODRUN_SUMMARY.md](PODRUN_SUMMARY.md)

---
**Need help?** Run: `python check_podrun_env.py`
