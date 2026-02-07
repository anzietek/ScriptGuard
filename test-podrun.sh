#!/bin/bash
"""
ScriptGuard Quick Test for Podrun
Tests the setup without running full training
"""

set -e

echo "========================================================"
echo "  ScriptGuard Podrun Quick Test"
echo "========================================================"
echo ""

# 1. Check Python
echo "[1/8] Checking Python..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# 2. Check uv
echo "[2/8] Checking uv..."
uv --version || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 3. Check ZenML
echo "[3/8] Checking ZenML installation..."
uv run python -c "import zenml; print(f'ZenML version: {zenml.__version__}')" || { echo "ERROR: ZenML not installed"; exit 1; }

# 4. Check PyTorch
echo "[4/8] Checking PyTorch..."
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch not installed"; exit 1; }

# 5. Check Qdrant
echo "[5/8] Checking Qdrant connection..."
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
curl -f "$QDRANT_URL/health" > /dev/null 2>&1 || { echo "ERROR: Cannot connect to Qdrant at $QDRANT_URL"; exit 1; }
echo "Qdrant is accessible"

# 6. Check config files
echo "[6/8] Checking configuration files..."
[ -f "config.yaml" ] || { echo "ERROR: config.yaml not found"; exit 1; }
[ -f ".env" ] || { echo "WARNING: .env not found (optional)"; }

# 7. Check project structure
echo "[7/8] Checking project structure..."
[ -d "src/scriptguard" ] || { echo "ERROR: src/scriptguard directory not found"; exit 1; }
[ -f "src/main.py" ] || { echo "ERROR: src/main.py not found"; exit 1; }

# 8. Run Python environment check
echo "[8/8] Running comprehensive environment check..."
uv run python check_podrun_env.py || { echo "ERROR: Environment check failed"; exit 1; }

echo ""
echo "========================================================"
echo "  ✓ All checks passed!"
echo "  You can now run: uv run python src/main.py"
echo "========================================================"
