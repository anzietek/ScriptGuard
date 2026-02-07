#!/bin/bash
# Installation script for ScriptGuard with unsloth support

set -e

echo "Installing ScriptGuard dependencies with unsloth..."
echo

cd "$(dirname "$0")"

echo "Step 1: Syncing base dependencies..."
uv sync

echo
echo "Step 2: Installing PyTorch 2.5.1 with CUDA 12.4..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo
echo "Step 3: Installing Triton 3.1.0..."
uv pip install triton==3.1.0

echo
echo "Step 4: Installing unsloth..."
uv pip install "unsloth[cu124-torch251] @ git+https://github.com/unslothai/unsloth.git"

echo
echo "Step 5: Testing installation..."
.venv/bin/python test_dependencies.py || echo "WARNING: Some dependencies might have issues"

echo
echo "========================================"
echo "Installation complete!"
echo "========================================"
