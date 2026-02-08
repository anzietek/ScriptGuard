#!/bin/bash
#
# ScriptGuard RunPod Setup Script (Optimized)
#
# Features:
# - Auto-installs 'uv' and Python 3.12 (managed environment)
# - Sets up ZenML Server accessible via RunPod Proxy/TCP
# - Checks for Persistent Volume (/workspace) to prevent data loss
# - Auto-starts training pipeline if requested
#
# Usage:
#   ./setup.sh         - Setup and ask to run training
#   ./setup.sh -y      - Setup and auto-start training (non-interactive)
#   ./setup.sh --check - Check environment only
#

set -e  # Exit on error

# --- Global Configuration ---
# Ensure local bin paths are prioritized for uv and pip
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Arguments parsing
CHECK_ONLY=0
AUTO_APPROVE=0

for arg in "$@"; do
  case $arg in
    --check)
      CHECK_ONLY=1
      shift
      ;;
    -y|--yes)
      AUTO_APPROVE=1
      shift
      ;;
  esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# --- Core Checks ---

check_workspace() {
    print_info "Checking filesystem location..."
    CURRENT_DIR=$(pwd)

    # RunPod persistent storage is usually at /workspace
    if [[ "$CURRENT_DIR" != *"/workspace"* ]]; then
        print_warning "-----------------------------------------------------------"
        print_warning "CRITICAL: You are NOT running inside '/workspace'."
        print_warning "Any data created here (/root) WILL BE LOST after Pod restart."
        print_warning "-----------------------------------------------------------"

        if [ $AUTO_APPROVE -eq 0 ]; then
             read -p "Are you sure you want to continue in ephemeral storage? (y/n) " -n 1 -r
             echo
             if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                 print_error "Aborted by user. Please move to /workspace."
                 exit 1
             fi
        fi
    else
        print_success "Running safely inside persistent volume (/workspace)."
    fi
}

load_env() {
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
}

check_uv() {
    print_info "Checking for 'uv' package manager..."
    if ! command -v uv &> /dev/null; then
        if [ $CHECK_ONLY -eq 1 ]; then
            print_error "uv is not installed."
            exit 1
        fi
        print_warning "uv is not installed. Installing latest version..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv installed successfully"
    else
        print_success "uv is ready: $(uv --version)"
    fi
}

check_python_system() {
    # We only check if python3 exists. We rely on 'uv' to fetch the correct version (3.12) defined in pyproject.toml.
    print_info "Checking system Python..."
    if command -v python3 &> /dev/null; then
        SYS_VER=$(python3 --version)
        print_success "System Python found: $SYS_VER (Note: Project will use Python 3.12 via 'uv')"
    else
        print_warning "System Python not found. 'uv' will attempt to fetch a managed Python version."
    fi
}

check_cuda() {
    print_info "Checking GPU/CUDA status..."
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected! Training will be extremely slow on CPU."
    fi
}

setup_env() {
    print_info "Configuring environment variables..."

    if [ ! -f .env ]; then
        if [ $CHECK_ONLY -eq 1 ]; then
            print_error ".env file missing!"
            exit 1
        fi

        if [ -f .env.example ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_success ".env created."
            print_warning "ACTION REQUIRED: Please edit .env with your real API keys!"
        else
            print_error "Neither .env nor .env.example found!"
            exit 1
        fi
    else
        print_success ".env file exists."
    fi
}

# --- Installation & Verification ---

install_dependencies() {
    print_info "Syncing dependencies with uv (this may take a moment)..."

    # uv sync reads pyproject.toml, creates .venv, installs Python 3.12 and all deps
    uv sync

    if [ $? -eq 0 ]; then
        print_success "Environment synchronized successfully."
    else
        print_error "Failed to sync dependencies."
        exit 1
    fi
}

verify_dependencies() {
    print_info "Verifying critical ML components..."

    # Check via the venv python
    uv run python -c "
import torch
import unsloth
import sys

print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
print(f'Unsloth: {unsloth.__version__}')

try:
    import flash_attn
    print('Flash Attention 2: Installed')
except ImportError:
    print('Flash Attention 2: NOT FOUND (Check compilation)')
"
    print_success "Verification check passed."
}

# --- Services ---

init_zenml() {
    print_info "Initializing ZenML..."

    if [ ! -d ".zen" ]; then
        uv run zenml init
        print_success "ZenML repository initialized."
    fi

    print_info "Starting ZenML Server (Background)..."

    # Check if port 8237 is in use
    if lsof -Pi :8237 -sTCP:LISTEN -t >/dev/null ; then
        print_warning "ZenML Server appears to be running already on port 8237."
    else
        # Bind to 0.0.0.0 to allow external access via RunPod Proxy/TCP
        nohup uv run zenml up --host 0.0.0.0 --port 8237 > logs/zenml_server.log 2>&1 &

        # Wait for startup
        sleep 5
        if lsof -Pi :8237 -sTCP:LISTEN -t >/dev/null ; then
            print_success "ZenML Server started!"
            echo ""
            echo -e "${YELLOW}Accessing Dashboard:${NC}"
            echo "1. RunPod Console -> Connect -> TCP Port Mapping"
            echo "2. Map Internal Port 8237 to a Public Port."
            echo "3. Or use SSH Tunnel: ssh -L 8237:localhost:8237 root@<POD_IP> -p <SSH_PORT>"
            echo ""
        else
            print_error "ZenML failed to start. Check logs/zenml_server.log"
        fi
    fi
}

check_services() {
    print_info "Checking external service connectivity..."
    load_env

    # Qdrant
    QDRANT_HOST="${QDRANT_HOST:-localhost}"
    QDRANT_PORT="${QDRANT_PORT:-6333}"

    if curl --max-time 3 -s "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" > /dev/null; then
        print_success "Qdrant reachable."
    else
        print_warning "Cannot reach Qdrant at ${QDRANT_HOST}:${QDRANT_PORT}. Is it running?"
    fi
}

create_directories() {
    mkdir -p data models logs model_checkpoints .cache
}

# --- Main Execution ---

main() {
    echo "========================================================"
    echo "   ScriptGuard RunPod Environment Setup"
    echo "========================================================"

    # 1. Preliminaries
    check_workspace
    check_python_system
    check_uv
    check_cuda
    create_directories
    setup_env

    # 2. Installation (Skip if check-only)
    if [ $CHECK_ONLY -eq 0 ]; then
        install_dependencies
        verify_dependencies
        init_zenml
    fi

    # 3. Final Checks
    check_services

    echo ""
    echo "========================================================"
    echo "   Setup Complete!"
    echo "========================================================"

    if [ $CHECK_ONLY -eq 1 ]; then
        exit 0
    fi

    # 4. Run Training
    if [ $AUTO_APPROVE -eq 1 ]; then
        print_info "Auto-start enabled. Launching training pipeline..."
        uv run python src/main.py
    else
        echo ""
        read -p "Do you want to start the training pipeline now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Launching src/main.py..."
            uv run python src/main.py
        else
            print_info "Ready. Run manually with: uv run python src/main.py"
        fi
    fi
}

# Run Main
main
