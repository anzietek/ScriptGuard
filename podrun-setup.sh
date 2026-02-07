#!/bin/bash
#
# ScriptGuard Podrun Setup Script
# This script prepares the environment for running training pipelines on Podrun with ZenML.
# It assumes Postgres and Qdrant are external services.
# It sets up a local ZenML server for orchestration/dashboard.
#
# Usage:
#   ./podrun-setup.sh         - Setup and ask to run training
#   ./podrun-setup.sh -y      - Setup and auto-start training (non-interactive)
#   ./podrun-setup.sh --check - Check environment only
#

set -e  # Exit on error

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

# Helper functions
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

# Load environment variables from .env safely
load_env() {
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed. Please install Python 3.10-3.12."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python ${PYTHON_VERSION} found"

    # Must match pyproject.toml: requires-python = ">=3.10,<3.13"
    if [[ ! $PYTHON_VERSION =~ ^3\.(10|11|12)\. ]]; then
        print_error "Python ${PYTHON_VERSION} is not compatible. Required: >=3.10,<3.13 (per pyproject.toml)"
        exit 1
    fi
}

# Check if uv is installed
check_uv() {
    print_info "Checking for uv package installer..."
    if ! command -v uv &> /dev/null; then
        if [ $CHECK_ONLY -eq 1 ]; then
            print_error "uv is not installed."
            exit 1
        fi
        print_warning "uv is not installed. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv installed successfully"
    else
        print_success "uv is already installed: $(uv --version)"
    fi
}

# Check CUDA availability
check_cuda() {
    print_info "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "CUDA not detected. Training will use CPU (slower)."
    fi
}

# Setup environment file
setup_env() {
    print_info "Setting up environment configuration..."

    if [ ! -f .env ]; then
        if [ $CHECK_ONLY -eq 1 ]; then
            print_error ".env file not found!"
            exit 1
        fi
        if [ -f .env.example ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_success ".env file created"
            print_warning "Action Required: Please edit .env file with your external Qdrant/Postgres credentials!"
        else
            print_error ".env.example not found!"
            exit 1
        fi
    else
        print_success ".env file already exists"
    fi
}

# Install dependencies with uv
install_dependencies() {
    print_info "Installing dependencies with uv..."

    # uv sync installs everything from pyproject.toml including
    # the correct PyTorch version (2.6.0+cu124) with platform-specific wheels
    uv sync

    print_success "Dependencies installed successfully"
}

# Verify critical dependencies
verify_dependencies() {
    print_info "Verifying critical dependencies..."

    # Verify ZenML
    if uv run zenml version &> /dev/null; then
        ZENML_VERSION=$(uv run zenml version)
        print_success "ZenML installed: ${ZENML_VERSION}"
    else
        print_error "ZenML installation failed!"
        exit 1
    fi

    # Verify unsloth (required by src/main.py)
    if uv run python -c "import unsloth; print(unsloth.__version__)" &> /dev/null; then
        UNSLOTH_VERSION=$(uv run python -c "import unsloth; print(unsloth.__version__)")
        print_success "Unsloth installed: ${UNSLOTH_VERSION}"
    else
        print_warning "Unsloth import failed. Training may not work with optimizations."
    fi

    # Verify PyTorch and CUDA
    TORCH_INFO=$(uv run python -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "not available")
    print_info "PyTorch: ${TORCH_INFO}"

    # Verify flash-attn on Linux (required by config.yaml: use_flash_attention_2: true)
    if [ "$(uname)" = "Linux" ]; then
        if uv run python -c "import flash_attn" &> /dev/null; then
            print_success "Flash Attention 2 available"
        else
            print_warning "Flash Attention 2 not available. Config has use_flash_attention_2: true - training may fall back to standard attention."
        fi
    fi
}

# Initialize and Start ZenML Server (Local on Pod)
init_zenml() {
    print_info "Initializing ZenML..."

    # Initialize repo if needed
    if [ -d ".zen" ]; then
        print_warning "ZenML already initialized in this directory"
    else
        uv run zenml init
        print_success "ZenML initialized"
    fi

    # Start ZenML Server locally (background)
    # Bind to 127.0.0.1 by default; use RunPod port forwarding for external access
    print_info "Starting local ZenML Server..."

    if pgrep -f "zenml" > /dev/null 2>&1; then
        print_warning "ZenML Server is already running."
    else
        nohup uv run zenml up --port 8237 > logs/zenml_server.log 2>&1 &

        # Give it a moment to start
        sleep 5
        if pgrep -f "zenml" > /dev/null 2>&1; then
            print_success "ZenML Server started on port 8237 (check logs/zenml_server.log)"
        else
            print_warning "ZenML Server may have failed to start. Check logs/zenml_server.log"
        fi
    fi
}

# Check external services (Qdrant & Postgres)
check_services() {
    print_info "Checking configuration for external services..."

    load_env

    # Check Qdrant using QDRANT_HOST/QDRANT_PORT (matching .env.example)
    QDRANT_HOST="${QDRANT_HOST:-localhost}"
    QDRANT_PORT="${QDRANT_PORT:-6333}"
    QDRANT_ENDPOINT="http://${QDRANT_HOST}:${QDRANT_PORT}"

    print_info "Checking Qdrant at ${QDRANT_ENDPOINT}..."
    if curl --max-time 5 -s "${QDRANT_ENDPOINT}/healthz" > /dev/null 2>&1; then
        print_success "Qdrant is reachable at ${QDRANT_ENDPOINT}"
    else
        print_warning "Could not reach Qdrant at ${QDRANT_ENDPOINT}. Check that Qdrant is running."
    fi

    # Check PostgreSQL
    if [ -n "$POSTGRES_HOST" ]; then
        print_success "PostgreSQL configured: ${POSTGRES_HOST}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-scriptguard}"
    else
        print_warning "POSTGRES_HOST is not set in .env"
    fi

    # Check WandB (config.yaml reports to wandb)
    if [ -n "$WANDB_API_KEY" ]; then
        print_success "WandB API key configured (project: ${WANDB_PROJECT:-scriptguard})"
    else
        print_warning "WANDB_API_KEY is not set in .env. config.yaml has report_to: [wandb] - training will fail without it."
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."

    mkdir -p data
    mkdir -p models
    mkdir -p logs
    mkdir -p model_checkpoints
    mkdir -p .cache

    print_success "Directories created"
}

# Verify configuration
verify_config() {
    print_info "Verifying configuration files..."

    if [ ! -f "config.yaml" ]; then
        print_error "config.yaml not found!"
        exit 1
    fi

    print_success "Configuration files verified"
}

# Display environment info
display_info() {
    load_env

    QDRANT_HOST="${QDRANT_HOST:-localhost}"
    QDRANT_PORT="${QDRANT_PORT:-6333}"

    echo ""
    echo "========================================================"
    echo "  ScriptGuard Podrun Environment Ready!"
    echo "========================================================"
    echo ""
    echo "Environment:"
    echo "  Python:      $(python3 --version)"
    echo "  uv:          $(uv --version)"
    echo "  ZenML:       $(uv run zenml version 2>/dev/null || echo 'N/A')"
    echo ""
    echo "Services:"
    echo "  ZenML Server: http://localhost:8237"
    echo "  Qdrant:       http://${QDRANT_HOST}:${QDRANT_PORT}"
    echo "  Postgres:     ${POSTGRES_HOST:-Not Set}:${POSTGRES_PORT:-5432}"
    echo "  WandB:        ${WANDB_PROJECT:-Not Set}"
    echo ""
    echo "Logs:"
    echo "  ZenML Logs:   logs/zenml_server.log"
    echo ""
    echo "Run training:   uv run python src/main.py"
    echo ""
    echo "========================================================"
    echo ""
}

# Run training pipeline
run_training() {
    print_info "Starting training pipeline..."

    load_env

    print_info "Running training with ZenML..."
    uv run python src/main.py

    print_success "Training pipeline completed!"
}

# Main execution
main() {
    echo "========================================================"
    echo "  ScriptGuard Podrun Setup"
    if [ $CHECK_ONLY -eq 1 ]; then
        echo "  MODE: Environment check only"
    else
        echo "  MODE: Full setup and training"
    fi
    echo "========================================================"
    echo ""

    # Phase 1: Checks (always run)
    check_python
    check_uv
    check_cuda
    setup_env
    create_directories
    verify_config

    # Phase 2: Installation (skip in check mode)
    if [ $CHECK_ONLY -eq 0 ]; then
        install_dependencies
        verify_dependencies
        check_services
        init_zenml
    else
        # In check mode, only verify services connectivity
        check_services
        print_success "Environment check complete!"
        exit 0
    fi

    display_info

    # Phase 3: Training
    if [ $AUTO_APPROVE -eq 1 ]; then
        print_info "Auto-approve flag detected (-y). Starting training..."
        run_training
    else
        # Interactive mode
        read -p "Start training pipeline now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_training
        else
            print_info "Training skipped. Run manually with: uv run python src/main.py"
        fi
    fi

    print_success "Setup script finished!"
}

# Run main function
main
