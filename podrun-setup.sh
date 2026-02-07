#!/bin/bash
"""
ScriptGuard Podrun Setup Script
This script prepares the environment for running training pipelines on Podrun with ZenML
Usage:
  ./podrun-setup.sh         - Setup and run training
  ./podrun-setup.sh --check - Check environment only
"""

set -e  # Exit on error

# Check for --check argument
CHECK_ONLY=0
if [ "$1" == "--check" ]; then
    CHECK_ONLY=1
fi

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

# Check Python version
check_python() {
    print_info "Checking Python version..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed. Please install Python 3.12 or 3.13."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python ${PYTHON_VERSION} found"

    # Check if version is compatible (3.12 or 3.13)
    if [[ ! $PYTHON_VERSION =~ ^3\.(12|13) ]]; then
        print_warning "Python version ${PYTHON_VERSION} detected. Recommended: 3.12 or 3.13"
    fi
}

# Check if uv is installed
check_uv() {
    print_info "Checking for uv package installer..."
    if ! command -v uv &> /dev/null; then
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
        if [ -f .env.example ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_success ".env file created"
            print_warning "Please edit .env file with your credentials!"
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

    # Sync dependencies from pyproject.toml
    uv sync

    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        print_info "Installing PyTorch with CUDA 12.4 support..."
        uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
    else
        print_info "Installing PyTorch (CPU version)..."
        uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
    fi

    print_success "Dependencies installed successfully"
}

# Verify ZenML installation
verify_zenml() {
    print_info "Verifying ZenML installation..."

    if uv run zenml version &> /dev/null; then
        ZENML_VERSION=$(uv run zenml version)
        print_success "ZenML installed: ${ZENML_VERSION}"
    else
        print_error "ZenML installation failed!"
        exit 1
    fi
}

# Initialize ZenML
init_zenml() {
    print_info "Initializing ZenML..."

    # Check if ZenML is already initialized
    if [ -d ".zen" ]; then
        print_warning "ZenML already initialized in this directory"
    else
        uv run zenml init
        print_success "ZenML initialized"
    fi

    # Connect to ZenML server if configured
    if [ -n "$ZENML_SERVER_URL" ]; then
        print_info "Connecting to ZenML server at ${ZENML_SERVER_URL}..."
        uv run zenml connect --url "$ZENML_SERVER_URL"
        print_success "Connected to ZenML server"
    else
        print_info "No ZenML server configured (using local ZenML)"
    fi
}

# Check required services
check_services() {
    print_info "Checking required services..."

    # Check Qdrant
    QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
    print_info "Checking Qdrant at ${QDRANT_URL}..."
    if curl -f "${QDRANT_URL}/health" &> /dev/null; then
        print_success "Qdrant is accessible"
    else
        print_error "Qdrant is not accessible at ${QDRANT_URL}"
        print_info "Please start Qdrant or update QDRANT_URL in .env"
        exit 1
    fi

    # Check PostgreSQL (optional)
    if [ -n "$DATABASE_URL" ]; then
        print_info "PostgreSQL configured: ${DATABASE_URL}"
    else
        print_warning "No PostgreSQL configured (optional)"
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

    if [ ! -f "zenml_config.yaml" ]; then
        print_warning "zenml_config.yaml not found (optional)"
    fi

    print_success "Configuration files verified"
}

# Display environment info
display_info() {
    echo ""
    echo "========================================================"
    echo "  ScriptGuard Podrun Environment Ready!"
    echo "========================================================"
    echo ""
    echo "Environment:"
    echo "  Python:      $(python3 --version)"
    echo "  uv:          $(uv --version)"
    echo "  ZenML:       $(uv run zenml version 2>/dev/null || echo 'Not available')"
    echo ""
    echo "Configuration:"
    echo "  Config file: config.yaml"
    echo "  Env file:    .env"
    echo ""
    echo "Services:"
    echo "  Qdrant:      ${QDRANT_URL:-http://localhost:6333}"
    if [ -n "$DATABASE_URL" ]; then
        echo "  PostgreSQL:  ${DATABASE_URL}"
    fi
    if [ -n "$ZENML_SERVER_URL" ]; then
        echo "  ZenML Server: ${ZENML_SERVER_URL}"
    fi
    echo ""
    echo "Useful Commands:"
    echo "  Run training:        uv run python src/main.py"
    echo "  ZenML status:        uv run zenml status"
    echo "  ZenML pipelines:     uv run zenml pipeline list"
    echo "  ZenML runs:          uv run zenml pipeline runs list"
    echo ""
    echo "  Interactive shell:   uv run python"
    echo "  Test environment:    uv run pytest tests/"
    echo ""
    echo "========================================================"
    echo ""
}

# Run training pipeline
run_training() {
    print_info "Starting training pipeline..."

    # Load environment variables
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi

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

    check_python
    check_uv
    check_cuda
    setup_env
    create_directories
    verify_config

    install_dependencies
    verify_zenml
    init_zenml
    check_services

    display_info

    if [ $CHECK_ONLY -eq 1 ]; then
        print_success "Environment check complete!"
        exit 0
    fi

    # Ask if user wants to start training
    read -p "Start training pipeline now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_training
    else
        print_info "Training skipped. Run manually with: uv run python src/main.py"
    fi

    print_success "Setup complete!"
}

# Run main function
main
