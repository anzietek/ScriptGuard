#!/bin/bash
"""
ScriptGuard Podrun Setup Script
This script prepares the environment for running training pipelines on Podrun with ZenML.
It assumes Postgres and Qdrant are external services.
It sets up a local ZenML server for orchestration/dashboard.

Usage:
  ./podrun-setup.sh         - Setup and ask to run training
  ./podrun-setup.sh -y      - Setup and auto-start training (non-interactive)
  ./podrun-setup.sh --check - Check environment only
"""

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
    if [[ ! $PYTHON_VERSION =~ ^3\.(10|11|12|13) ]]; then
        print_warning "Python version ${PYTHON_VERSION} detected. Recommended: 3.10+"
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
    # We bind to 0.0.0.0 to allow external access via RunPod proxy
    print_info "Starting local ZenML Server..."

    if pgrep -f "zenml.services.zen_server" > /dev/null; then
        print_warning "ZenML Server is already running."
    else
        # Using nohup or background flag if supported, here we use zenml up non-blocking
        # Note: --blocking=False might not be supported in all versions, falling back to background process
        nohup uv run zenml up --host 0.0.0.0 --port 8237 > logs/zenml_server.log 2>&1 &

        # Give it a moment to start
        sleep 5
        print_success "ZenML Server started on port 8237 (check logs/zenml_server.log)"
    fi
}

# Check external services (Qdrant & Postgres)
check_services() {
    print_info "Checking configuration for external services..."

    # Load environment variables just for this check
    if [ -f .env ]; then
        # Export vars ignoring comments
        export $(grep -v '^#' .env | xargs)
    fi

    # Check Qdrant URL
    if [ -n "$QDRANT_URL" ]; then
        print_info "External Qdrant URL found: ${QDRANT_URL}"
        # Simple connectivity check
        if curl --max-time 5 -s "${QDRANT_URL}/health" > /dev/null || curl --max-time 5 -s "${QDRANT_URL}" > /dev/null; then
             print_success "Qdrant endpoint is reachable."
        else
             print_warning "Could not reach Qdrant at ${QDRANT_URL}. Check your VPN/Firewall settings."
        fi
    else
        print_warning "QDRANT_URL is not set in .env"
    fi

    # Check Database URL
    if [ -n "$DATABASE_URL" ]; then
        print_success "PostgreSQL DATABASE_URL is configured."
    else
        print_warning "DATABASE_URL is not set in .env"
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
    echo ""
    echo "========================================================"
    echo "  ScriptGuard Podrun Environment Ready!"
    echo "========================================================"
    echo ""
    echo "Environment:"
    echo "  Python:      $(python3 --version)"
    echo "  uv:          $(uv --version)"
    echo "  ZenML:       $(uv run zenml version)"
    echo ""
    echo "Services:"
    echo "  ZenML Server: http://localhost:8237 (Exposed via RunPod port 8237)"
    echo "  Qdrant (Ext): ${QDRANT_URL:-Not Set}"
    echo "  Postgres:     ${DATABASE_URL:-Not Set}"
    echo ""
    echo "Logs:"
    echo "  ZenML Logs:   logs/zenml_server.log"
    echo ""
    echo "========================================================"
    echo ""
}

# Run training pipeline
run_training() {
    print_info "Starting training pipeline..."

    # Load environment variables
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    print_info "Running training with ZenML..."
    # Assuming src/main.py is the entrypoint
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
    check_services
    init_zenml # Starts local ZenML server

    display_info

    if [ $CHECK_ONLY -eq 1 ]; then
        print_success "Environment check complete!"
        exit 0
    fi

    # Handle automatic approval for RunPod non-interactive execution
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