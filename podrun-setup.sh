#!/bin/bash
#
# ScriptGuard RunPod Setup Script (Fixed: Auto-install lsof + dependencies)
#
# Features:
# - Auto-installs 'lsof', 'ssh', 'git', 'curl' if missing
# - Auto-installs 'uv' and Python 3.12
# - Sets up ZenML Server accessible via RunPod Proxy/TCP
# - Checks for Persistent Volume (/workspace)
# - Establishes Secure SSH Tunnel (Key is deleted from disk immediately after use)
# - Auto-starts training pipeline
# - Supports Clean Install (--clean) to wipe .venv
#

set -e  # Exit on error

# --- Global Configuration ---
# Fix for UV path: Add both .local/bin and .cargo/bin just in case
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_LINK_MODE=copy
# Tunnel Configuration
REMOTE_IP="62.171.130.236"
REMOTE_USER="deployer"
# Key will be stored here TEMPORARILY and then deleted
KEY_TEMP_PATH="/tmp/deployer_key_temp"
# Arguments parsing
CHECK_ONLY=0
AUTO_APPROVE=0
CLEAN_INSTALL=0

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
    --clean|-c)
      CLEAN_INSTALL=1
      shift
      ;;
  esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Helper Functions ---
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Core Checks & System Setup ---

check_workspace() {
    print_info "Checking filesystem location..."
    if [[ "$(pwd)" != *"/workspace"* ]]; then
        print_warning "-----------------------------------------------------------"
        print_warning "CRITICAL: You are NOT in '/workspace'."
        print_warning "Data will be lost on restart. Please move to /workspace."
        print_warning "-----------------------------------------------------------"
        if [ $AUTO_APPROVE -eq 0 ]; then
             read -p "Continue anyway (Not Recommended)? (y/n) " -n 1 -r; echo
             if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
        fi
    else
        print_success "Running safely inside persistent volume."
    fi
}

check_system_tools() {
    print_info "Checking system dependencies..."
    local MISSING_PKGS=""

    # Check for essential tools
    if ! command -v lsof &> /dev/null; then MISSING_PKGS="$MISSING_PKGS lsof"; fi
    if ! command -v ssh &> /dev/null; then MISSING_PKGS="$MISSING_PKGS openssh-client"; fi
    if ! command -v curl &> /dev/null; then MISSING_PKGS="$MISSING_PKGS curl"; fi
    if ! command -v git &> /dev/null; then MISSING_PKGS="$MISSING_PKGS git"; fi
    # procps is needed for 'ps' command, often missing in minimal containers
    if ! command -v ps &> /dev/null; then MISSING_PKGS="$MISSING_PKGS procps"; fi

    if [ ! -z "$MISSING_PKGS" ]; then
        print_warning "Missing system tools found: $MISSING_PKGS"
        print_info "Installing missing packages via apt-get..."

        # Ensure non-interactive installation
        export DEBIAN_FRONTEND=noninteractive

        # Run update and install in one go to be efficient
        apt-get update && apt-get install -y $MISSING_PKGS

        print_success "System tools installed successfully."
    else
        print_success "All system tools (lsof, ssh, git, curl) are present."
    fi
}

load_env() {
    if [ -f .env ]; then set -a; source .env; set +a; fi
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        print_warning "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # FIX: Refresh PATH to include the new installation
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    print_success "uv is ready: $(uv --version)"
}

check_python_system() {
    if command -v python3 &> /dev/null; then
        print_success "System Python found (uv will manage Python 3.12 for project)"
    else
        print_warning "System Python missing, relying on uv."
    fi
}

# --- SSH Tunnel Setup (Secure Mode) ---

setup_tunnel() {
    print_info "Setting up Secure SSH Tunnel to $REMOTE_IP..."

    # (Note: SSH client installation is now handled in check_system_tools)

    # 1. Check if tunnel is already running
    if pgrep -f "ssh.*$REMOTE_IP" > /dev/null; then
        print_success "Tunnel is already active."
        return
    fi

    # 2. Request Key (Since we delete it, we must ask every time if tunnel is down)
    echo "--------------------------------------------------------"
    echo -e "${YELLOW}SECURITY CHECK: SSH Tunnel is required.${NC}"
    echo "Please paste your Private Key content (id_ed25519) below."
    echo "The file will be used to connect and then IMMEDIATELY DELETED."
    echo "--------------------------------------------------------"
    echo "Press ENTER, paste the key, then press Ctrl+D (EOF)."

    # Read multiline input
    cat > "$KEY_TEMP_PATH"

    # Check if file is not empty
    if [ ! -s "$KEY_TEMP_PATH" ]; then
        print_error "Key file is empty. Aborting tunnel setup."
        return
    fi

    # 3. Set Permissions (Critical)
    chmod 600 "$KEY_TEMP_PATH"

    print_info "Establishing tunnel..."

    # 4. Start SSH in background
    ssh -4 -f -N \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -i "$KEY_TEMP_PATH" \
        -L 5432:127.0.0.1:5432 \
        -L 6333:127.0.0.1:6333 \
        -L 5050:127.0.0.1:5050 \
        $REMOTE_USER@$REMOTE_IP

    # Wait for connection
    sleep 5

    # 5. SECURITY WIPE - Delete key from disk
    rm -f "$KEY_TEMP_PATH"
    print_warning "Private key file has been deleted from disk for security."

    # 6. Verify
    if pgrep -f "ssh.*$REMOTE_IP" > /dev/null; then
        print_success "Tunnel ESTABLISHED."
        echo "   - Postgres: localhost:5432 -> Remote:5432"
        echo "   - Qdrant:   localhost:6333 -> Remote:6333"
    else
        print_error "Failed to establish tunnel. Check your key and try again."
        exit 1
    fi
}

setup_env_file() {
    print_info "Checking .env configuration..."
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env from template."
        else
            touch .env
            print_warning "Created empty .env file."
        fi
    fi
}

# --- Installation ---

install_dependencies() {
    # Clean Install Logic
    if [ "$CLEAN_INSTALL" -eq 1 ]; then
        print_warning "Clean Install requested: Removing .venv..."
        rm -rf .venv
        print_success ".venv removed."
    fi

    print_info "Syncing dependencies with uv..."
    uv sync
}

# --- Services ---

init_zenml() {
    print_info "Initializing ZenML..."
    [ ! -d ".zen" ] && uv run zenml init

    # FIXED: This check now relies on 'lsof' which is guaranteed to be installed
    if ! lsof -Pi :8237 -sTCP:LISTEN -t >/dev/null ; then
        nohup uv run zenml up --host 0.0.0.0 --port 8237 > logs/zenml_server.log 2>&1 &
        sleep 5
        print_success "ZenML Server running on port 8237 (Use TCP Port Mapping)"
    else
        print_success "ZenML Server is already running."
    fi
}

check_services() {
    print_info "Checking service connectivity via Tunnel..."

    # Check Qdrant Local (via Tunnel)
    if curl --max-time 3 -s "http://localhost:6333/healthz" > /dev/null; then
        print_success "Qdrant reachable via localhost:6333 (Tunnel OK)"
    else
        print_warning "Cannot reach Qdrant on localhost:6333. Tunnel might be down."
    fi
}

create_directories() {
    mkdir -p data models logs model_checkpoints .cache
}

# --- Main ---

main() {
    echo "========================================================"
    echo "   ScriptGuard RunPod Setup (Secure Tunnel + Auto-Fix)"
    echo "========================================================"

    check_workspace
    check_system_tools   # <--- ADDED: Installs lsof, ssh, etc.
    check_python_system
    check_uv
    create_directories
    setup_env_file

    # Setup SSH Tunnel BEFORE syncing deps
    setup_tunnel

    if [ $CHECK_ONLY -eq 0 ]; then
        install_dependencies
        init_zenml
    fi

    check_services

    echo "========================================================"
    echo "   Setup Complete!"
    echo "========================================================"

    if [ $CHECK_ONLY -eq 1 ]; then exit 0; fi

    # --- TRAINING LAUNCH SECTION ---

    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} Before starting training, ensure you have edited"
    echo -e "           the ${BLUE}.env${NC} file with your API keys (WandB, HF, etc.)!"
    echo ""

    if [ $AUTO_APPROVE -eq 1 ]; then
        print_info "Auto-start enabled. Launching training pipeline..."
        load_env
        uv run python src/main.py
    else
        read -p "Have you updated .env and want to start training now? (y/n) " -n 1 -r; echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            load_env
            print_info "Starting pipeline..."
            uv run python src/main.py
        else
            print_info "Pipeline skipped."
            echo "To run manually:"
            echo "1. nano .env"
            echo "2. uv run python src/main.py"
        fi
    fi
}

main "$@"
