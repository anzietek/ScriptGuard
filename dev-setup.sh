#!/bin/bash
# ScriptGuard Development Setup Script
# This script sets up the infrastructure and prepares local development environment
# Usage:
#   ./dev-setup.sh         - Normal setup
#   ./dev-setup.sh --clean - Clean databases and restart

set -e  # Exit on error

# Check for --clean argument
CLEAN_MODE=0
if [ "$1" == "--clean" ]; then
    CLEAN_MODE=1
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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed: $(docker-compose --version)"
}

# Setup environment file
setup_env() {
    if [ ! -f .env.dev ]; then
        print_error ".env.dev file not found!"
        exit 1
    fi

    if [ ! -f .env ]; then
        print_info "Creating .env from .env.dev..."
        cp .env.dev .env
        print_success ".env file created"
    else
        print_warning ".env file already exists, skipping..."
    fi
}

# Clean databases
clean_databases() {
    print_warning "Clean mode enabled - this will DELETE ALL DATA!"
    read -p "Are you sure you want to clean databases? (yes/no): " CONFIRM
    if [ "$CONFIRM" == "yes" ]; then
        print_info "Stopping services and cleaning databases..."
        cd docker
        docker-compose -f docker-compose.dev.yml down -v
        cd ..
        print_success "Databases cleaned"
    else
        print_info "Clean cancelled, proceeding with normal startup..."
        CLEAN_MODE=0
    fi
}

# Start infrastructure
start_infrastructure() {
    print_info "Starting infrastructure services (PostgreSQL, Qdrant, ZenML)..."

    cd docker
    docker-compose -f docker-compose.dev.yml --profile with-zenml up -d postgres qdrant zenml
    cd ..

    print_success "Infrastructure services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_info "Waiting for services to be healthy..."

    # Wait for PostgreSQL
    print_info "Waiting for PostgreSQL..."
    timeout=60
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if docker exec scriptguard-postgres-dev pg_isready -U scriptguard -d scriptguard > /dev/null 2>&1; then
            print_success "PostgreSQL is ready"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ $elapsed -ge $timeout ]; then
        print_error "PostgreSQL failed to start within ${timeout}s"
        exit 1
    fi

    # Wait for Qdrant
    print_info "Waiting for Qdrant..."
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:6333/ > /dev/null 2>&1; then
            print_success "Qdrant is ready"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ $elapsed -ge $timeout ]; then
        print_error "Qdrant failed to start within ${timeout}s"
        exit 1
    fi

    # Wait for ZenML
    print_info "Waiting for ZenML..."
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8237/health > /dev/null 2>&1; then
            print_success "ZenML is ready"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ $elapsed -ge $timeout ]; then
        print_warning "ZenML failed to start within ${timeout}s (may still be initializing)"
    fi
}
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ $elapsed -ge $timeout ]; then
        print_error "Qdrant failed to start within ${timeout}s"
        exit 1
    fi
}

# Initialize database
init_database() {
    print_info "Checking database initialization..."

    # Check if tables exist
    if docker exec scriptguard-postgres-dev psql -U scriptguard -d scriptguard -c '\dt' | grep -q 'samples'; then
        print_success "Database already initialized"
    else
        print_info "Initializing database schema..."
        docker exec scriptguard-postgres-dev psql -U scriptguard -d scriptguard -f /docker-entrypoint-initdb.d/init.sql
        print_success "Database initialized"
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."

    mkdir -p data
    mkdir -p models
    mkdir -p logs
    mkdir -p model_checkpoints

    print_success "Directories created"
}

# Setup Python environment
setup_python_env() {
    print_info "Setting up Python environment..."

    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi

    print_info "Activating virtual environment and installing dependencies..."

    # Activate venv and install
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi

    pip install --upgrade pip
    pip install -e .

    print_success "Python dependencies installed"
}

# Bootstrap Qdrant with CVE data
bootstrap_qdrant() {
    print_info "Bootstrapping Qdrant with CVE data..."

    # Activate venv
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi

    python -c "
from scriptguard.rag import QdrantStore, bootstrap_cve_data
import logging
logging.basicConfig(level=logging.INFO)
store = QdrantStore()
bootstrap_cve_data(store)
print('CVE data bootstrapped successfully')
"

    print_success "Qdrant bootstrapped with CVE data"
}

# Display connection info
display_info() {
    echo ""
    echo "========================================================"
    echo "  ScriptGuard Development Environment Ready!"
    echo "========================================================"
    echo ""
    echo "Infrastructure Services:"
    echo "  PostgreSQL:  localhost:5432"
    echo "    Database:  scriptguard"
    echo "    User:      scriptguard"
    echo "    Password:  scriptguard"
    echo ""
    echo "  Qdrant:      http://localhost:6333"
    echo "    Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "  ZenML:       http://localhost:8237"
    echo "    Dashboard: http://localhost:8237"
    echo ""
    echo "Connection Strings:"
    echo "  PostgreSQL: postgresql://scriptguard:scriptguard@localhost:5432/scriptguard"
    echo "  Qdrant:     http://localhost:6333"
    echo "  ZenML:      http://localhost:8237"
    echo ""
    echo "Useful Commands:"
    echo "  Start infrastructure:  cd docker && docker-compose -f docker-compose.dev.yml --profile with-zenml up -d"
    echo "  Stop infrastructure:   cd docker && docker-compose -f docker-compose.dev.yml down"
    echo "  Clean databases:       ./dev-setup.sh --clean"
    echo "  View logs:            cd docker && docker-compose -f docker-compose.dev.yml logs -f"
    echo ""
    echo "  Activate venv:        source venv/bin/activate  (Linux/Mac)"
    echo "                        venv\\Scripts\\activate     (Windows)"
    echo ""
    echo "  Run training:         python src/main.py"
    echo "  Run API:              uvicorn scriptguard.api.main:app --reload"
    echo ""
    echo "  Python shell:         python"
    echo "  PostgreSQL shell:     docker exec -it scriptguard-postgres-dev psql -U scriptguard -d scriptguard"
    echo ""
    echo "Optional Services (with profiles):"
    echo "  With pgAdmin:         cd docker && docker-compose -f docker-compose.dev.yml --profile with-pgadmin up -d"
    echo "    pgAdmin URL:        http://localhost:5050"
    echo ""
    echo "  With monitoring:      cd docker && docker-compose -f docker-compose.dev.yml --profile monitoring up -d"
    echo "    Prometheus:         http://localhost:9090"
    echo "    Grafana:            http://localhost:3000 (admin/admin)"
    echo ""
    echo "========================================================"
    echo ""
}

# Main execution
main() {
    echo "========================================================"
    echo "  ScriptGuard Development Setup"
    if [ $CLEAN_MODE -eq 1 ]; then
        echo "  MODE: Clean databases and restart"
    else
        echo "  MODE: Normal setup"
    fi
    echo "========================================================"
    echo ""

    check_docker
    check_docker_compose
    setup_env
    create_directories

    # Clean databases if --clean flag is set
    if [ $CLEAN_MODE -eq 1 ]; then
        clean_databases
    fi

    start_infrastructure
    wait_for_services
    init_database

    # Ask if user wants to setup Python environment
    read -p "Setup Python virtual environment and install dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_python_env
        bootstrap_qdrant
    fi

    display_info

    print_success "Setup complete! You can now start developing."
}

# Run main function
main
