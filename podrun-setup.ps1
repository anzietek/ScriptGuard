# ScriptGuard Podrun Setup Script for Windows/PowerShell
# This script prepares the environment for running training pipelines on Podrun with ZenML
# Usage:
#   .\podrun-setup.ps1         - Setup and run training
#   .\podrun-setup.ps1 -Check  - Check environment only

param(
    [switch]$Check
)

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check Python version
function Check-Python {
    Write-Info "Checking Python version..."

    if (!(Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python is not installed. Please install Python 3.12 or 3.13."
        exit 1
    }

    $pythonVersion = python --version
    Write-Success "Python found: $pythonVersion"

    # Must match pyproject.toml: requires-python = ">=3.10,<3.13"
    if ($pythonVersion -notmatch "Python 3\.(10|11|12)") {
        Write-Error "Python version is not compatible. Required: >=3.10,<3.13 (per pyproject.toml)"
        exit 1
    }
}

# Check if uv is installed
function Check-UV {
    Write-Info "Checking for uv package installer..."

    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Warning "uv is not installed. Installing uv..."

        # Install uv on Windows
        irm https://astral.sh/uv/install.ps1 | iex

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

        Write-Success "uv installed successfully"
    } else {
        $uvVersion = uv --version
        Write-Success "uv is already installed: $uvVersion"
    }
}

# Check CUDA availability
function Check-CUDA {
    Write-Info "Checking CUDA availability..."

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Success "CUDA detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } else {
        Write-Warning "CUDA not detected. Training will use CPU (slower)."
    }
}

# Setup environment file
function Setup-Env {
    Write-Info "Setting up environment configuration..."

    if (!(Test-Path .env)) {
        if (Test-Path .env.example) {
            Write-Info "Creating .env from .env.example..."
            Copy-Item .env.example .env
            Write-Success ".env file created"
            Write-Warning "Please edit .env file with your credentials!"
        } else {
            Write-Error ".env.example not found!"
            exit 1
        }
    } else {
        Write-Success ".env file already exists"
    }
}

# Install dependencies with uv
function Install-Dependencies {
    Write-Info "Installing dependencies with uv..."

    # uv sync installs everything from pyproject.toml including
    # the correct PyTorch version (2.6.0+cu124) with platform-specific wheels
    uv sync

    Write-Success "Dependencies installed successfully"
}

# Verify critical dependencies
function Verify-Dependencies {
    Write-Info "Verifying critical dependencies..."

    # Verify ZenML
    try {
        $zenmlVersion = uv run zenml version 2>$null
        Write-Success "ZenML installed: $zenmlVersion"
    } catch {
        Write-Error "ZenML installation failed!"
        exit 1
    }

    # Verify unsloth (required by src/main.py)
    try {
        $unslothVersion = uv run python -c "import unsloth; print(unsloth.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $unslothVersion) {
            Write-Success "Unsloth installed: $unslothVersion"
        } else {
            Write-Warning "Unsloth import failed. Training may not work with optimizations."
        }
    } catch {
        Write-Warning "Unsloth import failed. Training may not work with optimizations."
    }

    # Verify PyTorch and CUDA
    try {
        $torchInfo = uv run python -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>$null
        Write-Info "PyTorch: $torchInfo"
    } catch {
        Write-Warning "PyTorch verification failed."
    }
}

# Initialize ZenML
function Initialize-ZenML {
    Write-Info "Initializing ZenML..."

    if (Test-Path .zen) {
        Write-Warning "ZenML already initialized in this directory"
    } else {
        uv run zenml init
        Write-Success "ZenML initialized"
    }

    # Connect to ZenML server if configured
    if ($env:ZENML_SERVER_URL) {
        Write-Info "Connecting to ZenML server at $env:ZENML_SERVER_URL..."
        uv run zenml connect --url $env:ZENML_SERVER_URL
        Write-Success "Connected to ZenML server"
    } else {
        Write-Info "No ZenML server configured (using local ZenML)"
    }
}

# Check required services
function Check-Services {
    Write-Info "Checking required services..."

    # Load .env
    if (Test-Path .env) {
        Get-Content .env | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }

    # Check Qdrant using QDRANT_HOST/QDRANT_PORT (matching .env.example)
    $qdrantHost = if ($env:QDRANT_HOST) { $env:QDRANT_HOST } else { "localhost" }
    $qdrantPort = if ($env:QDRANT_PORT) { $env:QDRANT_PORT } else { "6333" }
    $qdrantUrl = "http://${qdrantHost}:${qdrantPort}"
    Write-Info "Checking Qdrant at $qdrantUrl..."

    try {
        $response = Invoke-WebRequest -Uri "$qdrantUrl/healthz" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Qdrant is reachable at $qdrantUrl"
        }
    } catch {
        Write-Warning "Could not reach Qdrant at $qdrantUrl. Check that Qdrant is running."
    }

    # Check PostgreSQL
    if ($env:POSTGRES_HOST) {
        $pgPort = if ($env:POSTGRES_PORT) { $env:POSTGRES_PORT } else { "5432" }
        $pgDb = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "scriptguard" }
        Write-Success "PostgreSQL configured: ${env:POSTGRES_HOST}:${pgPort}/${pgDb}"
    } else {
        Write-Warning "POSTGRES_HOST is not set in .env"
    }

    # Check WandB (config.yaml reports to wandb)
    if ($env:WANDB_API_KEY) {
        $wandbProject = if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { "scriptguard" }
        Write-Success "WandB API key configured (project: $wandbProject)"
    } else {
        Write-Warning "WANDB_API_KEY is not set in .env. config.yaml has report_to: [wandb] - training will fail without it."
    }
}

# Create necessary directories
function Create-Directories {
    Write-Info "Creating necessary directories..."

    $directories = @("data", "models", "logs", "model_checkpoints", ".cache")

    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
        }
    }

    Write-Success "Directories created"
}

# Verify configuration
function Verify-Config {
    Write-Info "Verifying configuration files..."

    if (!(Test-Path "config.yaml")) {
        Write-Error "config.yaml not found!"
        exit 1
    }

    if (!(Test-Path "zenml_config.yaml")) {
        Write-Warning "zenml_config.yaml not found (optional)"
    }

    Write-Success "Configuration files verified"
}

# Display environment info
function Display-Info {
    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  ScriptGuard Podrun Environment Ready!"
    Write-Host "========================================================"
    Write-Host ""
    Write-Host "Environment:"
    Write-Host "  Python:      $(python --version)"
    Write-Host "  uv:          $(uv --version)"

    try {
        $zenmlVer = uv run zenml version 2>$null
        Write-Host "  ZenML:       $zenmlVer"
    } catch {
        Write-Host "  ZenML:       Not available"
    }

    Write-Host ""
    Write-Host "Configuration:"
    Write-Host "  Config file: config.yaml"
    Write-Host "  Env file:    .env"
    Write-Host ""
    Write-Host "Services:"
    $qdrantUrl = if ($env:QDRANT_URL) { $env:QDRANT_URL } else { "http://localhost:6333" }
    Write-Host "  Qdrant:      $qdrantUrl"

    if ($env:DATABASE_URL) {
        Write-Host "  PostgreSQL:  $env:DATABASE_URL"
    }

    if ($env:ZENML_SERVER_URL) {
        Write-Host "  ZenML Server: $env:ZENML_SERVER_URL"
    }

    Write-Host ""
    Write-Host "Useful Commands:"
    Write-Host "  Run training:        uv run python src/main.py"
    Write-Host "  ZenML status:        uv run zenml status"
    Write-Host "  ZenML pipelines:     uv run zenml pipeline list"
    Write-Host "  ZenML runs:          uv run zenml pipeline runs list"
    Write-Host ""
    Write-Host "  Interactive shell:   uv run python"
    Write-Host "  Test environment:    uv run pytest tests/"
    Write-Host ""
    Write-Host "========================================================"
    Write-Host ""
}

# Run training pipeline
function Run-Training {
    Write-Info "Starting training pipeline..."

    # Load environment variables from .env
    if (Test-Path .env) {
        Get-Content .env | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }

    Write-Info "Running training with ZenML..."
    uv run python src/main.py

    Write-Success "Training pipeline completed!"
}

# Main execution
function Main {
    Write-Host "========================================================"
    Write-Host "  ScriptGuard Podrun Setup"

    if ($Check) {
        Write-Host "  MODE: Environment check only"
    } else {
        Write-Host "  MODE: Full setup and training"
    }

    Write-Host "========================================================"
    Write-Host ""

    # Phase 1: Checks (always run)
    Check-Python
    Check-UV
    Check-CUDA
    Setup-Env
    Create-Directories
    Verify-Config

    # Phase 2: Installation (skip in check mode)
    if (-not $Check) {
        Install-Dependencies
        Verify-Dependencies
        Check-Services
        Initialize-ZenML
    } else {
        # In check mode, only verify services connectivity
        Check-Services
        Write-Success "Environment check complete!"
        exit 0
    }

    Display-Info

    # Phase 3: Training
    $response = Read-Host "Start training pipeline now? (y/n)"

    if ($response -eq "y" -or $response -eq "Y") {
        Run-Training
    } else {
        Write-Info "Training skipped. Run manually with: uv run python src/main.py"
    }

    Write-Success "Setup complete!"
}

# Run main function
Main
