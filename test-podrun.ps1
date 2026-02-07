# ScriptGuard Quick Test for Podrun (PowerShell)
# Tests the setup without running full training

Write-Host "========================================================"
Write-Host "  ScriptGuard Podrun Quick Test"
Write-Host "========================================================"
Write-Host ""

$ErrorActionPreference = "Stop"

# 1. Check Python
Write-Host "[1/8] Checking Python..."
try {
    $pythonVersion = python --version
    Write-Host "✓ $pythonVersion"
} catch {
    Write-Host "✗ ERROR: Python3 not found" -ForegroundColor Red
    exit 1
}

# 2. Check uv
Write-Host "[2/8] Checking uv..."
try {
    $uvVersion = uv --version
    Write-Host "✓ $uvVersion"
} catch {
    Write-Host "✗ ERROR: uv not found. Install with: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Red
    exit 1
}

# 3. Check ZenML
Write-Host "[3/8] Checking ZenML installation..."
try {
    $zenmlCheck = uv run python -c "import zenml; print(f'ZenML version: {zenml.__version__}')"
    Write-Host "✓ $zenmlCheck"
} catch {
    Write-Host "✗ ERROR: ZenML not installed" -ForegroundColor Red
    exit 1
}

# 4. Check PyTorch
Write-Host "[4/8] Checking PyTorch..."
try {
    $torchCheck = uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    Write-Host "✓ $torchCheck"
} catch {
    Write-Host "✗ ERROR: PyTorch not installed" -ForegroundColor Red
    exit 1
}

# 5. Check Qdrant
Write-Host "[5/8] Checking Qdrant connection..."
$qdrantUrl = if ($env:QDRANT_URL) { $env:QDRANT_URL } else { "http://localhost:6333" }
try {
    $response = Invoke-WebRequest -Uri "$qdrantUrl/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "✓ Qdrant is accessible"
} catch {
    Write-Host "✗ ERROR: Cannot connect to Qdrant at $qdrantUrl" -ForegroundColor Red
    exit 1
}

# 6. Check config files
Write-Host "[6/8] Checking configuration files..."
if (!(Test-Path "config.yaml")) {
    Write-Host "✗ ERROR: config.yaml not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ config.yaml found"

if (!(Test-Path ".env")) {
    Write-Host "⚠ WARNING: .env not found (optional)" -ForegroundColor Yellow
}

# 7. Check project structure
Write-Host "[7/8] Checking project structure..."
if (!(Test-Path "src\scriptguard")) {
    Write-Host "✗ ERROR: src\scriptguard directory not found" -ForegroundColor Red
    exit 1
}
if (!(Test-Path "src\main.py")) {
    Write-Host "✗ ERROR: src\main.py not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Project structure OK"

# 8. Run Python environment check
Write-Host "[8/8] Running comprehensive environment check..."
try {
    uv run python check_podrun_env.py
    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  ✓ All checks passed!" -ForegroundColor Green
    Write-Host "  You can now run: uv run python src/main.py"
    Write-Host "========================================================"
} catch {
    Write-Host "✗ ERROR: Environment check failed" -ForegroundColor Red
    exit 1
}
