#!/bin/bash
# Test script to verify ZenML automatic authentication fix

set -e

echo "=== ZenML Authentication Fix Verification ==="
echo ""

# Source .env.podrun to get credentials
if [ -f ".env.podrun" ]; then
    echo "[1/5] Loading .env.podrun configuration..."
    set -a
    source .env.podrun
    set +a
    echo "✓ Configuration loaded"
else
    echo "❌ .env.podrun not found"
    exit 1
fi

# Check required variables
echo ""
echo "[2/5] Verifying required environment variables..."

if [ -z "${ZENML_SERVER_URL:-}" ]; then
    echo "❌ ZENML_SERVER_URL not set"
    exit 1
fi
echo "✓ ZENML_SERVER_URL: ${ZENML_SERVER_URL}"

if [ -z "${ZENML_API_KEY:-}" ]; then
    echo "❌ ZENML_API_KEY not set"
    exit 1
fi
echo "✓ ZENML_API_KEY: ${ZENML_API_KEY:0:20}... (${#ZENML_API_KEY} chars)"

# Set ZENML_URL (same as podrun-setup.sh does)
ZENML_URL="${ZENML_SERVER_URL:-http://localhost:8237}"
echo "✓ ZENML_URL: ${ZENML_URL}"

# Export ZenML client environment variables (as fixed script does)
echo ""
echo "[3/5] Exporting ZenML client environment variables..."
export ZENML_STORE_URL="${ZENML_URL}"
export ZENML_STORE_API_KEY="${ZENML_API_KEY}"
export ZENML_SERVER_URL="${ZENML_URL}"
echo "✓ ZENML_STORE_URL exported"
echo "✓ ZENML_STORE_API_KEY exported"
echo "✓ ZENML_SERVER_URL exported"

# Test health endpoint
echo ""
echo "[4/5] Testing ZenML server connection..."
if curl --max-time 5 -s "${ZENML_URL}/health" > /dev/null 2>&1; then
    echo "✓ ZenML server is accessible at ${ZENML_URL}"
else
    echo "⚠ ZenML server not accessible (this is OK if SSH tunnel is not active)"
    echo "  To test full authentication, ensure SSH tunnel is running:"
    echo "  ssh -N -L 8237:localhost:8237 deployer@<VPS_IP>"
    exit 0
fi

# Test Python client connection
echo ""
echo "[5/5] Testing ZenML Python client authentication..."
uv run python -c "
import os
from zenml.client import Client

try:
    # Verify environment variables are accessible
    print(f'  Environment check:')
    print(f'  - ZENML_STORE_URL: {os.getenv(\"ZENML_STORE_URL\")}')
    print(f'  - ZENML_STORE_API_KEY: {os.getenv(\"ZENML_STORE_API_KEY\", \"\")[:20]}...')

    # Test client connection
    client = Client()
    print(f'  ✓ Connected to: {client.zen_store.url}')
    print(f'  ✓ Active workspace: {client.active_workspace.name}')
    print(f'  ✓ Active user: {client.active_user.name}')
    print('')
    print('✅ ZenML authentication successful - NO browser prompt needed!')
except Exception as e:
    print(f'  ❌ Authentication failed: {e}')
    exit(1)
" || {
    echo "❌ Python client authentication failed"
    exit 1
}

echo ""
echo "=== All Verification Tests Passed ==="
echo ""
echo "Next steps:"
echo "1. Run './podrun-setup.sh' and verify no browser prompts appear"
echo "2. Look for log message: 'Authentication will use API key from environment (no browser needed)'"
echo "3. Run pipeline: 'uv run python -m scriptguard.main --mode pipeline'"
