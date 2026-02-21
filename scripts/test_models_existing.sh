#!/bin/bash
# Test both models using EXISTING collections (no re-vectorization)
#
# Prerequisites:
# 1. Collections must exist:
#    - code_samples_balanced (UniXcoder)
#    - code_samples_jina_v3 (Jina-v3)
#
# Usage:
#    bash scripts/test_models_existing.sh

echo "======================================================================="
echo "Testing Models with Existing Collections"
echo "======================================================================="

# Test UniXcoder
echo ""
echo "[1/2] Testing UniXcoder (code_samples_balanced)..."
python scripts/test_hybrid_full_eval.py --use-existing

if [ $? -eq 0 ]; then
    echo "✓ UniXcoder test completed"
else
    echo "✗ UniXcoder test failed"
    exit 1
fi

# Test Jina-v3
echo ""
echo "[2/2] Testing Jina-v3 (code_samples_jina_v3)..."
python scripts/test_jina_v3_comparison.py --use-existing

if [ $? -eq 0 ]; then
    echo "✓ Jina-v3 test completed"
else
    echo "✗ Jina-v3 test failed"
    exit 1
fi

echo ""
echo "======================================================================="
echo "All tests completed successfully!"
echo "======================================================================="
