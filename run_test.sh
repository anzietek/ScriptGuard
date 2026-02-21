#!/bin/bash
# Quick test script for ScriptGuard pipeline
# Uses config.test.yaml for fast validation

echo "================================================"
echo "ScriptGuard Pipeline - Quick Test"
echo "================================================"
echo ""
echo "This will run a minimal test of the entire pipeline:"
echo "- Minimal data ingestion (~30 samples)"
echo "- Fast vectorization (100 samples max)"
echo "- 1 epoch training with small batches"
echo "- Quick evaluation"
echo ""
echo "Estimated time: 10-15 minutes"
echo "================================================"
echo ""

# Set environment to use test config
export CONFIG_PATH=config.test.yaml

# Create necessary directories
mkdir -p logs test_models cache

echo "[1/4] Starting pipeline with test configuration..."
python -m scriptguard.pipelines.training_pipeline

if [ $? -ne 0 ]; then
    echo ""
    echo "================================================"
    echo "ERROR: Pipeline failed with exit code $?"
    echo "================================================"
    echo ""
    echo "Check logs in: logs/scriptguard-test.log"
    echo ""
    exit 1
fi

echo ""
echo "================================================"
echo "SUCCESS: Pipeline completed successfully!"
echo "================================================"
echo ""
echo "Model saved to: test_models/scriptguard-test/"
echo "Logs saved to: logs/scriptguard-test.log"
echo ""
echo "To test with production config, run:"
echo "  python -m scriptguard.pipelines.training_pipeline"
echo ""
