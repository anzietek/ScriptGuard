# Test both models using EXISTING collections (no re-vectorization)
#
# Prerequisites:
# 1. Collections must exist:
#    - code_samples_balanced (UniXcoder)
#    - code_samples_jina_v3 (Jina-v3)
#
# Usage:
#    powershell scripts/test_models_existing.ps1

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Testing Models with Existing Collections" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

# Test UniXcoder
Write-Host ""
Write-Host "[1/2] Testing UniXcoder (code_samples_balanced)..." -ForegroundColor Yellow
python scripts/test_hybrid_full_eval.py --use-existing

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] UniXcoder test completed" -ForegroundColor Green
} else {
    Write-Host "[ERROR] UniXcoder test failed" -ForegroundColor Red
    exit 1
}

# Test Jina-v3
Write-Host ""
Write-Host "[2/2] Testing Jina-v3 (code_samples_jina_v3)..." -ForegroundColor Yellow
python scripts/test_jina_v3_comparison.py --use-existing

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Jina-v3 test completed" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Jina-v3 test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "All tests completed successfully!" -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan
