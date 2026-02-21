# Quick test for hello world false positive fix
# Usage: .\test_hello_world_fix.ps1

$API_URL = $env:SCRIPTGUARD_API_URL
if (-not $API_URL) {
    $API_URL = "http://localhost:8000"
}

$API_KEY = $env:SCRIPTGUARD_API_KEY
if (-not $API_KEY) {
    $API_KEY = "your_key_here"
}

Write-Host "=" * 80
Write-Host "ScriptGuard False Positive Fix - Quick Test"
Write-Host "=" * 80
Write-Host ""
Write-Host "API URL: $API_URL"
Write-Host "Testing code: print('hello world')"
Write-Host ""

# Test payload
$body = @{
    script_content = "print('hello world')"
    include_rag = $true
} | ConvertTo-Json

# Headers
$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = $API_KEY
}

try {
    # Make request
    $response = Invoke-RestMethod -Uri "$API_URL/analyze" -Method Post -Headers $headers -Body $body -TimeoutSec 30

    Write-Host "Result:"
    Write-Host "  Is Malicious: $($response.is_malicious)"
    Write-Host "  Confidence: $($response.confidence)"
    Write-Host ""

    if ($response.is_malicious -eq $false) {
        Write-Host "✅ SUCCESS: Correctly classified as BENIGN" -ForegroundColor Green
    } else {
        Write-Host "❌ FAILED: Incorrectly classified as MALICIOUS" -ForegroundColor Red
        Write-Host "   This indicates the false positive issue is NOT fixed." -ForegroundColor Red
    }

    # Show RAG context if available
    if ($response.related_cves -and $response.related_cves.Count -gt 0) {
        Write-Host ""
        Write-Host "RAG Context ($($response.related_cves.Count) examples):"
        foreach ($cve in $response.related_cves) {
            $desc = $cve.description.Substring(0, [Math]::Min(60, $cve.description.Length))
            Write-Host "  - $desc... (score: $($cve.score))"
        }
    }

} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure the API is running:" -ForegroundColor Yellow
    Write-Host "  python -m scriptguard.api.main" -ForegroundColor Yellow
}

Write-Host ""
