# Server Settings
$ServerIP = "62.171.130.236"
$User = "deployer"
# CHANGE THIS to the path of your private key (e.g., "C:\Users\You\.ssh\id_rsa" or ".\my-key.pem")
$KeyPath = "$HOME\.ssh\id_ed25519"

# Clear screen and display info
Clear-Host
Write-Host "=== ScriptGuard Secure Tunnel ===" -ForegroundColor Cyan
Write-Host "Target: $User@$ServerIP"
Write-Host "Auth:   Private Key ($KeyPath)"
Write-Host ""
Write-Host "Opening local ports:"
Write-Host " [Postgres] 127.0.0.1:5432 -> Server:5432" -ForegroundColor Green
Write-Host " [Qdrant]   127.0.0.1:6333 -> Server:6333" -ForegroundColor Green
Write-Host " [PgAdmin]  127.0.0.1:5050 -> Server:5050" -ForegroundColor Green
Write-Host " [ZenML]    127.0.0.1:8237 -> Server:8237" -ForegroundColor Green
Write-Host ""
Write-Host "WARNING: Do not close this window. Minimize it." -ForegroundColor Yellow
Write-Host ""

# Check if key exists
if (-Not (Test-Path $KeyPath)) {
    Write-Host "Error: Private key file not found at: $KeyPath" -ForegroundColor Red
    Write-Host "Please update the `$KeyPath variable in the script."
    Pause
    exit
}

# Start SSH with Key
# -i specifies the identity file (key)
ssh -i $KeyPath -N -L 5432:127.0.0.1:5432 -L 6333:127.0.0.1:6333 -L 5050:127.0.0.1:5050 -L 8237:127.0.0.1:8237 $User@$ServerIP

# Prevent window from closing immediately on error
Write-Host ""
Write-Host "Connection closed." -ForegroundColor Red
Pause