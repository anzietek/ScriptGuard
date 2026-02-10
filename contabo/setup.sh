#!/bin/bash
set -e

echo "=== ScriptGuard VPS Setup (SECURE MODE) ==="

# 1. Sprawdzenie roota
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Uruchom jako root (sudo)."
    exit 1
fi

# 2. Instalacja Dockera
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# 3. Katalogi danych
DATA_PATH="${DATA_PATH:-/var/lib/scriptguard}"
mkdir -p "${DATA_PATH}/postgres" "${DATA_PATH}/qdrant" "${DATA_PATH}/qdrant-snapshots"
chmod -R 777 "${DATA_PATH}"

# 4. Firewall (ZAMYKANIE PORTÓW)
echo "Blokowanie firewalla..."
if command -v ufw &> /dev/null; then
    ufw --force enable
    # Usuwamy stare reguły otwierające porty
    ufw delete allow 5432/tcp 2>/dev/null || true
    ufw delete allow 6333/tcp 2>/dev/null || true
    ufw delete allow 6334/tcp 2>/dev/null || true
    ufw delete allow 5050/tcp 2>/dev/null || true
    
    # Tylko SSH zostaje otwarte
    ufw allow 22/tcp
    echo "Firewall zablokowany. Tylko port 22 (SSH) jest otwarty."
fi

# 5. Restart usług (z bindowaniem do localhost)
echo "Restartowanie kontenerów..."
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up -d

echo ""
echo "=== ZABEZPIECZONO ==="
echo "Usługi są dostępne TYLKO z wewnątrz serwera (127.0.0.1)."
echo ""
echo "ABY SIĘ POŁĄCZYĆ Z KOMPUTERA, UŻYJ TEGO TUNELU:"
echo "---------------------------------------------------"
echo "ssh -L 5432:127.0.0.1:5432 -L 6333:127.0.0.1:6333 -L 8080:127.0.0.1:5050 deployer@TWOJE_IP"
echo "---------------------------------------------------"
echo "Dostęp:"
echo " Postgres: localhost:5432"
echo " Qdrant:   http://localhost:6333"
echo " PgAdmin:  http://localhost:8080"
