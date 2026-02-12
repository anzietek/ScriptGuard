#!/bin/bash
set -e

echo "=== ScriptGuard VPS Setup (SECURE MODE) ==="

# 1. Check for root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Run as root (use sudo)."
    exit 1
fi

# 2. Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# 3. Create data directories
DATA_PATH="${DATA_PATH:-/var/lib/scriptguard}"
mkdir -p "${DATA_PATH}/postgres" "${DATA_PATH}/qdrant" "${DATA_PATH}/qdrant-snapshots" "${DATA_PATH}/zenml"
chmod -R 777 "${DATA_PATH}"

# 4. Configure firewall (LOCK DOWN PORTS)
echo "Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw --force enable
    # Remove old rules that opened ports
    ufw delete allow 5432/tcp 2>/dev/null || true
    ufw delete allow 6333/tcp 2>/dev/null || true
    ufw delete allow 6334/tcp 2>/dev/null || true
    ufw delete allow 5050/tcp 2>/dev/null || true
    ufw delete allow 8237/tcp 2>/dev/null || true

    # Only SSH stays open
    ufw allow 22/tcp
    echo "Firewall configured. Only port 22 (SSH) is open."
fi

# 5. Restart services (with localhost binding)
echo "Restarting containers..."
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up -d

echo ""
echo "=== SECURED ==="
echo "Services are accessible ONLY from inside the server (127.0.0.1)."
echo ""
echo "TO CONNECT FROM YOUR COMPUTER, USE THIS SSH TUNNEL:"
echo "---------------------------------------------------"
echo "ssh -L 5432:127.0.0.1:5432 -L 6333:127.0.0.1:6333 -L 5050:127.0.0.1:5050 -L 8237:127.0.0.1:8237 deployer@YOUR_IP"
echo "---------------------------------------------------"
echo "Access:"
echo " Postgres: localhost:5432"
echo " Qdrant:   http://localhost:6333"
echo " PgAdmin:  http://localhost:5050"
echo " ZenML:    http://localhost:8237 (if started with --profile with-zenml)"
echo ""
echo "To start with ZenML:"
echo "  docker compose --profile with-zenml up -d"
