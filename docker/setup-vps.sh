#!/bin/bash

# ScriptGuard VPS Setup Script
# This script prepares VPS environment and deploys PostgreSQL and Qdrant

set -e

echo "=== ScriptGuard VPS Setup ==="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl enable docker
    systemctl start docker
    rm get-docker.sh
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${GREEN}Docker is already installed${NC}"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}Docker Compose not found. Installing Docker Compose...${NC}"
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}Docker Compose is already installed${NC}"
fi

# Create data directories
DATA_PATH="${DATA_PATH:-/var/lib/scriptguard}"
echo -e "${YELLOW}Creating data directories at ${DATA_PATH}...${NC}"
mkdir -p "${DATA_PATH}/postgres"
mkdir -p "${DATA_PATH}/qdrant"
mkdir -p "${DATA_PATH}/qdrant-snapshots"
chmod -R 755 "${DATA_PATH}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env file not found. Creating from template...${NC}"
    if [ -f ".env.vps.example" ]; then
        cp .env.vps.example .env
        echo -e "${RED}IMPORTANT: Edit .env file and set secure passwords!${NC}"
        echo -e "${YELLOW}Run: nano .env${NC}"
        exit 1
    else
        echo -e "${RED}Template file .env.vps.example not found!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}.env file found${NC}"
fi

# Verify environment variables
source .env

if [ "$POSTGRES_PASSWORD" = "CHANGE_ME_SECURE_PASSWORD" ]; then
    echo -e "${RED}ERROR: Please change POSTGRES_PASSWORD in .env file!${NC}"
    exit 1
fi

if [ "$PGADMIN_PASSWORD" = "CHANGE_ME_ADMIN_PASSWORD" ]; then
    echo -e "${YELLOW}WARNING: Please change PGADMIN_PASSWORD in .env file if you plan to use PgAdmin${NC}"
fi

# Configure firewall (ufw)
echo -e "${YELLOW}Configuring firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw --force enable
    ufw allow 22/tcp  # SSH
    ufw allow 5432/tcp  # PostgreSQL
    ufw allow 6333/tcp  # Qdrant HTTP
    ufw allow 6334/tcp  # Qdrant gRPC
    echo -e "${GREEN}Firewall configured${NC}"
else
    echo -e "${YELLOW}ufw not found, skipping firewall configuration${NC}"
fi

# Pull Docker images
echo -e "${YELLOW}Pulling Docker images...${NC}"
docker-compose -f docker-compose.vps.yml pull

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose -f docker-compose.vps.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check service status
echo ""
echo "=== Service Status ==="
docker-compose -f docker-compose.vps.yml ps

# Display connection information
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "PostgreSQL:"
echo "  Host: localhost (or your VPS IP)"
echo "  Port: ${POSTGRES_PORT}"
echo "  Database: ${POSTGRES_DB}"
echo "  User: ${POSTGRES_USER}"
echo "  Password: (set in .env)"
echo ""
echo "Qdrant:"
echo "  HTTP API: http://localhost:${QDRANT_HTTP_PORT}"
echo "  gRPC API: localhost:${QDRANT_GRPC_PORT}"
echo "  Dashboard: http://localhost:${QDRANT_HTTP_PORT}/dashboard"
echo ""
echo "PgAdmin (optional):"
echo "  URL: http://localhost:${PGADMIN_PORT}"
echo "  Start with: docker-compose -f docker-compose.vps.yml --profile admin up -d"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose -f docker-compose.vps.yml logs -f"
echo "  Stop services: docker-compose -f docker-compose.vps.yml down"
echo "  Restart: docker-compose -f docker-compose.vps.yml restart"
echo "  Backup PostgreSQL: docker exec scriptguard-postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > backup.sql"
echo "  Backup Qdrant: docker exec scriptguard-qdrant wget -O- http://localhost:6333/collections/scriptguard_collection/snapshots"
echo ""
echo -e "${YELLOW}Remember to secure your VPS:${NC}"
echo "  1. Change SSH port"
echo "  2. Use SSH keys instead of passwords"
echo "  3. Set up fail2ban"
echo "  4. Regular security updates: apt update && apt upgrade"
echo "  5. Consider using a reverse proxy (nginx) with SSL/TLS"
echo ""
