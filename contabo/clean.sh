#!/bin/bash

# ScriptGuard Data Cleanup Script
# WARNING: This script deletes ALL database data!

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}=== DANGER ZONE: DATA CLEANUP ===${NC}"
echo ""

# 1. Check root permissions
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run this script as root (sudo)."
    exit 1
fi

# 2. Confirm action
echo -e "${YELLOW}WARNING: This will perform the following actions:${NC}"
echo "  1. Stop and remove all ScriptGuard containers."
echo "  2. PERMANENTLY DELETE all data in Postgres and Qdrant."
echo "  3. Remove the data directory."
echo ""
read -p "Are you sure you want to proceed? (Type 'y' to confirm): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

# 3. Load variables to find DATA_PATH
if [ -f ".env" ]; then
    source .env
else
    echo "Notice: .env file not found. Using default paths."
fi

# Set default if not in .env
DATA_PATH="${DATA_PATH:-/var/lib/scriptguard}"

# 4. Stop Docker containers
echo -e "${YELLOW}Stopping Docker containers...${NC}"
if command -v docker &> /dev/null; then
    docker compose down -v
else
    echo "Docker not found, skipping container stop."
fi

# 5. Remove Data Directory
echo -e "${YELLOW}Removing data from: ${DATA_PATH}...${NC}"

if [ -d "$DATA_PATH" ]; then
    # Safety check: Ensure we are not deleting root
    if [ "$DATA_PATH" == "/" ]; then
        echo -e "${RED}ERROR: DATA_PATH is set to root (/). Aborting to prevent system destruction.${NC}"
        exit 1
    fi
    
    rm -rf "$DATA_PATH"
    echo -e "${GREEN}Data directory deleted.${NC}"
else
    echo "Directory $DATA_PATH does not exist. Nothing to delete."
fi

echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo "System is clean. You can run ./setup.sh again to start fresh."

