#!/bin/bash

# ScriptGuard VPS Backup Script
# Run this script periodically to backup PostgreSQL and Qdrant data

set -e

BACKUP_DIR="${BACKUP_DIR:-/var/backups/scriptguard}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DATE=$(date +%Y%m%d_%H%M%S)

echo "=== ScriptGuard Backup ==="
echo "Date: $(date)"
echo "Backup directory: ${BACKUP_DIR}"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
docker exec scriptguard-postgres pg_dump -U scriptguard scriptguard | gzip > "${BACKUP_DIR}/postgres_${DATE}.sql.gz"
echo "PostgreSQL backup: ${BACKUP_DIR}/postgres_${DATE}.sql.gz"

# Backup Qdrant snapshots
echo "Backing up Qdrant..."
SNAPSHOT_RESPONSE=$(curl -s -X POST "http://localhost:6333/collections/scriptguard_collection/snapshots" 2>/dev/null || echo "")

if [ -n "$SNAPSHOT_RESPONSE" ]; then
    SNAPSHOT_NAME=$(echo "$SNAPSHOT_RESPONSE" | grep -oP '"name":"\K[^"]+' || echo "")
    if [ -n "$SNAPSHOT_NAME" ]; then
        curl -s "http://localhost:6333/collections/scriptguard_collection/snapshots/${SNAPSHOT_NAME}" --output "${BACKUP_DIR}/qdrant_${DATE}.snapshot"
        echo "Qdrant backup: ${BACKUP_DIR}/qdrant_${DATE}.snapshot"
    else
        echo "Warning: Could not create Qdrant snapshot"
    fi
else
    echo "Warning: Qdrant API not responding"
fi

# Backup configuration files
echo "Backing up configuration..."
tar -czf "${BACKUP_DIR}/config_${DATE}.tar.gz" -C /root/scriptguard .env docker-compose.vps.yml 2>/dev/null || true

# Remove old backups
echo "Removing backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -type f -mtime +${RETENTION_DAYS} -delete

# List backups
echo ""
echo "Current backups:"
ls -lh "${BACKUP_DIR}"

# Calculate total size
TOTAL_SIZE=$(du -sh "${BACKUP_DIR}" | cut -f1)
echo ""
echo "Total backup size: ${TOTAL_SIZE}"
echo ""
echo "=== Backup Complete ==="
