#!/bin/bash
set -e

# ScriptGuard Docker Volumes Setup Script
# Creates all required host directories for bind mounts

echo "🔧 Setting up ScriptGuard Docker volumes..."

# Read DATA_PATH from .env or use default
if [ -f .env ]; then
    export $(grep "^DATA_PATH=" .env | xargs)
fi

DATA_PATH="${DATA_PATH:-/var/lib/scriptguard}"

echo "📁 Creating directories in: $DATA_PATH"

# Create all required directories
sudo mkdir -p "$DATA_PATH"/{postgres,qdrant,qdrant-snapshots,zenml}

# Set permissions
# Option 1: If running Docker as current user
echo "🔐 Setting ownership to current user: $USER"
sudo chown -R $USER:$USER "$DATA_PATH"
sudo chmod -R 755 "$DATA_PATH"

# Option 2: If running Docker as root (comment out option 1 and uncomment below)
# echo "🔐 Setting permissions for Docker (root)"
# sudo chmod -R 755 "$DATA_PATH"

echo "✅ Volumes setup complete!"
echo ""
echo "Directory structure:"
tree -L 2 "$DATA_PATH" 2>/dev/null || ls -la "$DATA_PATH"
echo ""
echo "You can now run: docker-compose up -d"
