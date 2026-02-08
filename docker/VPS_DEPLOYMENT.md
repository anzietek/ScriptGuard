# ScriptGuard VPS Deployment Guide

## Quick Start

### 1. Prerequisites

Ensure your VPS has:
- Ubuntu 20.04+ or Debian 11+
- At least 4GB RAM
- 20GB free disk space
- Root or sudo access

### 2. Upload Files to VPS

```bash
# On your local machine
scp docker/docker-compose.vps.yml root@YOUR_VPS_IP:/root/scriptguard/
scp docker/.env.vps.example root@YOUR_VPS_IP:/root/scriptguard/
scp docker/setup-vps.sh root@YOUR_VPS_IP:/root/scriptguard/
scp docker/init-db.sql root@YOUR_VPS_IP:/root/scriptguard/
```

### 3. Connect to VPS and Run Setup

```bash
ssh root@YOUR_VPS_IP
cd /root/scriptguard

# Make setup script executable
chmod +x setup-vps.sh

# Copy and edit environment file
cp .env.vps.example .env
nano .env  # Set secure passwords!

# Run setup
./setup-vps.sh
```

## Manual Setup (Alternative)

### 1. Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. Install Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Create Data Directories

```bash
sudo mkdir -p /var/lib/scriptguard/postgres
sudo mkdir -p /var/lib/scriptguard/qdrant
sudo mkdir -p /var/lib/scriptguard/qdrant-snapshots
```

### 4. Configure Environment

```bash
cp .env.vps.example .env
nano .env
```

**Important**: Change the following in `.env`:
- `POSTGRES_PASSWORD` - Set a strong password
- `PGADMIN_PASSWORD` - Set admin password (if using PgAdmin)

### 5. Start Services

```bash
docker-compose -f docker-compose.vps.yml up -d
```

### 6. Verify Services

```bash
docker-compose -f docker-compose.vps.yml ps
docker-compose -f docker-compose.vps.yml logs -f
```

## Service Access

### PostgreSQL
- **Host**: `YOUR_VPS_IP`
- **Port**: `5432`
- **Database**: `scriptguard`
- **User**: `scriptguard`
- **Password**: (from `.env`)

Connection string:
```
postgresql://scriptguard:YOUR_PASSWORD@YOUR_VPS_IP:5432/scriptguard
```

### Qdrant
- **HTTP API**: `http://YOUR_VPS_IP:6333`
- **gRPC API**: `YOUR_VPS_IP:6334`
- **Dashboard**: `http://YOUR_VPS_IP:6333/dashboard`

### PgAdmin (Optional)

To enable PgAdmin:
```bash
docker-compose -f docker-compose.vps.yml --profile admin up -d
```

Access at: `http://YOUR_VPS_IP:5050`

## Security Recommendations

### 1. Firewall Configuration

```bash
# Allow SSH (change port if needed)
sudo ufw allow 22/tcp

# Allow PostgreSQL (restrict to specific IPs if possible)
sudo ufw allow from YOUR_IP to any port 5432

# Allow Qdrant (restrict to specific IPs if possible)
sudo ufw allow from YOUR_IP to any port 6333
sudo ufw allow from YOUR_IP to any port 6334

# Enable firewall
sudo ufw enable
```

### 2. Use SSH Keys

```bash
# On local machine, generate key if needed
ssh-keygen -t ed25519

# Copy to VPS
ssh-copy-id root@YOUR_VPS_IP
```

### 3. Disable Password Authentication

```bash
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

### 4. Install Fail2Ban

```bash
sudo apt update
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 5. Regular Updates

```bash
sudo apt update && sudo apt upgrade -y
```

## Management Commands

### View Logs

```bash
# All services
docker-compose -f docker-compose.vps.yml logs -f

# Specific service
docker-compose -f docker-compose.vps.yml logs -f postgres
docker-compose -f docker-compose.vps.yml logs -f qdrant
```

### Stop Services

```bash
docker-compose -f docker-compose.vps.yml down
```

### Restart Services

```bash
docker-compose -f docker-compose.vps.yml restart
```

### Update Images

```bash
docker-compose -f docker-compose.vps.yml pull
docker-compose -f docker-compose.vps.yml up -d
```

## Backup & Restore

### PostgreSQL Backup

```bash
# Create backup
docker exec scriptguard-postgres pg_dump -U scriptguard scriptguard > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i scriptguard-postgres psql -U scriptguard scriptguard < backup_20260207.sql
```

### Qdrant Backup

```bash
# Create snapshot
curl -X POST "http://localhost:6333/collections/scriptguard_collection/snapshots"

# List snapshots
curl "http://localhost:6333/collections/scriptguard_collection/snapshots"

# Download snapshot
curl "http://localhost:6333/collections/scriptguard_collection/snapshots/SNAPSHOT_NAME" --output qdrant_backup.snapshot
```

### Full Data Backup

```bash
# Stop services
docker-compose -f docker-compose.vps.yml down

# Backup data directory
tar -czf scriptguard_backup_$(date +%Y%m%d).tar.gz /var/lib/scriptguard/

# Start services
docker-compose -f docker-compose.vps.yml up -d
```

## Monitoring

### Check Resource Usage

```bash
# Container stats
docker stats

# Disk usage
df -h /var/lib/scriptguard/

# Docker disk usage
docker system df
```

### Health Checks

```bash
# PostgreSQL health
docker exec scriptguard-postgres pg_isready -U scriptguard

# Qdrant health
curl http://localhost:6333/readyz
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose.vps.yml logs

# Check Docker status
sudo systemctl status docker

# Check disk space
df -h
```

### Connection Issues

```bash
# Check if ports are open
sudo netstat -tulpn | grep -E '5432|6333|6334'

# Test PostgreSQL connection
docker exec scriptguard-postgres psql -U scriptguard -d scriptguard -c "SELECT version();"

# Test Qdrant connection
curl http://localhost:6333/collections
```

### Reset Everything

```bash
# WARNING: This will delete all data!
docker-compose -f docker-compose.vps.yml down -v
rm -rf /var/lib/scriptguard/*
docker-compose -f docker-compose.vps.yml up -d
```

## Performance Tuning

### PostgreSQL Configuration

Edit `docker-compose.vps.yml` and add to postgres environment:
```yaml
- POSTGRES_SHARED_BUFFERS=256MB
- POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
- POSTGRES_MAX_CONNECTIONS=100
```

### Qdrant Configuration

Edit `docker-compose.vps.yml` and add to qdrant environment:
```yaml
- QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64
- QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4
```

## Contact & Support

For issues or questions, refer to the main project documentation or create an issue on GitHub.
