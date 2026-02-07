# ScriptGuard Deployment Guide

Complete guide for deploying ScriptGuard in production with Docker.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Configuration](#service-configuration)
- [Production Deployment](#production-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

ScriptGuard v2.1 uses a microservices architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (Optional)                      │
│                  Reverse Proxy / SSL                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ScriptGuard API Service                     │
│           (FastAPI + Uvicorn + CUDA)                     │
└──────┬─────────────────────────┬───────────────────┬────┘
       │                         │                   │
       ▼                         ▼                   ▼
┌────────────┐          ┌─────────────┐     ┌──────────────┐
│ PostgreSQL │          │   Qdrant    │     │   Models     │
│  Database  │          │ Vector DB   │     │   Storage    │
└────────────┘          └─────────────┘     └──────────────┘
```

### Services

1. **PostgreSQL** - Primary data storage for code samples
2. **Qdrant** - Vector database for RAG (CVE/malware knowledge)
3. **Training Service** - Model training (runs once)
4. **API Service** - Inference endpoint (always running)
5. **Nginx** (optional) - Reverse proxy and SSL termination
6. **Prometheus/Grafana** (optional) - Monitoring stack

## Prerequisites

### System Requirements
- **CPU**: 8+ cores
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for training and inference)
- **Storage**: 500GB+ SSD
- **OS**: Linux (Ubuntu 22.04+ recommended)

### Software Requirements
```bash
# Docker
docker --version  # 24.0.0+

# Docker Compose
docker-compose --version  # 2.20.0+

# NVIDIA Container Toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
```

### 2. Configure Environment
```bash
cp .env.example .env
nano .env
```

Edit `.env`:
```env
# PostgreSQL
POSTGRES_PASSWORD=your_secure_password_here

# API Keys
GITHUB_API_TOKEN=ghp_your_token
HUGGINGFACE_TOKEN=hf_your_token
WANDB_API_KEY=your_wandb_key

# Model
BASE_MODEL_ID=bigcode/starcoder2-3b
DEVICE=cuda

# API
API_PORT=8000
LOG_LEVEL=INFO
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scriptguard
POSTGRES_USER=scriptguard
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Services
```bash
# Start database and vector store
docker-compose -f docker/docker-compose.prod.yml up -d postgres qdrant

# Wait for services to be healthy
docker-compose -f docker/docker-compose.prod.yml ps

# Initialize database schema
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  psql -U scriptguard -d scriptguard -f /docker-entrypoint-initdb.d/init-db.sql

# Train model (one-time)
docker-compose -f docker/docker-compose.prod.yml run --rm training

# Start API
docker-compose -f docker/docker-compose.prod.yml up -d api
```

### 4. Test API
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello World\")"}'
```

## Service Configuration

### PostgreSQL Configuration

**Connection String:**
```
postgresql://scriptguard:password@localhost:5432/scriptguard
```

**Custom Configuration:**
Edit `docker/postgresql.conf`:
```ini
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Qdrant Configuration

**Storage Path:** `/qdrant/storage` (volume)

**Custom Configuration:**
```yaml
qdrant:
  service:
    http_port: 6333
    grpc_port: 6334
  storage:
    storage_path: /qdrant/storage
    wal_capacity_mb: 32
    wal_segments_ahead: 0
  collections:
    on_disk_payload: true
```

### API Configuration

**Environment Variables:**
```env
MODEL_PATH=/app/models/scriptguard-model
DEVICE=cuda
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=scriptguard
POSTGRES_USER=scriptguard
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

## Production Deployment

### Using Docker Compose

**Full Stack:**
```bash
docker-compose -f docker/docker-compose.prod.yml up -d
```

**With Nginx:**
```bash
docker-compose -f docker/docker-compose.prod.yml --profile with-nginx up -d
```

**With Monitoring:**
```bash
docker-compose -f docker/docker-compose.prod.yml --profile monitoring up -d
```

### Using Kubernetes

**Apply manifests:**
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/qdrant.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/ingress.yaml
```

### SSL/TLS Configuration

**Generate Self-Signed Certificate:**
```bash
mkdir -p docker/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/scriptguard.key \
  -out docker/ssl/scriptguard.crt \
  -subj "/CN=scriptguard.local"
```

**Nginx Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name scriptguard.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/scriptguard.crt;
    ssl_certificate_key /etc/nginx/ssl/scriptguard.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

### Health Checks

**API Health:**
```bash
curl http://localhost:8000/health
```

**Database Health:**
```bash
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  pg_isready -U scriptguard -d scriptguard
```

**Qdrant Health:**
```bash
curl http://localhost:6333/health
```

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

**Targets:**
- API: `http://api:8000/metrics`
- PostgreSQL: `postgres-exporter:9187`
- Qdrant: `http://qdrant:6333/metrics`

### Grafana Dashboards

Access Grafana at `http://localhost:3000`

**Default credentials:** `admin` / `admin` (change on first login)

**Import dashboards:**
- PostgreSQL Dashboard: ID 9628
- Qdrant Dashboard: Custom (provided in `docker/grafana/dashboards/`)

### Logs

**View logs:**
```bash
# All services
docker-compose -f docker/docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.prod.yml logs -f api

# Last 100 lines
docker-compose -f docker/docker-compose.prod.yml logs --tail=100 api
```

**Persistent logs:**
```yaml
volumes:
  - api_logs:/app/logs
```

## Backup & Recovery

### Database Backup

**Manual backup:**
```bash
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  pg_dump -U scriptguard scriptguard > backup_$(date +%Y%m%d).sql
```

**Automated backup:**
```bash
# Add to crontab
0 2 * * * docker-compose -f /path/to/docker-compose.prod.yml exec -T postgres \
  pg_dump -U scriptguard scriptguard | gzip > /backups/scriptguard_$(date +\%Y\%m\%d).sql.gz
```

**Restore:**
```bash
docker-compose -f docker/docker-compose.prod.yml exec -T postgres \
  psql -U scriptguard scriptguard < backup.sql
```

### Qdrant Backup

**Create snapshot:**
```bash
curl -X POST http://localhost:6333/collections/malware_knowledge/snapshots
```

**Download snapshot:**
```bash
curl http://localhost:6333/collections/malware_knowledge/snapshots/{snapshot_name} \
  --output qdrant_backup.snapshot
```

**Restore snapshot:**
```bash
curl -X PUT http://localhost:6333/collections/malware_knowledge/snapshots/upload \
  -F 'snapshot=@qdrant_backup.snapshot'
```

## Scaling

### Horizontal Scaling (API)

**Docker Compose:**
```bash
docker-compose -f docker/docker-compose.prod.yml up -d --scale api=3
```

**Nginx load balancing:**
```nginx
upstream scriptguard_api {
    least_conn;
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}
```

### Vertical Scaling

**Increase resources:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
```

## Troubleshooting

### Common Issues

**1. API not starting:**
```bash
# Check logs
docker-compose -f docker/docker-compose.prod.yml logs api

# Check GPU
nvidia-smi

# Restart service
docker-compose -f docker/docker-compose.prod.yml restart api
```

**2. Database connection failed:**
```bash
# Check PostgreSQL status
docker-compose -f docker/docker-compose.prod.yml ps postgres

# Check connection
docker-compose -f docker/docker-compose.prod.yml exec api \
  python -c "import psycopg2; psycopg2.connect('postgresql://scriptguard:password@postgres:5432/scriptguard')"
```

**3. Out of memory:**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8

# Or use CPU inference
environment:
  - DEVICE=cpu
```

**4. Slow inference:**
```bash
# Use quantized model
- MODEL_PATH=/app/models/scriptguard-4bit

# Enable caching
- CACHE_SIZE=1000
```

### Debug Mode

**Enable debug logs:**
```yaml
environment:
  - LOG_LEVEL=DEBUG
  - PYTHONUNBUFFERED=1
```

**Interactive debugging:**
```bash
docker-compose -f docker/docker-compose.prod.yml exec api /bin/bash
```

## Security

### Best Practices

1. **Change default passwords**
2. **Use secrets management** (Docker Swarm secrets, Kubernetes secrets)
3. **Enable SSL/TLS**
4. **Restrict network access** (firewall rules)
5. **Regular updates** (security patches)
6. **Audit logs** (log analysis)
7. **Rate limiting** (API rate limits)
8. **Authentication** (API keys, OAuth)

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

## Maintenance

### Regular Tasks

**Daily:**
- Check service health
- Monitor disk usage
- Review error logs

**Weekly:**
- Database vacuum
- Update statistics
- Backup verification

**Monthly:**
- Security updates
- Model retraining (if needed)
- Performance review

### Database Maintenance

```bash
# Vacuum
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  psql -U scriptguard -d scriptguard -c "VACUUM ANALYZE;"

# Reindex
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  psql -U scriptguard -d scriptguard -c "REINDEX DATABASE scriptguard;"
```

## Support

- **Documentation**: [docs/](../docs/)
- **GitHub Issues**: https://github.com/yourusername/ScriptGuard/issues
- **Email**: support@scriptguard.io
