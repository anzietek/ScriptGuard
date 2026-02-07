# Qdrant API Key Configuration Guide

## Overview

ScriptGuard now supports Qdrant API key authentication for secured deployments. The API key can be provided via:
1. Environment variable: `QDRANT_API_KEY` (recommended)
2. Configuration file: `config.yaml` (less secure)

## When to Use API Key

### Required:
- **Qdrant Cloud** (https://cloud.qdrant.io/)
- **Production VPS** with security enabled
- **Multi-tenant deployments**

### Not Required:
- **Local development** (Docker without auth)
- **Trusted internal networks**

## Configuration

### Method 1: Environment Variable (Recommended)

Add to your `.env` file:
```bash
QDRANT_HOST=your-vps-ip-or-domain
QDRANT_PORT=6333
QDRANT_API_KEY=your_secure_api_key_here
```

### Method 2: Config File (Less Secure)

Edit `config.yaml`:
```yaml
qdrant:
  host: "your-vps-ip-or-domain"
  port: 6333
  api_key: "your_secure_api_key_here"
  use_https: false  # Set to true for production
```

**⚠️ Warning**: Avoid committing API keys to version control!

## Enabling API Key on Qdrant Server

### Docker Compose (VPS)

Edit `docker-compose.vps.yml`:
```yaml
services:
  qdrant:
    environment:
      - QDRANT__SERVICE__API_KEY=your_secure_api_key_here
```

Or use environment file:
```bash
# .env.vps
QDRANT_API_KEY=your_secure_api_key_here
```

Then update `docker-compose.vps.yml`:
```yaml
services:
  qdrant:
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
```

### Standalone Qdrant

Start with API key:
```bash
docker run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=your_secure_api_key_here \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

## Testing Connection

### Test Script

```bash
cd C:\Users\anzie\workspace\ScriptGuard
python scripts/tests/test_qdrant_api_key.py
```

Expected output:
```
=== Qdrant API Key Authentication Test ===

Host: 62.171.130.236
Port: 6333
API Key: ***A1DA03

✓ Qdrant connected successfully. Collections: 2

Available collections:
  - malware_knowledge
  - code_samples
```

### Manual Test (Python)

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://your-vps-ip:6333",
    api_key="your_secure_api_key_here"
)

collections = client.get_collections()
print(f"Connected! Collections: {len(collections.collections)}")
```

### Manual Test (curl)

```bash
curl -H "api-key: your_secure_api_key_here" \
     http://your-vps-ip:6333/collections
```

## Code Changes

The following components now support `QDRANT_API_KEY`:

### 1. QdrantStore (CVE patterns)
```python
# File: src/scriptguard/rag/qdrant_store.py
from scriptguard.rag.qdrant_store import QdrantStore

store = QdrantStore()  # Automatically reads QDRANT_API_KEY from env
```

### 2. CodeSimilarityStore (Few-shot examples)
```python
# File: src/scriptguard/rag/code_similarity_store.py
from scriptguard.rag.code_similarity_store import CodeSimilarityStore

store = CodeSimilarityStore()  # Automatically reads QDRANT_API_KEY from env
```

### 3. Pipeline Scripts
All scripts that connect to Qdrant now automatically use `QDRANT_API_KEY`:
- `src/main.py`
- `scripts/bootstrap_qdrant.py`
- `scripts/enrich_qdrant_cve.py`
- `src/scriptguard/steps/qdrant_augmentation.py`

## Security Best Practices

### 1. Generate Strong API Keys

```bash
# Linux/macOS
openssl rand -hex 32

# Windows PowerShell
[System.Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

### 2. Use Environment Variables

Never hardcode API keys:
```bash
# ✓ Good
export QDRANT_API_KEY=$(cat /secure/qdrant.key)

# ✗ Bad
QDRANT_API_KEY="hardcoded_key_in_script"
```

### 3. Restrict Access

Use firewall rules:
```bash
# Allow only specific IPs
sudo ufw allow from YOUR_IP to any port 6333
sudo ufw allow from YOUR_IP to any port 6334
```

### 4. Enable HTTPS

For production deployments:
```yaml
qdrant:
  host: "your-domain.com"
  port: 443
  api_key: "${QDRANT_API_KEY}"
  use_https: true
```

### 5. Rotate Keys Regularly

```bash
# Generate new key
NEW_KEY=$(openssl rand -hex 32)

# Update Qdrant server
docker exec scriptguard-qdrant \
  sh -c "echo 'QDRANT__SERVICE__API_KEY=$NEW_KEY' >> /etc/qdrant/config.yaml"
docker restart scriptguard-qdrant

# Update client
echo "QDRANT_API_KEY=$NEW_KEY" >> .env
```

## Troubleshooting

### Error: "Unauthorized" or "401"

**Cause**: Missing or incorrect API key

**Solution**:
```bash
# Check if API key is set
echo $QDRANT_API_KEY

# Verify it matches server configuration
curl -H "api-key: $QDRANT_API_KEY" http://your-vps-ip:6333/collections
```

### Error: "Connection refused"

**Cause**: Firewall or network issue

**Solution**:
```bash
# Test connectivity
telnet your-vps-ip 6333

# Check firewall
sudo ufw status

# Allow Qdrant ports
sudo ufw allow 6333/tcp
sudo ufw allow 6334/tcp
```

### Error: "Certificate verification failed" (HTTPS)

**Cause**: Invalid SSL certificate

**Solution**:
```yaml
# Disable SSL verification (development only!)
qdrant:
  use_https: true
  verify_ssl: false
```

## Migration from Non-Authenticated Setup

### Step 1: Enable API Key on Server

```bash
# Edit docker-compose.vps.yml
nano docker-compose.vps.yml

# Add to qdrant service:
environment:
  - QDRANT__SERVICE__API_KEY=your_new_secure_key

# Restart
docker-compose -f docker-compose.vps.yml restart qdrant
```

### Step 2: Update Client Configuration

```bash
# Add to .env
echo "QDRANT_API_KEY=your_new_secure_key" >> .env
```

### Step 3: Test Connection

```bash
python scripts/tests/test_qdrant_api_key.py
```

### Step 4: Verify All Scripts Work

```bash
# Test bootstrap
python scripts/bootstrap_qdrant.py

# Test enrichment
python scripts/enrich_qdrant_cve.py

# Test training pipeline
python src/main.py
```

## Support

For issues or questions:
1. Check logs: `docker logs scriptguard-qdrant`
2. Review documentation: `docs/QDRANT_SETUP.md`
3. Test connection: `scripts/tests/test_qdrant_api_key.py`
