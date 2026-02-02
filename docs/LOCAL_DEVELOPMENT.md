# ScriptGuard Local Development Guide

Complete guide for setting up and running ScriptGuard locally with infrastructure in Docker.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Manual Setup](#manual-setup)
- [Running the Application](#running-the-application)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [IDE Configuration](#ide-configuration)

---

## Overview

The local development setup separates **infrastructure** (PostgreSQL, Qdrant) from **application code**:

```
┌─────────────────────────────────────────────────────┐
│  Your IDE/CLI (Code runs here)                      │
│  - Python code in ./src                             │
│  - Edit, debug, hot-reload                          │
└────────────────┬────────────────────────────────────┘
                 │ Connects to ↓
┌────────────────┴────────────────────────────────────┐
│  Docker Infrastructure (docker-compose.dev.yml)     │
│  - PostgreSQL (localhost:5432)                      │
│  - Qdrant (localhost:6333)                          │
│  - Optional: pgAdmin, Prometheus, Grafana           │
└─────────────────────────────────────────────────────┘
```

**Benefits:**
✅ Fast code changes (no Docker rebuild)
✅ Easy debugging in IDE
✅ Hot reload with `--reload`
✅ Native Python performance
✅ Isolated infrastructure

---

## Quick Start

### Prerequisites

**Required:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux)
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

**Optional:**
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (for GPU training)
- [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/)

### One-Command Setup

**Linux/Mac:**
```bash
chmod +x dev-setup.sh
./dev-setup.sh
```

**Windows:**
```cmd
dev-setup.bat
```

The script will:
1. ✅ Check Docker installation
2. ✅ Create `.env` from `.env.dev`
3. ✅ Start PostgreSQL and Qdrant
4. ✅ Wait for services to be healthy
5. ✅ Initialize database schema
6. ✅ (Optional) Setup Python venv and install dependencies
7. ✅ (Optional) Bootstrap Qdrant with CVE data

**That's it!** Skip to [Running the Application](#running-the-application)

---

## Manual Setup

If you prefer manual setup or the script doesn't work:

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
```

### Step 2: Configure Environment

```bash
# Copy development environment template
cp .env.dev .env

# Edit .env and add your API keys (optional)
nano .env  # or use any text editor
```

**Minimal `.env` for local dev:**
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scriptguard
POSTGRES_USER=scriptguard
POSTGRES_PASSWORD=scriptguard

QDRANT_HOST=localhost
QDRANT_PORT=6333

DEVICE=cuda  # or cpu
LOG_LEVEL=DEBUG
```

### Step 3: Start Infrastructure

```bash
cd docker
docker-compose -f docker-compose.dev.yml up -d postgres qdrant
cd ..
```

**Check services are running:**
```bash
docker-compose -f docker/docker-compose.dev.yml ps
```

Expected output:
```
NAME                          STATUS    PORTS
scriptguard-postgres-dev     Up (healthy)  0.0.0.0:5432->5432/tcp
scriptguard-qdrant-dev       Up (healthy)  0.0.0.0:6333->6333/tcp
```

### Step 4: Initialize Database

Database is automatically initialized on first start via `init-db.sql`.

**Verify tables exist:**
```bash
docker exec -it scriptguard-postgres-dev psql -U scriptguard -d scriptguard -c "\dt"
```

Should show:
```
 public | samples          | table | scriptguard
 public | dataset_versions | table | scriptguard
```

### Step 5: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install ScriptGuard
pip install -e .
```

### Step 6: Bootstrap Qdrant

```bash
# Activate venv first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Bootstrap CVE data
python -c "
from scriptguard.rag import QdrantStore, bootstrap_cve_data
store = QdrantStore()
bootstrap_cve_data(store)
print('CVE data loaded successfully')
"
```

**Verify Qdrant:**
Open http://localhost:6333/dashboard in browser

---

## Running the Application

### Training Pipeline

```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate

# Run training
python src/main.py
```

**Custom config:**
```bash
# Edit config.yaml first, then:
python src/main.py
```

### Inference API

```bash
# Activate venv
source venv/bin/activate

# Run API with hot reload (development)
uvicorn scriptguard.api.main:app --reload --host 0.0.0.0 --port 8000

# Or run API without reload (production-like)
uvicorn scriptguard.api.main:app --host 0.0.0.0 --port 8000
```

**Test API:**
```bash
# Health check
curl http://localhost:8000/health

# Analyze script
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "import os; os.system(\"ls\")"}'
```

### Interactive Python Shell

```bash
source venv/bin/activate
python
```

```python
>>> from scriptguard.database import DatasetManager
>>> from scriptguard.rag import QdrantStore
>>>
>>> # Test database
>>> db = DatasetManager()
>>> stats = db.get_dataset_stats()
>>> print(stats)
>>>
>>> # Test Qdrant
>>> store = QdrantStore()
>>> results = store.search("remote code execution", limit=3)
>>> print(results)
```

---

## Development Workflow

### Typical Development Day

1. **Start infrastructure** (if not running):
   ```bash
   cd docker && docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Activate Python environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Make code changes** in `src/scriptguard/`

4. **Run/Test:**
   - **Training:** `python src/main.py`
   - **API:** `uvicorn scriptguard.api.main:app --reload`
   - **Tests:** `pytest tests/`

5. **Check logs:**
   ```bash
   # Infrastructure logs
   docker-compose -f docker/docker-compose.dev.yml logs -f

   # PostgreSQL only
   docker logs -f scriptguard-postgres-dev

   # Qdrant only
   docker logs -f scriptguard-qdrant-dev
   ```

6. **Stop infrastructure** (when done):
   ```bash
   cd docker && docker-compose -f docker-compose.dev.yml down
   ```

### Hot Reload Development

**API with hot reload:**
```bash
uvicorn scriptguard.api.main:app --reload
```

Changes to Python files will automatically reload the server!

**Alternative with auto-reload on any file change:**
```bash
pip install watchdog
watchmedo auto-restart --pattern="*.py" --recursive \
  -- python -m uvicorn scriptguard.api.inference:app
```

### Database Management

**Connect to PostgreSQL:**
```bash
docker exec -it scriptguard-postgres-dev psql -U scriptguard -d scriptguard
```

**Common SQL queries:**
```sql
-- Count samples
SELECT COUNT(*) FROM samples;

-- Check balance
SELECT label, COUNT(*) FROM samples GROUP BY label;

-- View recent samples
SELECT id, label, source, created_at FROM samples ORDER BY created_at DESC LIMIT 10;

-- Check dataset versions
SELECT * FROM dataset_versions ORDER BY created_at DESC;

-- Refresh statistics
REFRESH MATERIALIZED VIEW sample_statistics;
SELECT * FROM sample_statistics;
```

**Backup database:**
```bash
docker exec scriptguard-postgres-dev pg_dump -U scriptguard scriptguard > backup.sql
```

**Restore database:**
```bash
docker exec -i scriptguard-postgres-dev psql -U scriptguard scriptguard < backup.sql
```

### Qdrant Management

**Open dashboard:**
http://localhost:6333/dashboard

**Python API:**
```python
from scriptguard.rag import QdrantStore

store = QdrantStore()

# Search
results = store.search("SQL injection", limit=5)

# Get collection info
info = store.get_collection_info()
print(f"Points: {info['points_count']}")

# Clear collection (careful!)
# store.clear_collection()
```

---

## Troubleshooting

### Issue: "Cannot connect to PostgreSQL"

**Solution 1:** Check if container is running
```bash
docker ps | grep postgres
```

**Solution 2:** Check PostgreSQL logs
```bash
docker logs scriptguard-postgres-dev
```

**Solution 3:** Restart PostgreSQL
```bash
docker restart scriptguard-postgres-dev
```

**Solution 4:** Check connection
```bash
docker exec scriptguard-postgres-dev pg_isready -U scriptguard
```

### Issue: "Cannot connect to Qdrant"

**Solution 1:** Check if running
```bash
curl http://localhost:6333/health
```

**Solution 2:** Check logs
```bash
docker logs scriptguard-qdrant-dev
```

**Solution 3:** Restart Qdrant
```bash
docker restart scriptguard-qdrant-dev
```

### Issue: "ModuleNotFoundError"

**Solution:** Make sure venv is activated and package is installed
```bash
source venv/bin/activate  # or venv\Scripts\activate
pip install -e .
```

### Issue: "Port already in use"

**Solution:** Kill process using the port

**Linux/Mac:**
```bash
# Find process
lsof -i :5432  # or :6333, :8000

# Kill process
kill -9 <PID>
```

**Windows:**
```cmd
# Find process
netstat -ano | findstr :5432

# Kill process
taskkill /PID <PID> /F
```

### Issue: "Database not initialized"

**Solution:** Run init script manually
```bash
docker exec -i scriptguard-postgres-dev psql -U scriptguard scriptguard < docker/init-db.sql
```

---

## IDE Configuration

### Visual Studio Code

**1. Install extensions:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Docker (ms-azuretools.vscode-docker)

**2. Select Python interpreter:**
- `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose `./venv/bin/python`

**3. Create `.vscode/launch.json`:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Training Pipeline",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/main.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "API Server",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "scriptguard.api.inference:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000"
            ],
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

**4. Create `.vscode/settings.json`:**
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

### PyCharm

**1. Configure Python Interpreter:**
- File → Settings → Project → Python Interpreter
- Add → Virtualenv Environment → Existing
- Select `./venv/bin/python`

**2. Configure Run Configurations:**

**Training:**
- Run → Edit Configurations → + → Python
- Name: Training Pipeline
- Script path: `src/main.py`
- Environment variables: Load from `.env`

**API:**
- Run → Edit Configurations → + → Python
- Name: API Server
- Module name: `uvicorn`
- Parameters: `scriptguard.api.inference:app --reload`

**3. Configure Database:**
- View → Tool Windows → Database
- + → Data Source → PostgreSQL
- Host: localhost, Port: 5432
- Database: scriptguard, User: scriptguard, Password: scriptguard

---

## Optional Services

### pgAdmin (Database UI)

```bash
cd docker
docker-compose -f docker-compose.dev.yml --profile with-pgadmin up -d
```

**Access:** http://localhost:5050
**Login:** admin@scriptguard.local / admin

**Add Server:**
1. Right-click "Servers" → Create → Server
2. Name: ScriptGuard Dev
3. Connection tab:
   - Host: postgres (container name)
   - Port: 5432
   - Database: scriptguard
   - Username: scriptguard
   - Password: scriptguard

### Prometheus + Grafana (Monitoring)

```bash
cd docker
docker-compose -f docker-compose.dev.yml --profile monitoring up -d
```

**Access:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## Quick Reference

### Start/Stop Commands

```bash
# Start all infrastructure
cd docker && docker-compose -f docker-compose.dev.yml up -d

# Start specific service
cd docker && docker-compose -f docker-compose.dev.yml up -d postgres

# Stop all
cd docker && docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes (fresh start)
cd docker && docker-compose -f docker-compose.dev.yml down -v

# View logs
cd docker && docker-compose -f docker-compose.dev.yml logs -f

# View logs for specific service
cd docker && docker-compose -f docker-compose.dev.yml logs -f postgres
```

### Python Commands

```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Run training
python src/main.py

# Run API
uvicorn scriptguard.api.inference:app --reload

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

### Database Commands

```bash
# Connect to PostgreSQL
docker exec -it scriptguard-postgres-dev psql -U scriptguard -d scriptguard

# Backup
docker exec scriptguard-postgres-dev pg_dump -U scriptguard scriptguard > backup.sql

# Restore
docker exec -i scriptguard-postgres-dev psql -U scriptguard scriptguard < backup.sql

# View tables
docker exec scriptguard-postgres-dev psql -U scriptguard -d scriptguard -c "\dt"
```

---

## Next Steps

- **Start coding!** Edit files in `src/scriptguard/`
- **Train a model:** [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)
- **Build features:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Deploy to production:** [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## Support

Issues with local development?
- **GitHub Issues**: https://github.com/yourusername/ScriptGuard/issues
- **Documentation**: [docs/](../docs/)
- **Email**: support@scriptguard.io
