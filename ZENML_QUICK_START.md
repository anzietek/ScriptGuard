# ZenML Quick Start Guide

## 🎯 Goal
Enable persistent ZenML pipeline history by centralizing the server on Contabo VPS.

## 📋 What Was Changed?

### ✅ Files Modified
1. **`contabo/docker-compose.yml`** - Added ZenML service with PostgreSQL backend
2. **`.env.podrun`** - Added `ZENML_MODE` configuration
3. **`podrun-setup.sh`** - Made `init_zenml()` support both local and remote modes
4. **`contabo/setup.sh`** - Added ZenML directory creation and tunnel instructions

### ✅ Files Created
1. **`docs/ZENML_SETUP.md`** - Complete configuration guide
2. **`contabo/.env.example`** - VPS environment template
3. **`ZENML_CENTRALIZATION_IMPLEMENTATION.md`** - Implementation details

## 🚀 Quick Start

### Option 1: Local Mode (Current Behavior - Default)

No changes needed! System defaults to local mode.

```bash
# On RunPod
cd /workspace/ScriptGuard
./podrun-setup.sh -y
```

**What happens:**
- Starts ZenML server locally on RunPod (port 8237)
- Uses SQLite backend (ephemeral)
- Data lost on pod restart
- No VPS dependency

### Option 2: Remote Mode (Production - Persistent)

#### Step 1: Set up VPS ZenML Server (One-Time Setup)

```bash
# SSH to VPS
ssh deployer@62.171.130.236

# Navigate to project
cd /path/to/ScriptGuard/contabo

# Create environment file
cp .env.example .env
nano .env  # Set POSTGRES_PASSWORD to a strong password

# Create ZenML data directory
sudo mkdir -p /var/lib/scriptguard/zenml
sudo chmod -R 777 /var/lib/scriptguard/zenml

# Start ZenML service
docker compose --profile with-zenml up -d zenml

# Verify it's running
docker logs -f scriptguard-zenml
# Wait for: "ZenML server started"

# Test health endpoint
curl http://localhost:8237/health
# Expected: {"status":"ok"}
```

#### Step 2: Configure RunPod to Use Remote Mode

```bash
# On RunPod
cd /workspace/ScriptGuard

# Update configuration
nano .env.podrun
# Change line 24 from:
#   ZENML_MODE=local
# To:
#   ZENML_MODE=remote

# Save and exit (Ctrl+X, Y, Enter)

# Run setup
./podrun-setup.sh -y

# The script will:
# - Detect remote mode
# - Check VPS server health via SSH tunnel
# - Connect to remote server
# - Configure project
```

**What happens:**
- Connects to VPS ZenML server via SSH tunnel
- Uses PostgreSQL backend (persistent)
- Data survives pod restarts
- All team members share same history

## 🔄 Switching Between Modes

### From Local to Remote

```bash
# 1. Stop local server
pkill -f "zenml.*server"

# 2. Update mode
nano .env.podrun  # Set ZENML_MODE=remote

# 3. Clear local config
rm -rf .zen/

# 4. Re-run setup
./podrun-setup.sh -y
```

### From Remote to Local

```bash
# 1. Update mode
nano .env.podrun  # Set ZENML_MODE=local

# 2. Clear remote config
rm -rf .zen/

# 3. Re-run setup
./podrun-setup.sh -y
```

## 🎬 What Happens Behind the Scenes

### Local Mode
```
podrun-setup.sh
  ↓
  Detects ZENML_MODE=local
  ↓
  Starts ZenML server on port 8237
  ↓
  Uses SQLite database
  ↓
  Configures default project
```

### Remote Mode
```
podrun-setup.sh
  ↓
  Detects ZENML_MODE=remote
  ↓
  Checks SSH tunnel is active (port 8237 forwarded)
  ↓
  Checks VPS ZenML server health (10 retries)
  ↓
  If healthy: Connects to remote server
  ↓
  If failed: Falls back to local mode
  ↓
  Configures default project
```

## ⚠️ Troubleshooting

### Remote Mode: "Cannot connect to remote ZenML server"

**Symptoms:**
```
Cannot connect to remote ZenML server at http://localhost:8237
Falling back to local mode...
```

**Solutions:**

1. **Check SSH tunnel is running:**
   ```bash
   pgrep -f "ssh.*62.171.130.236"
   ```
   If no output, tunnel is down. The `podrun-setup.sh` script should start it automatically.

2. **Check VPS server is running:**
   ```bash
   ssh deployer@62.171.130.236 'docker ps | grep zenml'
   ```
   Should show: `scriptguard-zenml` with status `(healthy)`

3. **Test connection manually:**
   ```bash
   curl http://localhost:8237/health
   ```
   Should return: `{"status":"ok"}`

4. **Check VPS logs:**
   ```bash
   ssh deployer@62.171.130.236 'docker logs scriptguard-zenml --tail 50'
   ```

### Local Mode: "ZenML server failed to start"

**Solutions:**

1. **Check if port is already in use:**
   ```bash
   lsof -Pi :8237
   pkill -f "zenml.*server"  # Kill existing process
   ```

2. **Check logs:**
   ```bash
   cat logs/zenml_server.log
   ```

3. **Restart manually:**
   ```bash
   uv run zenml up --host 0.0.0.0 --port 8237
   ```

## 📊 Configuration Comparison

| Feature | Local Mode | Remote Mode |
|---------|-----------|-------------|
| **Backend** | SQLite | PostgreSQL |
| **Persistence** | ❌ Lost on restart | ✅ Survives restarts |
| **Team Sharing** | ❌ No | ✅ Yes |
| **VPS Required** | ❌ No | ✅ Yes |
| **Memory (RunPod)** | +200MB | +50MB |
| **Setup Time** | ~5s | ~10s |
| **Latency** | 0ms | +20-50ms |

## 🎯 Recommended Usage

- **Development/Testing:** Use **local mode**
  - Fast iteration
  - No VPS dependency
  - Quick experiments

- **Production/Training:** Use **remote mode**
  - Persistent history
  - Team collaboration
  - PostgreSQL reliability

## 📚 Full Documentation

For complete details, troubleshooting, and advanced features:
- **Setup Guide:** `docs/ZENML_SETUP.md`
- **Implementation Details:** `ZENML_CENTRALIZATION_IMPLEMENTATION.md`

## ✅ Verification

After setup, verify it's working:

```bash
# Check ZenML status
uv run zenml status

# Expected output (Local):
# Connected to a local ZenML server at http://localhost:8237

# Expected output (Remote):
# Connected to a ZenML server at http://localhost:8237
```

## 🔐 Security

Both modes are secure:
- **Local:** Server accessible only via RunPod TCP mapping
- **Remote:** VPS binds to 127.0.0.1 only, accessible via SSH tunnel
- **Tunnel:** Private key deleted after connection
- **Firewall:** Port 8237 blocked from internet on VPS

## 🚨 Important Notes

1. **Backwards Compatible:** If you don't change anything, system defaults to local mode (current behavior)

2. **Automatic Fallback:** If remote server is unreachable, script automatically falls back to local mode

3. **No Data Migration:** Switching from local to remote starts fresh (local SQLite data is ephemeral anyway)

4. **.env.podrun Not Tracked:** The `.env.podrun` file is not tracked by Git (contains API keys). You'll need to update it manually on RunPod.

5. **SSH Tunnel Required:** Remote mode requires the SSH tunnel to be active (already set up by `podrun-setup.sh`)

## 🎬 Example Workflow

### Quick Development Session (Local)
```bash
cd /workspace/ScriptGuard
./podrun-setup.sh -y
uv run python src/main.py
# Results: Ephemeral (lost on restart)
```

### Production Training Run (Remote)
```bash
# One-time: Set up VPS server (see Step 1 above)

# Every run:
cd /workspace/ScriptGuard
nano .env.podrun  # Set ZENML_MODE=remote (only needed once)
./podrun-setup.sh -y
uv run python src/main.py
# Results: Persistent (survives restarts)
```

## 💡 Tips

1. **Use local mode** for quick experiments and debugging
2. **Use remote mode** for production training runs you want to track
3. **Switch modes** by changing one environment variable
4. **Check logs** if connection fails: `docker logs scriptguard-zenml`
5. **Monitor VPS** resources: `docker stats scriptguard-zenml`

---

**Need Help?** See `docs/ZENML_SETUP.md` for complete documentation and troubleshooting.
