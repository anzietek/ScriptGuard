@echo off
echo ===================================
echo ScriptGuard Services Status Check
echo ===================================
echo.

echo [1] Docker Containers Status:
docker ps --filter "name=scriptguard" --format "  - {{.Names}}: {{.Status}}"
echo.

echo [2] Qdrant Health Details:
docker inspect scriptguard-qdrant-dev --format="  Status: {{.State.Health.Status}}"
docker inspect scriptguard-qdrant-dev --format="  FailingStreak: {{.State.Health.FailingStreak}}"
echo.

echo [3] PostgreSQL Health:
docker inspect scriptguard-postgres-dev --format="  Status: {{.State.Health.Status}}"
echo.

echo [4] ZenML Health:
docker inspect scriptguard-zenml-dev --format="  Status: {{.State.Health.Status}}" 2>nul || echo   Status: not running
echo.

echo [5] Testing Database Connection:
python -c "import sys; sys.path.insert(0, 'src'); from scriptguard.database.db_schema import DatabasePool; DatabasePool.initialize(); print('  ✓ DB Connection: OK')" 2>&1
echo.

echo [6] Testing Qdrant Connection:
curl -s http://localhost:6333/ >nul 2>&1 && echo   ✓ Qdrant API: OK || echo   ✗ Qdrant API: FAILED
echo.

echo [7] Testing ZenML Connection:
curl -s http://localhost:8237/health >nul 2>&1 && echo   ✓ ZenML API: OK || echo   ✗ ZenML API: FAILED
echo.

echo ===================================
