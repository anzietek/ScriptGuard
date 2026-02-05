@echo off
REM ScriptGuard Development Setup Script for Windows
REM This script sets up the infrastructure and prepares local development environment
REM Usage:
REM   dev-setup.bat         - Normal setup
REM   dev-setup.bat --clean - Clean databases and restart

setlocal enabledelayedexpansion

REM Check for --clean argument
set CLEAN_MODE=0
if "%1"=="--clean" set CLEAN_MODE=1

echo ========================================================
echo   ScriptGuard Development Setup (Windows)
if %CLEAN_MODE%==1 (
    echo   MODE: Clean databases and restart
) else (
    echo   MODE: Normal setup
)
echo ========================================================
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is installed

REM Check if Docker Compose is installed
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)
echo [SUCCESS] Docker Compose is installed

REM Setup environment file
if not exist .env.dev (
    echo [ERROR] .env.dev file not found!
    pause
    exit /b 1
)

if not exist .env (
    echo [INFO] Creating .env from .env.dev...
    copy .env.dev .env
    echo [SUCCESS] .env file created
) else (
    echo [WARNING] .env file already exists, skipping...
)

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs
if not exist model_checkpoints mkdir model_checkpoints
echo [SUCCESS] Directories created

REM Clean databases if --clean flag is set
if %CLEAN_MODE%==1 (
    echo.
    echo [WARNING] Clean mode enabled - this will DELETE ALL DATA!
    set /p CONFIRM="Are you sure you want to clean databases? (yes/no): "
    if /i "!CONFIRM!"=="yes" (
        echo [INFO] Stopping services and cleaning databases...
        cd docker
        docker-compose -f docker-compose.dev.yml down -v
        echo [SUCCESS] Databases cleaned
        cd ..
    ) else (
        echo [INFO] Clean cancelled, proceeding with normal startup...
        set CLEAN_MODE=0
    )
)

REM Start infrastructure
echo [INFO] Starting infrastructure services (PostgreSQL, Qdrant, ZenML)...
cd docker
docker-compose -f docker-compose.dev.yml --profile with-zenml up -d postgres qdrant zenml
cd ..
echo [SUCCESS] Infrastructure services started

REM Wait for services
echo [INFO] Waiting for services to be healthy (this may take 30-60 seconds)...
timeout /t 10 /nobreak >nul

REM Check PostgreSQL
echo [INFO] Checking PostgreSQL...
:check_postgres
docker exec scriptguard-postgres-dev pg_isready -U scriptguard -d scriptguard >nul 2>nul
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto check_postgres
)
echo [SUCCESS] PostgreSQL is ready

REM Check Qdrant
echo [INFO] Checking Qdrant...
:check_qdrant
curl -f http://localhost:6333/ >nul 2>nul
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto check_qdrant
)
echo [SUCCESS] Qdrant is ready

REM Check ZenML
echo [INFO] Checking ZenML...
:check_zenml
curl -f http://localhost:8237/health >nul 2>nul
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto check_zenml
)
echo [SUCCESS] ZenML is ready

REM Initialize database
echo [INFO] Initializing database schema...
docker exec scriptguard-postgres-dev psql -U scriptguard -d scriptguard -c "\dt" | findstr "samples" >nul
if %errorlevel% neq 0 (
    docker exec scriptguard-postgres-dev psql -U scriptguard -d scriptguard -f /docker-entrypoint-initdb.d/init.sql
    echo [SUCCESS] Database initialized
) else (
    echo [SUCCESS] Database already initialized
)

REM Ask about Python setup
echo.
set /p SETUP_PYTHON="Setup Python virtual environment and install dependencies? (y/n): "
if /i "%SETUP_PYTHON%"=="y" (
    echo [INFO] Setting up Python environment...

    if not exist venv (
        echo [INFO] Creating virtual environment...
        python -m venv venv
        echo [SUCCESS] Virtual environment created
    ) else (
        echo [WARNING] Virtual environment already exists
    )

    echo [INFO] Installing dependencies...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -e .
    echo [SUCCESS] Python dependencies installed

    echo [INFO] Bootstrapping Qdrant with CVE data...
    python -c "from scriptguard.rag import QdrantStore, bootstrap_cve_data; import logging; logging.basicConfig(level=logging.INFO); store = QdrantStore(); bootstrap_cve_data(store); print('CVE data bootstrapped')"
    echo [SUCCESS] Qdrant bootstrapped
)

REM Display info
echo.
echo ========================================================
echo   ScriptGuard Development Environment Ready!
echo ========================================================
echo.
echo Infrastructure Services:
echo   PostgreSQL:  localhost:5432
echo     Database:  scriptguard
echo     User:      scriptguard
echo     Password:  scriptguard
echo.
echo   Qdrant:      http://localhost:6333
echo     Dashboard: http://localhost:6333/dashboard
echo.
echo   ZenML:       http://localhost:8237
echo     Dashboard: http://localhost:8237
echo.
echo Connection Strings:
echo   PostgreSQL: postgresql://scriptguard:scriptguard@localhost:5432/scriptguard
echo   Qdrant:     http://localhost:6333
echo   ZenML:      http://localhost:8237
echo.
echo Useful Commands:
echo   Start infrastructure:  cd docker ^&^& docker-compose -f docker-compose.dev.yml --profile with-zenml up -d
echo   Stop infrastructure:   cd docker ^&^& docker-compose -f docker-compose.dev.yml down
echo   Clean databases:       dev-setup.bat --clean
echo   View logs:            cd docker ^&^& docker-compose -f docker-compose.dev.yml logs -f
echo.
echo   Activate venv:        venv\Scripts\activate.bat
echo   Run training:         python src\main.py
echo   Run API:              uvicorn scriptguard.api.main:app --reload
echo.
echo   Python shell:         python
echo   PostgreSQL shell:     docker exec -it scriptguard-postgres-dev psql -U scriptguard -d scriptguard
echo.
echo Optional Services (with profiles):
echo   With pgAdmin:         cd docker ^&^& docker-compose -f docker-compose.dev.yml --profile with-pgadmin up -d
echo     pgAdmin URL:        http://localhost:5050 (admin@scriptguard.local / admin)
echo.
echo   With monitoring:      cd docker ^&^& docker-compose -f docker-compose.dev.yml --profile monitoring up -d
echo     Prometheus:         http://localhost:9090
echo     Grafana:            http://localhost:3000 (admin/admin)
echo.
echo ========================================================
echo.
echo [SUCCESS] Setup complete! You can now start developing.
echo.
pause
