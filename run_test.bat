@echo off
REM Quick test script for ScriptGuard pipeline
REM Uses config.test.yaml for fast validation

echo ================================================
echo ScriptGuard Pipeline - Quick Test
echo ================================================
echo.
echo This will run a minimal test of the entire pipeline:
echo - Minimal data ingestion (~30 samples)
echo - Fast vectorization (100 samples max)
echo - 1 epoch training with small batches
echo - Quick evaluation
echo.
echo Estimated time: 10-15 minutes
echo ================================================
echo.

REM Set environment to use test config
set CONFIG_PATH=config.test.yaml

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "test_models" mkdir test_models
if not exist "cache" mkdir cache

echo [1/4] Starting pipeline with test configuration...
python -m scriptguard.pipelines.training_pipeline

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================
    echo ERROR: Pipeline failed with exit code %ERRORLEVEL%
    echo ================================================
    echo.
    echo Check logs in: logs\scriptguard-test.log
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================
echo SUCCESS: Pipeline completed successfully!
echo ================================================
echo.
echo Model saved to: test_models\scriptguard-test\
echo Logs saved to: logs\scriptguard-test.log
echo.
echo To test with production config, run:
echo   python -m scriptguard.pipelines.training_pipeline
echo.
pause
