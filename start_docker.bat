@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"
title Blind Navigation Docker Launcher

echo ============================================
echo Blind Navigation Demo - Docker Launcher
echo ============================================
echo.

set "MODE=%~1"
set "COMPOSE_FILES=-f docker-compose.yml"

if /I "%MODE%"=="gpu" (
    set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
    echo [INFO] GPU mode enabled.
) else (
    echo [INFO] CPU-compatible mode enabled.
    echo [INFO] Use start_docker.bat gpu to request NVIDIA GPU.
)

echo [1/4] Checking Docker CLI...
docker --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH.
    pause
    exit /b 1
)

echo [2/4] Checking Docker daemon...
docker info >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running.
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [3/4] Starting containers...
docker compose %COMPOSE_FILES% up -d --build
if errorlevel 1 (
    echo [ERROR] Docker compose startup failed.
    echo Try: docker compose logs --tail=200
    pause
    exit /b 1
)

echo [4/4] Opening browser: http://127.0.0.1:8081
start "" "http://127.0.0.1:8081"

echo.
echo Containers are running.
echo View logs: docker compose %COMPOSE_FILES% logs -f
echo Stop: docker compose %COMPOSE_FILES% down
echo.
pause
