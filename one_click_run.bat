@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"
title Blind Navigation Unified Launcher

echo ============================================
echo Blind Navigation Demo - Unified Launcher
echo ============================================
echo.

echo [1/5] Checking Docker Desktop...
docker --version >nul 2>nul
if errorlevel 1 goto :local_mode

docker info >nul 2>nul
if errorlevel 1 goto :local_mode

echo [INFO] Docker is available. Starting Docker mode...
call "%~dp0start_docker.bat"
exit /b %errorlevel%

:local_mode
echo [INFO] Docker is unavailable. Switching to local Python mode.
echo.

echo [2/5] Checking Python command...
python --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [3/5] Checking local virtual environment folder: VENV
if exist "%~dp0VENV\Scripts\python.exe" goto :start_local

echo [4/5] Creating local virtual environment...
python -m venv VENV
if errorlevel 1 (
    echo [ERROR] Failed to create VENV.
    pause
    exit /b 1
)

echo [5/5] Installing Python dependencies...
call "%~dp0VENV\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate VENV.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:start_local
echo [INFO] Starting local mode...
call "%~dp0start_project.bat"
exit /b %errorlevel%
