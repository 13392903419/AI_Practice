@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"
title Blind Navigation Demo Launcher

echo ============================================
echo Blind Navigation Demo - One Click Launcher
echo ============================================
echo.

set "VENV=VENV"
set "CONDA_BAT="
set "ACTIVATED_BY_SCRIPT=0"
set "ACTIVATE_MODE=none"

if exist "%~dp0%VENV%\Scripts\activate.bat" (
    echo [1/4] Found local environment folder: %VENV%
    call "%~dp0%VENV%\Scripts\activate.bat"
    if not errorlevel 1 (
        set "ACTIVATED_BY_SCRIPT=1"
        set "ACTIVATE_MODE=venv"
    )
)

if "%ACTIVATED_BY_SCRIPT%"=="1" goto :after_activation

for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%I"
    goto :found_conda
)

:found_conda
if defined CONDA_BAT (
    echo [1/4] Trying Conda env: %VENV%
    call "%CONDA_BAT%" activate %VENV%
    if not errorlevel 1 (
        set "ACTIVATED_BY_SCRIPT=1"
        set "ACTIVATE_MODE=conda"
    )
) else (
    echo [1/4] Conda not found in PATH, skipping Conda activation.
)

:after_activation
if "%ACTIVATED_BY_SCRIPT%"=="1" (
    if "%ACTIVATE_MODE%"=="venv" echo [INFO] Local venv activated: %VENV%
    if "%ACTIVATE_MODE%"=="conda" echo [INFO] Conda env activated: %VENV%
) else (
    echo.
    echo [WARN] Did not activate environment named: %VENV%
    echo [WARN] Will continue with current Python in this terminal.
    echo [WARN] If startup fails, create/select a ready Python environment first.
    echo.
)

echo [2/4] Checking Python command...
python --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH.
    echo Please install Python or open this project in a configured environment.
    echo.
    pause
    exit /b 1
)

echo [3/4] Opening browser: http://127.0.0.1:8081
echo [INFO] 127.0.0.1 means this computer itself (local demo address).
start "" "http://127.0.0.1:8081"

echo [4/4] Starting project...
echo Press Ctrl + C to stop.
echo.
python app_main.py

echo.
echo Project stopped.
pause
