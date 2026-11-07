@echo off
setlocal enabledelayedexpansion
title FloodSense-AI Master Controller

echo ===============================================================================
echo   FLOODSENSE-AI: TACTICAL DISASTER COMMAND CENTER - LAUNCH CONTROLLER
echo ===============================================================================
echo.

:: 1. Terminate any previous instances running on port 8000 (Backend)
echo [*] Checking and stopping existing processes on port 8000 (FastAPI Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /R /C:":8000 .*LISTENING"') do (
    echo     [-] Terminating PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Terminate any previous instances running on port 5173 (Frontend)
echo [*] Checking and stopping existing processes on port 5173 (Vite Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /R /C:":5173 .*LISTENING"') do (
    echo     [-] Terminating PID %%a on port 5173...
    taskkill /F /PID %%a >nul 2>&1
)

echo [OK] Ports 8000 and 5173 cleared.
echo.

:: 3. Launch Backend in a dedicated titled window
echo [*] Launching FastAPI Backend on http://localhost:8000...
cd /d "%~dp0backend"
start "FloodSense-AI [BACKEND - Port 8000]" cmd /k "python run.py"

:: Wait brief moment for backend to initialize
timeout /t 3 /nobreak >nul

:: 4. Launch Frontend in a dedicated titled window
echo [*] Launching React Tactical Command Center on http://localhost:5173...
cd /d "%~dp0frontend"
start "FloodSense-AI [FRONTEND - Port 5173]" cmd /k "npm.cmd run dev"

:: 5. Open browser automatically
timeout /t 3 /nobreak >nul
echo [*] Opening Command Center Dashboard in browser...
start http://localhost:5173

echo.
echo ===============================================================================
echo   [SUCCESS] Both servers are actively running:
echo     - Backend API Docs : http://localhost:8000/docs
echo     - Frontend Web UI  : http://localhost:5173
echo.
echo   Leave the two opened terminal windows running.
echo   To restart anytime, simply run this launch.bat file again!
echo ===============================================================================
echo.
pause
