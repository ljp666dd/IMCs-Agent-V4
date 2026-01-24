@echo off
setlocal

echo ============================================
echo      IMCs Scientific Platform Launcher
echo ============================================
echo.

:: 1. Check Root Directory (Simple check for src)
if not exist "src" (
    echo [ERROR] Please run this script from the project root directory.
    pause
    exit /b
)

:: 2. Start Backend (API)
echo [1/2] Starting Backend Server (Port 8000)...
start "IMCs Backend" cmd /k "python src/api/main.py"

:: 3. Start Frontend (Web UI)
echo [2/2] Starting Frontend Interface (Port 3000)...
cd src/ui/web
if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    call npm install
)
start "IMCs Frontend" cmd /k "npm run dev"

:: 4. Launch Browser (Wait for servers to spin up)
echo [INFO] Waiting for services to initialize...
timeout /t 8 >nul
start http://localhost:3000

echo.
echo ============================================
echo [SUCCESS] System is starting up!
echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
echo You can close this launcher window now.
echo ============================================
timeout /t 5
