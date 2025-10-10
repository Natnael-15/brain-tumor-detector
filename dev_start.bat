@echo off
title Brain MRI Tumor Detector - Development Start

echo 🧠 Brain MRI Tumor Detector - Development Mode
echo.

cd /d "%~dp0"

echo Checking requirements...
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js first.
    pause
    exit /b 1
)

echo ✅ Requirements satisfied
echo.

echo Installing/updating backend dependencies...
cd backend
venv\Scripts\pip.exe install -q uvicorn fastapi python-multipart aiofiles numpy

echo Starting backend with detailed logging...
start "Backend-Dev" cmd /k "echo Backend Development Server && venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug"

echo.
echo Starting frontend in development mode...
cd ..\frontend
start "Frontend-Dev" cmd /k "echo Frontend Development Server && npm run dev"

echo.
echo ✅ Development servers starting...
echo.
echo 📊 Backend (Debug): http://localhost:8000
echo 🌐 Frontend (Dev): http://localhost:3000
echo 📚 API Docs: http://localhost:8000/docs
echo.

timeout /t 15 /nobreak >nul
start http://localhost:3000

echo Press any key to exit...
pause >nul
start "🔴 Backend Server - Port 8000" cmd /c "cd /d \"%~dp0\" && echo ====================================================== && echo 🔴 BACKEND SERVER STARTING && echo ====================================================== && echo 📍 Location: %cd% && echo 🐍 Python Version: && python --version && echo 📦 Starting FastAPI with detailed logs... && echo. && python -m uvicorn backend.main:app --reload --port 8000 && echo. && echo ❌ Backend server stopped. Press any key to close... && pause"

REM Wait for backend to initialize
echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Create a new console window for Frontend with detailed logging
echo 🚀 Starting Frontend Server with detailed logging...
start "🟢 Frontend Server - Port 3000" cmd /c "cd /d \"%~dp0frontend\" && echo ====================================================== && echo 🟢 FRONTEND SERVER STARTING && echo ====================================================== && echo 📍 Location: %cd% && echo 📦 Node Version: && node --version && echo 📦 NPM Version: && npm --version && echo 🚀 Starting Next.js development server... && echo. && npm run dev && echo. && echo ❌ Frontend server stopped. Press any key to close... && pause"

REM Wait for frontend to initialize
echo ⏳ Waiting for frontend to initialize...
timeout /t 8 /nobreak >nul

echo.
echo ====================================================
echo 🎉 Development Environment Started!
echo ====================================================
echo.
echo 📊 Backend API:      http://localhost:8000
echo 🌐 Frontend App:     http://localhost:3000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 Health Check:     http://localhost:8000/api/v1/health
echo.
echo 🛠️ Development Features:
echo   • Hot Reload: Frontend automatically refreshes on changes
echo   • API Reload: Backend restarts on Python file changes
echo   • Detailed Logs: Check the server windows for debugging
echo   • WebSocket: Real-time communication between frontend/backend
echo.
echo 🔧 Debugging:
echo   • Backend logs: Check the red "Backend Server" window
echo   • Frontend logs: Check the green "Frontend Server" window
echo   • Browser DevTools: F12 in browser for frontend debugging
echo.

REM Test backend health
echo 🔍 Testing backend health...
timeout /t 2 /nobreak >nul
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/health' -TimeoutSec 5; Write-Host '✅ Backend health check: PASS' -ForegroundColor Green; } catch { Write-Host '⚠️ Backend health check: PENDING (might still be starting)' -ForegroundColor Yellow; }"

echo.
echo 🌐 Opening application in browser...
timeout /t 3 /nobreak >nul
start http://localhost:3000

echo.
echo ====================================================
echo 🎯 Ready for Testing!
echo ====================================================
echo.
echo 📝 Test Workflow:
echo   1. Upload medical image (drag & drop or click)
echo   2. Select AI model (6 available: Ensemble, nnU-Net, etc.)
echo   3. Click "Start Analysis"
echo   4. Watch real-time progress in dashboard
echo   5. View 3D visualization when complete
echo.
echo 🛑 To stop: Run stop_app.bat or close server windows
echo 🔄 To restart: Run this file again
echo.
echo Press any key to close this monitoring window...
echo (Servers will continue running in their own windows)
pause >nul