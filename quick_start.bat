@echo off
title Brain MRI Tumor Detector - Quick Start

echo 🧠 Brain MRI Tumor Detector - Quick Start
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

echo Stopping any existing project servers...
echo (Only stopping servers on ports 3000 and 8000)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo.
echo Starting backend server...
cd backend
start "Backend" cmd /k "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Starting frontend server...
cd ..\frontend  
start "Frontend" cmd /k "npm run dev"

echo.
echo ✅ Servers started!
echo.
echo 📊 Backend: http://localhost:8000
echo 🌐 Frontend: http://localhost:3000
echo.

timeout /t 10 /nobreak >nul
start http://localhost:3000

echo Press any key to exit...
pause >nul