@echo off
title Brain MRI Tumor Detector - Stop Servers

echo 🛑 Stopping Brain MRI Tumor Detector servers...
echo.

echo Stopping Node.js processes...
taskkill /f /im node.exe >nul 2>&1

echo Stopping Python processes...
taskkill /f /im python.exe >nul 2>&1

echo Stopping processes on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo.
echo ✅ All servers stopped!
echo.
echo Press any key to exit...
pause >nul