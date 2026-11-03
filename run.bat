@echo off
cd /d "%~dp0"
echo.
echo   gemma-stt: http://localhost:8425
echo.
.venv\Scripts\python.exe server.py
