@echo off
echo === YT2 Setup ===

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

where yt-dlp >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: yt-dlp not found. Install with: pip install yt-dlp
)

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg not found. Install from https://ffmpeg.org/download.html
)

if not exist "%~dp0.venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv"
)

echo Installing dependencies...
call "%~dp0.venv\Scripts\activate.bat"
pip install -r "%~dp0requirements.txt" -q

echo.
echo === Setup complete ===
echo Run: run.bat URL
pause
