@echo off
call "%~dp0.venv\Scripts\activate.bat" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
python "%~dp0app.py" %*
