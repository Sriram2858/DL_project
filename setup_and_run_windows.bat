@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo =========================================
echo   VoxAssist - setup and launch
echo =========================================

if not exist ".venv\Scripts\python.exe" (
    set "PY_LAUNCHER=py -3.11"
    py -3.11 --version >nul 2>&1
    if errorlevel 1 set "PY_LAUNCHER=py -3.12"
    echo Creating virtual environment with !PY_LAUNCHER!...
    !PY_LAUNCHER! -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Install Python 3.11 or 3.12 and enable the Python Launcher, then try again.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

if not exist "artifacts\model.pt" (
    echo Training the intent classifier for the first time...
    python train.py
    if errorlevel 1 goto failed
)

echo.
echo Open http://127.0.0.1:5000 in Chrome or Edge.
echo Leave this window open while using VoxAssist.
echo Press Ctrl+C in this window to stop the server.
echo.
python app.py
if errorlevel 1 goto failed
goto end

:failed
echo.
echo Setup did not complete. Check that Python 3.11 or 3.12 and internet access are available.
pause
exit /b 1

:end
endlocal
