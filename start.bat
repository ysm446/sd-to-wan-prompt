@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "MODE=%~1"
if "%MODE%"=="" set "MODE=gradio"

set "API_HOST=127.0.0.1"
set "API_PORT=7861"

echo ========================================
echo   WAN Prompt Generator Launcher
echo ========================================
echo Mode: %MODE%
echo.

REM Initialize Conda
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "D:\miniconda3\Scripts\activate.bat" (
    call "D:\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else (
    echo [ERROR] Conda not found. Please check your Conda installation.
    echo.
    echo Use a Conda prompt and run:
    echo   conda activate main
    echo   python app.py --mode gradio
    echo.
    pause
    exit /b 1
)

echo Activating conda environment: main
call conda activate main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Environment 'main' not found
    echo.
    echo Please setup the environment first:
    echo   1. conda create -n main python=3.10 -y
    echo   2. conda activate main
    echo   3. pip install -r requirements.txt
    echo   4. python scripts/setup.py
    echo.
    pause
    exit /b 1
)

echo Environment activated successfully
echo.

if /I "%MODE%"=="gradio" goto RUN_GRADIO
if /I "%MODE%"=="api" goto RUN_API
if /I "%MODE%"=="electron" goto RUN_ELECTRON
if /I "%MODE%"=="help" goto SHOW_HELP

echo [ERROR] Unknown mode: %MODE%
goto SHOW_HELP

:RUN_GRADIO
echo Starting Gradio UI...
echo Press Ctrl+C to stop
echo.
python app.py --mode gradio
goto END

:RUN_API
echo Starting API server on %API_HOST%:%API_PORT% ...
echo Press Ctrl+C to stop
echo.
python app.py --mode api --host %API_HOST% --port %API_PORT%
goto END

:RUN_ELECTRON
echo Starting Electron desktop app...
set "WAN_API_HOST=%API_HOST%"
set "WAN_API_PORT=%API_PORT%"
set "PYTHON_EXECUTABLE=python"
set "ELECTRON_RUN_AS_NODE="

if not exist "desktop\electron\package.json" (
    echo [ERROR] desktop\electron\package.json not found.
    echo Run this from the project root.
    goto END
)

pushd desktop\electron
if not exist "node_modules" (
    echo Installing Electron dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] npm install failed.
        popd
        goto END
    )
)

call npm start
popd
goto END

:SHOW_HELP
echo.
echo Usage:
echo   start.bat [mode]
echo.
echo Modes:
echo   gradio   Start legacy Gradio UI (default)
echo   api      Start FastAPI backend only
echo   electron Start Electron desktop app
echo   help     Show this help

:END
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo [ERROR] Process ended with errors
    echo ========================================
    pause
)
endlocal
