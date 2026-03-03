@echo off
chcp 65001 >nul
setlocal

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=%~dp0"
set "OVERWRITE=%~2"

set "PS1=%~dp0convert_txt_to_json.ps1"
if not exist "%PS1%" (
  echo [ERROR] convert_txt_to_json.ps1 not found next to this bat file.
  echo Place both files in the same folder and run again.
  pause
  exit /b 1
)

echo ========================================
echo   Convert TXT to Session JSON
echo ========================================
echo Target: %TARGET%
echo.

if /I "%OVERWRITE%"=="overwrite" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -TargetDir "%TARGET%" -Overwrite
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -TargetDir "%TARGET%"
)

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] Conversion failed.
  pause
  exit /b 1
)

echo.
echo Done.
endlocal
