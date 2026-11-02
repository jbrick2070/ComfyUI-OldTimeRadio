@echo off
rem Double-click to start Radio Play Studio: it installs the app's two packages
rem the first time, starts comfy-api-proxy if it is not running, checks the
rem setup, and opens the app in your browser. Start ComfyUI first.
rem
rem Which Python: RPS_PYTHON if you set it, else a .venv in this folder,
rem else the py launcher, else python on PATH.
rem
rem No goto labels: the repo stores this file with LF line endings, and cmd can
rem miss a label in an LF-only batch file.
setlocal
cd /d "%~dp0"

set "PYARGS="
if defined RPS_PYTHON (
    set "PY=%RPS_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
    where py >nul 2>nul && set "PY=py" && set "PYARGS=-3"
)

"%PY%" %PYARGS% -c "import comfy_sdk, comfy_api_proxy" >nul 2>nul
if errorlevel 1 (
    echo Installing the app's packages - comfy-sdk and comfy-api-proxy - this happens once.
    "%PY%" %PYARGS% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo The packages did not install. The lines above say why.
        pause
        exit /b 1
    )
)

"%PY%" %PYARGS% -m radio_play_studio start %*
if errorlevel 1 (
    echo.
    echo Radio Play Studio stopped with an error. The lines above say why.
    pause
    exit /b 1
)
