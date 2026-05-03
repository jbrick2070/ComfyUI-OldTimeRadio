@echo off
REM ===========================================================================
REM scripts\setup_dev_env.cmd  --  one-time PATH bootstrap for OTR development
REM ===========================================================================
REM Some Windows installs ship Git but never put it on the system PATH.
REM Same story for Python (the ComfyUI venv lives outside Program Files
REM and isn't on PATH by default). This script:
REM
REM   1. Checks whether `git` already resolves on PATH -- no-op if so.
REM   2. Otherwise scans the standard install locations and prepends the
REM      first one it finds to PATH for the CURRENT cmd session only.
REM   3. If you pass --persist, it also writes the change to your user
REM      environment so new shells inherit it (uses setx).
REM
REM USAGE
REM   scripts\setup_dev_env.cmd            -- session-only PATH fix
REM   scripts\setup_dev_env.cmd --persist  -- session + permanent (setx)
REM
REM Re-run any time. Safe to call repeatedly. ASCII-only, no BOM.
REM ===========================================================================

setlocal EnableDelayedExpansion

set "_ADDED="

REM ---- Git -----------------------------------------------------------------
where git >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [setup_dev_env] git: already on PATH, skipping
) else (
    set "_GIT_CANDIDATES=C:\Program Files\Git\cmd;C:\Program Files (x86)\Git\cmd;%LOCALAPPDATA%\Programs\Git\cmd"
    for %%G in ("C:\Program Files\Git\cmd" "C:\Program Files (x86)\Git\cmd" "%LOCALAPPDATA%\Programs\Git\cmd") do (
        if exist "%%~G\git.exe" (
            if not defined _GIT_HIT (
                set "_GIT_HIT=%%~G"
                set "PATH=%%~G;!PATH!"
                set "_ADDED=!_ADDED! %%~G;"
                echo [setup_dev_env] git:   prepended %%~G
            )
        )
    )
    if not defined _GIT_HIT (
        echo [setup_dev_env] git: NOT FOUND in any standard install location
        echo                  Install Git for Windows from https://git-scm.com/download/win
    )
)

REM ---- ComfyUI venv python -------------------------------------------------
set "_VENV_PY=C:\Users\%USERNAME%\Documents\ComfyUI\.venv\Scripts"
if exist "%_VENV_PY%\python.exe" (
    echo %PATH% | findstr /C:"%_VENV_PY%" >nul
    if errorlevel 1 (
        set "PATH=%_VENV_PY%;%PATH%"
        set "_ADDED=!_ADDED! %_VENV_PY%;"
        echo [setup_dev_env] venv:  prepended %_VENV_PY%
    ) else (
        echo [setup_dev_env] venv:  already on PATH, skipping
    )
) else (
    echo [setup_dev_env] venv:  no ComfyUI venv at %_VENV_PY% -- skipping
)

REM ---- System32 sanity (some shells inherit a broken PATH) -----------------
echo %PATH% | findstr /I /C:"%SystemRoot%\System32" >nul
if errorlevel 1 (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
    set "_ADDED=!_ADDED! %SystemRoot%\System32;"
    echo [setup_dev_env] sys32: PATH was missing System32 -- restored
)

REM ---- Persist? ------------------------------------------------------------
if /I "%~1"=="--persist" (
    if defined _ADDED (
        echo [setup_dev_env] persisting to user environment via setx ...
        "%SystemRoot%\System32\setx.exe" PATH "%PATH%" >nul
        if errorlevel 1 (
            echo [setup_dev_env] setx failed -- you may need to run as admin
        ) else (
            echo [setup_dev_env] persisted. Open a NEW cmd window for it to take effect everywhere.
        )
    ) else (
        echo [setup_dev_env] nothing to persist -- PATH was already complete
    )
)

echo.
echo [setup_dev_env] done. git location:
where git
echo.
endlocal & set "PATH=%PATH%"
