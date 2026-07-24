@echo off
REM otr_check -- client source-bank checker (independent source banks v1).
REM
REM Resolution order, most explicit first:
REM   1. OTR_PYTHON            -- an operator override, always wins
REM   2. the ComfyUI venv      -- the interpreter that actually runs the nodes
REM   3. py -3 / python        -- a plain install, last resort
REM
REM The venv comes before a bare `python` on purpose: the checker imports the
REM same node modules ComfyUI does, so it must judge the bundle with the
REM interpreter that will import it, not whichever python is first on PATH.
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OTR_CHECK_PY=%~dp0otr_check.py"

if defined OTR_PYTHON (
  "%OTR_PYTHON%" "%OTR_CHECK_PY%" %*
  exit /b %ERRORLEVEL%
)

set "OTR_VENV_PY=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%OTR_VENV_PY%" (
  "%OTR_VENV_PY%" "%OTR_CHECK_PY%" %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 "%OTR_CHECK_PY%" %*
  exit /b %ERRORLEVEL%
)

python "%OTR_CHECK_PY%" %*
exit /b %ERRORLEVEL%
