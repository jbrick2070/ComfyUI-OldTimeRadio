@echo off
REM hyworld_smoke_dc.cmd — Desktop Commander one-liner for HyWorld smoke test
REM Run via Desktop Commander start_process or paste into cmd.exe
REM
REM Usage:
REM   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
REM   scripts\hyworld_smoke_dc.cmd
REM
REM Or via Desktop Commander:
REM   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && scripts\hyworld_smoke_dc.cmd

cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

echo ============================================================
echo HyWorld Smoke Test — %date% %time%
echo ============================================================

REM AST parse check (syntax errors would be caught here)
echo.
echo --- AST Parse Check ---
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -c "import ast, sys; [ast.parse(open(f).read()) for f in ['otr_v2/hyworld/__init__.py','otr_v2/hyworld/shotlist.py','otr_v2/hyworld/bridge.py','otr_v2/hyworld/poll.py','otr_v2/hyworld/renderer.py','otr_v2/hyworld/worker.py']]; print('AST: All 6 hyworld files parse OK')"
if errorlevel 1 (
    echo AST PARSE FAILED — fix syntax before continuing
    exit /b 1
)

REM Smoke test
echo.
echo --- Smoke Test ---
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts/hyworld_smoketest.py --cleanup

echo.
echo --- Done ---
echo Report at: logs\hyworld_smoketest_report.md
