@echo off
REM watch.cmd  --  overnight FULL-run ledger watcher
REM
REM Run this in a separate cmd window WHILE the FULL workflow is
REM queued in ComfyUI. It polls output/otr/audio/*_ledger.json
REM every 30s and writes a human-readable progress log to
REM scripts/full_run_watch_<timestamp>.log. When the run completes
REM (final_video_path appears) it runs a deep BUG-100/101/102
REM audit on the ledger and exits.
REM
REM If the most-recent ledger at start is already complete (e.g.
REM Stellar Shadows 20260428_140251), the watcher will skip its
REM audit and wait for a NEW ledger from the upcoming run.

cd /d %~dp0\..

C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\watch_full_run.py

echo.
echo Watcher exited. Log file:
dir /B /O-D scripts\full_run_watch_*.log | findstr /N "^" | findstr /B "1:"
echo.
pause
