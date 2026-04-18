@echo off
REM Visual trio smoke test launcher.
REM Exit 0 = green, non-zero = red.
cd /d %~dp0..
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\smoke_visual_e2e.py
exit /b %ERRORLEVEL%
