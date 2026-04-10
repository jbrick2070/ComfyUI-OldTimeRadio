@echo off
title Old Time Radio - Live VRAM Telemetry
color 0F
cls
echo Starting VRAM Telemetry Dashboard...
"..\..\..\.venv\Scripts\python.exe" vram_telemetry.py
pause
