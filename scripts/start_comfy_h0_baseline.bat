@echo off
REM Sprint H bug-hunt: ComfyUI headless launcher for the H0 baseline.
REM Mirrors C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat but writes
REM to a timestamped log so each session has its own scrollback.
REM
REM Tested Blackwell settings carried over verbatim:
REM   TORCH_SDPA_BACKEND=math
REM   --highvram --force-fp16 --cuda-malloc
REM   --user-directory C:\Users\jeffr\Documents\ComfyUI
REM
REM Sprint H iter 2 fix (2026-05-17): HF_HOME explicitly set so the
REM OTR model catalog scan finds the curated cache at
REM C:\ComfyUI-Models\huggingface\hub. The HKCU user-scope env var
REM does NOT propagate through PowerShell Start-Process -> cmd /c
REM child processes, which caused all 7 curated models to label as
REM [NOT DOWNLOADED] in iter 2 even though the cache was on disk.
REM
set TORCH_SDPA_BACKEND=math
set HF_HOME=C:\ComfyUI-Models\huggingface
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py --port 8000 --highvram --force-fp16 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI > C:\Users\jeffr\Documents\ComfyUI\logs\comfy_session_h0_baseline.log 2>&1
