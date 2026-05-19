@echo off
REM Sprint H bug-hunt: ComfyUI headless launcher for the H0 baseline.
REM Mirrors C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat but writes
REM to a timestamped log so each session has its own scrollback.
REM
REM Tested Blackwell settings carried over verbatim:
REM   TORCH_SDPA_BACKEND=math
REM   --highvram --cuda-malloc
REM   --user-directory C:\Users\jeffr\Documents\ComfyUI
REM
REM 2026-05-18 fix: removed --force-fp16. On RTX 5080 16 GB the flag
REM upcast flux1-dev-fp8.safetensors from ~11 GiB (fp8) to ~22 GiB
REM (fp16), forcing the dynamic offloader to thrash per sampler step
REM at ~9-15 minutes per step. With the flag removed FLUX loads at
REM native fp8 weight dtype and the sampler runs at ~10-15 sec/step.
REM See docs/2026-05-19-flux-fp8-dtype-fix.md.
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
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py --port 8000 --highvram --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI > C:\Users\jeffr\Documents\ComfyUI\logs\comfy_session_h0_baseline.log 2>&1
