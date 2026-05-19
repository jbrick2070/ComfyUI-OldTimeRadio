@echo off
REM Sprint H bug-hunt: ComfyUI headless launcher for the H0 baseline.
REM Mirrors C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat but writes
REM to a timestamped log so each session has its own scrollback.
REM
REM Blackwell settings carried over verbatim from Jeffrey's working
REM start_comfy.bat at Sprint H 1f7240f:
REM   --highvram --cuda-malloc
REM   --user-directory C:\Users\jeffr\Documents\ComfyUI
REM
REM 2026-05-18 fix (BUG-LOCAL-230): removed --force-fp16. On RTX 5080
REM 16 GB the flag upcast flux1-dev-fp8.safetensors from ~11 GiB (fp8)
REM to ~22 GiB (fp16), forcing the dynamic offloader to thrash per
REM sampler step at ~9-15 minutes per step. With the flag removed FLUX
REM loads at native fp8 weight dtype.
REM See BUG_LOG.md::BUG-LOCAL-230.
REM
REM 2026-05-19 fix (BUG-LOCAL-231): removed TORCH_SDPA_BACKEND=math. The
REM env var was a defensive carry-over from 1f7240f with no empirical
REM benchmark justifying it. Math SDPA is the slowest backend; held
REM FLUX sampler at ~180 s/it baseline. Removing -> PyTorch default
REM SDPA dispatch picks efficient/flash on Blackwell sm_120.
REM See BUG_LOG.md::BUG-LOCAL-231.
REM
REM Sprint H iter 2 fix (2026-05-17): HF_HOME explicitly set so the
REM OTR model catalog scan finds the curated cache at
REM C:\ComfyUI-Models\huggingface\hub. The HKCU user-scope env var
REM does NOT propagate through PowerShell Start-Process -> cmd /c
REM child processes, which caused all 7 curated models to label as
REM [NOT DOWNLOADED] in iter 2 even though the cache was on disk.
REM
set HF_HOME=C:\ComfyUI-Models\huggingface
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py --port 8000 --highvram --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI > C:\Users\jeffr\Documents\ComfyUI\logs\comfy_session_h0_baseline.log 2>&1
