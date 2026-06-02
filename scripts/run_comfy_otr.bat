@echo off
REM ===========================================================================
REM OTR determinism launcher (C-1 / C-7) -- THE entrypoint for OTR runs.
REM Exports the determinism env BEFORE python/torch import, then starts ComfyUI
REM from the project venv. If your ComfyUI entry script is not main.py, adjust
REM the final line; the env block above it must stay first.
REM ===========================================================================
setlocal
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTHONHASHSEED=0"
set "NVIDIA_TF32_OVERRIDE=0"
set "TOKENIZERS_PARALLELISM=false"

set "COMFY_ROOT=C:\Users\jeffr\Documents\ComfyUI"
set "VENV_PY=%COMFY_ROOT%\.venv\Scripts\python.exe"

cd /d "%COMFY_ROOT%"
"%VENV_PY%" main.py %*
endlocal
