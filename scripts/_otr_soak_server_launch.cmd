@echo off
rem v1.4 capstone soak server launch -- verified recipe (docs/VIDEO_BUILD_HANDOFF.md)
rem usage: launch_soak_server.cmd <logfile> [HUMO=0]
set HF_HOME=C:\ComfyUI-Models\huggingface
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONHASHSEED=0
set NVIDIA_TF32_OVERRIDE=0
set TOKENIZERS_PARALLELISM=false
set OTR_CAST_SEED=42
set OTR_STYLE_SEED=42
rem OUTPUT HYGIENE (operator directive 2026-06-09): every temp/scratch write
rem (tempfile.* in-process AND ffmpeg children) stays UNDER output\otr\tmp --
rem nothing lands outside the output\otr\ tree.
set OTR_TMP=C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output\otr\tmp
if not exist "%OTR_TMP%" mkdir "%OTR_TMP%"
set TEMP=%OTR_TMP%
set TMP=%OTR_TMP%
set OTR_GPU_LEASE_DIR=%OTR_TMP%
if "%2"=="HUMO=0" (
  set OTR_ENABLE_HUMO=
  echo [launch] heavy engines OFF ^(floor leg^)
) else (
  set OTR_ENABLE_HUMO=1
)
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
  --port 8000 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI ^
  > "%1" 2>&1
