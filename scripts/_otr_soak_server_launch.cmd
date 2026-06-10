@echo off
rem v1.4 capstone headless server launch -- verified recipe (docs/VIDEO_BUILD_HANDOFF.md)
rem usage: _otr_soak_server_launch.cmd <logfile> [HUMO=0 | LTX | WAN]
rem   (default)  humo:5 production path (OTR_ENABLE_HUMO=1)
rem   HUMO=0     heavy engines OFF (the still-floor soak leg)
rem   LTX        Sage-free boot lane: LTX opt-in ON, HuMo OFF (BUG-070)
rem   WAN        Wan i2v opt-in ON, HuMo OFF
set HF_HOME=C:\ComfyUI-Models\huggingface
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONHASHSEED=0
set NVIDIA_TF32_OVERRIDE=0
set TOKENIZERS_PARALLELISM=false
set OTR_CAST_SEED=42
set OTR_STYLE_SEED=42
rem OUTPUT UNIFICATION (operator directive 2026-06-09): even headless, ALL
rem outputs -- episodes, portraits, finals, EVERYTHING -- land in the REAL
rem output folder the operator watches. --output-directory pins ComfyUI's
rem folder_paths; OTR_OUTPUT_DIR pins the OTR writers to the same tree.
set OTR_REAL_OUTPUT=C:\Users\jeffr\Documents\ComfyUI\output
set OTR_OUTPUT_DIR=%OTR_REAL_OUTPUT%
set OTR_OBS_DIR=%OTR_REAL_OUTPUT%\otr\obs
rem OUTPUT HYGIENE: every temp/scratch write (tempfile.* in-process AND ffmpeg
rem children) stays UNDER output\otr\tmp -- nothing outside the otr tree.
set OTR_TMP=%OTR_REAL_OUTPUT%\otr\tmp
if not exist "%OTR_TMP%" mkdir "%OTR_TMP%"
set TEMP=%OTR_TMP%
set TMP=%OTR_TMP%
set OTR_GPU_LEASE_DIR=%OTR_TMP%
if /i "%2"=="HUMO=0" (
  set OTR_ENABLE_HUMO=
  echo [launch] heavy engines OFF ^(floor leg^)
) else if /i "%2"=="LTX" (
  set OTR_ENABLE_HUMO=
  set OTR_ENABLE_LTX_VIDEO=1
  echo [launch] LTX lane: Sage-free boot, OTR_ENABLE_LTX_VIDEO=1, HuMo OFF
) else if /i "%2"=="WAN" (
  set OTR_ENABLE_HUMO=
  set OTR_ENABLE_WAN_I2V=1
  set OTR_WAN_I2V_CKPT=C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  echo [launch] WAN lane: OTR_ENABLE_WAN_I2V=1, HuMo OFF
) else (
  set OTR_ENABLE_HUMO=1
)
rem Marathon per-leg env injection: the playlist runner writes this file with
rem extra `set X=Y` lines (engine override map, opt-in flags) and deletes it
rem after the leg. Absent file = no-op.
if exist "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd" call "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd"
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
  --port 8000 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI ^
  --output-directory %OTR_REAL_OUTPUT% ^
  > "%1" 2>&1
