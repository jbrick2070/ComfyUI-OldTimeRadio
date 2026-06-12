@echo off
rem v1.4 capstone headless server launch -- verified recipe (docs/VIDEO_BUILD_HANDOFF.md)
rem usage: _otr_soak_server_launch.cmd <logfile> [FLOOR | LTX | WAN]
rem   (default)  humo:5 production path (OTR_ENABLE_HUMO=1)
rem   FLOOR      heavy engines OFF (the still-floor soak leg). NOTE: the old
rem              token "HUMO=0" NEVER matched -- cmd splits arguments on "=",
rem              so %2 arrived as "HUMO" and the else-branch enabled HuMo
rem              (2026-06-10 marathon catch).
rem   LTX        Sage-free boot lane: LTX opt-in ON, HuMo OFF (BUG-070)
rem   WAN        Wan i2v opt-in ON, HuMo OFF
set HF_HOME=C:\ComfyUI-Models\huggingface
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONHASHSEED=0
set NVIDIA_TF32_OVERRIDE=0
set TOKENIZERS_PARALLELISM=false
rem C7 byte-identity seeds (BUG-LOCAL-269/270): ONLY pinned when the caller
rem sets OTR_C7=1 (regression/baseline runs). Production headless runs leave
rem these UNSET so every episode rolls a fresh OS-entropy cast + style --
rem pinning them here was why every run cast GULLIVER REEVES with the same
rem red-wash style (2026-06-12 operator catch).
if defined OTR_C7 (
  set OTR_CAST_SEED=42
  set OTR_STYLE_SEED=42
  echo [launch] C7 mode: OTR_CAST_SEED=42 OTR_STYLE_SEED=42 ^(byte-identity^)
) else (
  set OTR_CAST_SEED=
  set OTR_STYLE_SEED=
)
rem Hydrate per-user secrets a detached shell may not have inherited (the DC
rem service env snapshot predates setx -- known gotcha). Value is NEVER echoed.
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY','User')"`) do set OPENROUTER_API_KEY=%%k
rem OUTPUT UNIFICATION (operator directive 2026-06-09): even headless, ALL
rem outputs -- episodes, portraits, finals, EVERYTHING -- land in the REAL
rem output folder the operator watches. --output-directory pins ComfyUI's
rem folder_paths; OTR_OUTPUT_DIR pins the OTR writers to the same tree.
set OTR_REAL_OUTPUT=C:\Users\jeffr\Documents\ComfyUI\output
set OTR_OUTPUT_DIR=%OTR_REAL_OUTPUT%
set OTR_OBS_DIR=%OTR_REAL_OUTPUT%\otr\obs
rem OUTPUT HYGIENE (OH-2, output-tree contract 2026-06-11): every temp/scratch
rem write (tempfile.* in-process AND ffmpeg children) stays UNDER the reserved
rem system tier episodes\_shared\tmp -- the otr top level is EXACTLY
rem episodes + obs. The janitor sweeps stale entries here (OH-3).
set OTR_TMP=%OTR_REAL_OUTPUT%\otr\episodes\_shared\tmp
if not exist "%OTR_TMP%" mkdir "%OTR_TMP%"
set TEMP=%OTR_TMP%
set TMP=%OTR_TMP%
set OTR_GPU_LEASE_DIR=%OTR_TMP%
if /i "%2"=="FLOOR" (
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
rem Optional %3 DEBUG (CS-4 diagnosis, 2026-06-11): comfy model-management
rem logs at DEBUG show per-model partial load/unload sizes -- the residency
rem attribution evidence. Same recipe otherwise.
set _OTR_VERBOSE=
if /i "%3"=="DEBUG" set _OTR_VERBOSE=--verbose DEBUG
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
  --port 8000 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI ^
  --output-directory %OTR_REAL_OUTPUT% %_OTR_VERBOSE% ^
  > "%1" 2>&1
