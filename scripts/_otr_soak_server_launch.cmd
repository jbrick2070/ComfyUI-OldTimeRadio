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
rem UTF-8 stdio (2026-06-12): a detached cmd inherits the cp1252 console codec,
rem so ComfyUI's logger crashes the instant OTR prestartup prints an emoji
rem (UnicodeEncodeError on the U+2705/U+2713 glyphs) -> boot dies ~13s, exit 1,
rem "SERVER DID NOT COME UP" failure. Desktop used to set this for us; the
rem v2 install move dropped it. Force UTF-8 mode for stdio + filesystem.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
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
rem Hydrate OTR_BLENDER_EXE from the User env too (mesh_stage's pinned portable
rem Blender; a detached cmd doesn't inherit setx User env -- the same gotcha).
rem Without it mesh_stage fails closed missing_model -> falls back to
rem still_parallax (the 2026-06-12 catch: PASS but engine NOT in the trace).
for /f "usebackq delims=" %%b in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OTR_BLENDER_EXE','User')"`) do set OTR_BLENDER_EXE=%%b
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
  rem GO_FORWARD 4A (2026-06-14): the 8GB-tier Wan2.2 TI2V-5B engine. Enabling
  rem BOTH Wan engines is what the full --acceptance sweep preflight requires
  rem (M3: every registered core Wan engine's enable flag must be 1). The 5B
  rem REQUIRES the Wan2.2 VAE (M8), not the 2.1 VAE.
  set OTR_ENABLE_WAN_TI2V=1
  set OTR_WAN_TI2V_CKPT=C:\ComfyUI-Models\diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf
  set OTR_WAN_TI2V_VAE_NAME=wan2.2_vae.safetensors
  echo [launch] WAN lane: OTR_ENABLE_WAN_I2V=1 + OTR_ENABLE_WAN_TI2V=1, HuMo OFF
) else (
  set OTR_ENABLE_HUMO=1
)
rem Marathon per-leg env injection: the playlist runner writes this file with
rem extra `set X=Y` lines (engine override map, opt-in flags). CONSUME-ONCE
rem (root-cause fix 2026-07-02): a crashed leg used to ORPHAN this hook, and
rem the orphan then silently forced its env (the caught case:
rem OTR_FORCE_ENGINE_MAP=*=humo + OTR_ENABLE_HUMO_HOSTS=1) onto EVERY later
rem headless boot -- an all-ltx probe rendered all-HuMo. One hook = ONE boot:
rem echo it LOUD, call it, DELETE it. Absent file = no-op. (All writers --
rem otr_run_leg.ps1 / _otr_soak_marathon.py / otr_3d_quick_tests.ps1 --
rem re-write it per leg, so consume-once matches the intended lifecycle.)
if exist "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd" (
  echo [launch] consuming marathon extra-env hook ^(one boot only^):
  type "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd"
  call "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd"
  del /q "%~dp0_otr_soak_capstone_results\_marathon_extra_env.cmd"
)
rem Optional %3 DEBUG (CS-4 diagnosis, 2026-06-11): comfy model-management
rem logs at DEBUG show per-model partial load/unload sizes -- the residency
rem attribution evidence. Same recipe otherwise.
set _OTR_VERBOSE=
if /i "%3"=="DEBUG" set _OTR_VERBOSE=--verbose DEBUG
rem Optional VRAM clamp (Wan TI2V-5B low-VRAM bakeoff, 2026-06-27): set
rem OTR_HEADLESS_RESERVE_VRAM_GB to reserve that many GB away from model loading,
rem so a 16GB card simulates an 8GB/6GB card and ComfyUI's allocator forces the
rem same aggressive offload / sysmem spill a low-VRAM user hits. Default UNSET =
rem no clamp = byte-identical to every prior boot (no other lane sets it).
set _OTR_RESERVE=
if defined OTR_HEADLESS_RESERVE_VRAM_GB set _OTR_RESERVE=--reserve-vram %OTR_HEADLESS_RESERVE_VRAM_GB%
rem CUSTOM NODES (2026-06-12, Desktop-v2 install move): the install root's
rem custom_nodes holds ONLY the OldTimeRadio junction -- the wrapper packs
rem (ComfyUI-LTXVideo, KJNodes, VideoHelperSuite, kokorotts, ...) live in
rem Documents\ComfyUI\custom_nodes, mapped by the Desktop app's
rem extra_models_config.yaml. Headless boots MUST pass a yaml or every
rem ltx_video render falls to the floor (WrapperNodeMissing:
rem LTXVImgToVideoConditionOnly -- the 3D quick-smoke catch). We pass OUR
rem headless copy (_otr_headless_model_paths.yaml) because the Desktop yaml's
rem desktop_extensions entry points at the dead v1 install path and crashes
rem main.py's prestartup scan (FileNotFoundError).
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
  --port 8000 --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI ^
  --output-directory %OTR_REAL_OUTPUT% ^
  --extra-model-paths-config "%~dp0_otr_headless_model_paths.yaml" ^
  --disable-metadata ^
  %_OTR_RESERVE% ^
  %_OTR_VERBOSE% ^
  > "%1" 2>&1
rem --disable-metadata (2026-06-12): the core V3 SaveGLB node (mesh_stage) does
rem `if cls.hidden.prompt is not None:` -- but cls.hidden is None when the node
rem runs via OTR's in-process wrapper_bridge (ComfyUI only injects the hidden
rem context in its own prompt executor) -> AttributeError -> mesh_stage fell
rem back to still_parallax. --disable-metadata skips that metadata block (OTR
rem carries its own ledger/manifest; embedded workflow JSON is unused). Content
rem of saved mp4/wav is unchanged, so audio byte-identity holds.
