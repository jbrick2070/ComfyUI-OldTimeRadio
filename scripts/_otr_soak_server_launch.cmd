@echo off
rem v1.4 capstone headless server launch -- verified recipe (docs/VIDEO_BUILD_HANDOFF.md)
rem usage: _otr_soak_server_launch.cmd <logfile> [FLOOR | HUMO | LTX | WAN]
rem   (default)  heavy engines OFF. Video model selection belongs to the
rem              workflow dropdown/profile, not an implicit launcher switch.
rem   FLOOR      heavy engines OFF (same as default). NOTE: the old
rem              token "HUMO=0" NEVER matched -- cmd splits arguments on "=",
rem              so %2 arrived as "HUMO" and the else-branch enabled HuMo
rem              (2026-06-10 marathon catch).
rem   HUMO       explicit legacy HuMo lane for bakeoffs/single-engine probes.
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
rem OTR_SCIFI_NEWS_PRO_SEED is pinned alongside CAST/STYLE under C7 (r3 ruling B8): the
rem fable2 frame-card/stance deal + voice tie-breaks reproduce only when it is
rem fixed, so a snapshot replay of a fable2-family bank is byte-stable. Cleared
rem in the else branch so production rolls fresh OS entropy, same as CAST/STYLE.
if defined OTR_C7 (
  set OTR_CAST_SEED=42
  set OTR_STYLE_SEED=42
  set OTR_SCIFI_NEWS_PRO_SEED=42
  echo [launch] C7 mode: OTR_CAST_SEED=42 OTR_STYLE_SEED=42 OTR_SCIFI_NEWS_PRO_SEED=42 ^(byte-identity^)
) else (
  set OTR_CAST_SEED=
  set OTR_STYLE_SEED=
  set OTR_SCIFI_NEWS_PRO_SEED=
  rem ECHOED IN BOTH BRANCHES ON PURPOSE. Silence here is what let a stale
  rem OTR_CAST_SEED=42 ride an entire bake-off unnoticed on 2026-08-22: all four
  rem episodes cast GULLIVER REEVES, the operator spotted it by watching them,
  rem and the only trace was one writer log line nobody reads. A leg log must
  rem state which mode it ran in whether or not anything was pinned.
  echo [launch] production seeds: cast/style/scifi-news UNSET ^(fresh OS entropy per episode^)
)
rem Bake-off source-snapshot manifest (r3 ruling B8): the process-wide frozen-
rem source map keyed by BASE bank. Passed in the caller's own process env
rem (auditable, not a hidden per-leg hook file per the note below); echoed here
rem for the server log. Unset => every bank sources live (RSS / random / custom).
if defined OTR_SOURCE_SNAPSHOT_MANIFEST echo [launch] source-snapshot manifest: %OTR_SOURCE_SNAPSHOT_MANIFEST%
rem Hydrate per-user secrets a detached shell may not have inherited (the DC
rem service env snapshot predates setx -- known gotcha). Value is NEVER echoed.
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY','User')"`) do set OPENROUTER_API_KEY=%%k
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OTR_GOOGLE_API_KEY','User')"`) do set OTR_GOOGLE_API_KEY=%%k
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')"`) do set GEMINI_API_KEY=%%k
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('GOOGLE_API_KEY','User')"`) do set GOOGLE_API_KEY=%%k
rem Hydrate OTR_BLENDER_EXE from the User env too (mesh_stage's pinned portable
rem Blender; a detached cmd doesn't inherit setx User env -- the same gotcha).
rem Without it mesh_stage fails closed missing_model -> falls back to
rem still_parallax (the 2026-06-12 catch: PASS but engine NOT in the trace).
for /f "usebackq delims=" %%b in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OTR_BLENDER_EXE','User')"`) do set OTR_BLENDER_EXE=%%b
rem Hydrate the two image-engine WEIGHT PATHS from the User env for the same
rem reason (2026-08-26). lumina_image and flux2_klein read an ABSOLUTE path out
rem of these and fail closed on it -- assert_usable is `os.getenv(...)` plus
rem `os.path.isfile(...)` with NO folder_paths fallback
rem (_otr_image_engines/lumina_image.py:405, flux2_klein.py:311). The image
rem dispatcher does not degrade either: it raises ImageRenderError "NO FALLBACK"
rem (otr_image_gen_dispatcher.py:1462), so an unhydrated boot does not lose one
rem picture -- it KILLS the episode. Both weights are on disk; only the variable
rem naming them goes missing, which is exactly the OTR_BLENDER_EXE gotcha above.
rem A boot that inherits them already simply re-sets the same value.
rem z_image_turbo is deliberately NOT here: it ranks and auto-discovers its own
rem unet (z_image_turbo.py:197-249), so it needs no variable to survive.
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OTR_LUMINA_CKPT','User')"`) do set OTR_LUMINA_CKPT=%%m
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('OTR_FLUX2_KLEIN_CKPT','User')"`) do set OTR_FLUX2_KLEIN_CKPT=%%m
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
rem The canonical wrapper chooses a free local port per leg. Direct/manual
rem launches keep the historical port as a harmless default.
if not defined OTR_HEADLESS_PORT set OTR_HEADLESS_PORT=8000
echo [launch] OTR headless port %OTR_HEADLESS_PORT%
if /i "%2"=="FLOOR" (
  set OTR_ENABLE_HUMO=
  echo [launch] heavy engines OFF ^(floor leg^)
) else if /i "%2"=="HUMO" (
  set OTR_ENABLE_HUMO=1
  echo [launch] HUMO lane: explicit OTR_ENABLE_HUMO=1
) else if /i "%2"=="LTX" (
  set OTR_ENABLE_HUMO=
  set OTR_ENABLE_LTX_VIDEO=1
  rem OTR_ENABLE_LTX_AV added in lane 7 (2026-08-11). This token is THE LTX
  rem boot lane and it was enabling only ONE of the two LTX engines, so the
  rem audio-in lane could not be smoked on the boot it declares without the
  rem operator exporting a flag by hand -- and a boot lane you have to
  rem supplement by hand is not a boot lane. Both LTX engines stay DEFAULT-OFF
  rem on every other token; this one turns them on together, which is what its
  rem name has always promised.
  set OTR_ENABLE_LTX_AV=1
  echo [launch] LTX lane: Sage-free boot, OTR_ENABLE_LTX_VIDEO=1, OTR_ENABLE_LTX_AV=1, HuMo OFF
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
  set OTR_ENABLE_HUMO=
  echo [launch] heavy engines OFF ^(default no-lane^)
)
rem Canonical headless boots do not consume hidden per-leg env hook files.
rem Harnesses that need a special env must pass it explicitly in their own
rem process environment so the server log and command line remain auditable.
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
rem Pinned-host-memory clamp (S8 boot contracts, 2026-08-11). The OTHER half of
rem the HuMo diet: --reserve-vram alone does not reproduce the measured 13.06
rem GiB envelope. Until this line existed, --disable-pinned-memory appeared in
rem ZERO non-doc files repo-wide, so a profile that "configured" the diet
rem clamped exactly one of its two knobs and the other was documentation. See
rem nodes/_otr_shared/boot_contracts.py, which names the contracts, and which
rem PROVES them against comfy.cli_args on the running server rather than
rem against the profile text -- a check that reads the same config the launcher
rem was meant to honour cannot tell "applied" from "written down". Default
rem UNSET = no clamp = byte-identical to every prior boot.
set _OTR_PINNED=
if defined OTR_HEADLESS_DISABLE_PINNED set _OTR_PINNED=--disable-pinned-memory
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
rem LOG ROTATION (2026-08-04): this line used to redirect with `>`, so every
rem reboot TRUNCATED the previous run's server log. That destroyed the only
rem record of the 2026-08-03 22:11 still-unmaterialized failure when the 23:06
rem relaunch came up. Move any existing log aside first, then APPEND.
rem
rem The caller's log path is unchanged -- eight harnesses read exactly %1.
rem If the rotation FAILS (a locked log), we append to the old file and say so
rem rather than truncating it: losing evidence must be loud, and it must never
rem kill a boot.
rem SUCCESS IS TESTED BY OUTCOME, NOT BY EXIT CODE. `if errorlevel N` is a
rem SIGNED >= test, and `powershell -File <missing path>` exits -196608 -- a
rem NEGATIVE code, which `if errorlevel 1` reads as SUCCESS. Trusting it meant
rem that a missing rotate script would silently truncate the prior log: the very
rem bug this rotation exists to prevent, via its own missing-dependency path.
rem Asking "is the old log still there?" is immune to every exit-code quirk.
set _OTR_ROT_FAILED=
if exist "%~1" (
  if exist "%~dp0otr_rotate_log.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0otr_rotate_log.ps1" "%~1"
  ) else (
    echo [launch] rotate helper missing at "%~dp0otr_rotate_log.ps1"
  )
)
rem Still present => rotation did not happen, whatever the reason.
if exist "%~1" set _OTR_ROT_FAILED=1
if not defined _OTR_ROT_FAILED type nul > "%~1"
if defined _OTR_ROT_FAILED echo [launch] LOG ROTATION FAILED -- prior log could not be moved; this run is APPENDED below and the earlier content is preserved above.>> "%~1"
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
  C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py ^
  --port %OTR_HEADLESS_PORT% --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI ^
  --output-directory %OTR_REAL_OUTPUT% ^
  --extra-model-paths-config "%~dp0_otr_headless_model_paths.yaml" ^
  --disable-metadata ^
  %_OTR_RESERVE% ^
  %_OTR_PINNED% ^
  %_OTR_VERBOSE% ^
  >> "%~1" 2>&1
rem --disable-metadata (2026-06-12): the core V3 SaveGLB node (mesh_stage) does
rem `if cls.hidden.prompt is not None:` -- but cls.hidden is None when the node
rem runs via OTR's in-process wrapper_bridge (ComfyUI only injects the hidden
rem context in its own prompt executor) -> AttributeError -> mesh_stage fell
rem back to still_parallax. --disable-metadata skips that metadata block (OTR
rem carries its own ledger/manifest; embedded workflow JSON is unused). Content
rem of saved mp4/wav is unchanged, so audio byte-identity holds.
