# HuMo r3-HARDENED -- STANDALONE bakeoff harness (Codex + AntiGravity CONVERGED; Claude judge)

Build-ready. Diagnostic ONLY: it adds NEW files + ONE sibling node package and edits NOTHING in the
OTR pack, eng_humo.py, or workflows/otr_scifi_16gb_full.json. Measures quality/speed/VRAM so the
operator eyeballs clips + metrics BEFORE any production change is ever proposed.

## 1. ISOLATION -- the two-stage eviction without touching production (RESOLVED)
The two-stage (conditioning -> evict umt5+whisper -> sampler) can't be two `/prompt` calls: ComfyUI has
no Save/LoadConditioning to round-trip embeddings across sessions. SOLUTION = a SEPARATE sibling
custom-node package, auto-loaded by ComfyUI, holding ONE diagnostic node:
- NEW `custom_nodes/otr_bakeoff_helper/__init__.py` registering `OTR_BakeoffReclaim` -- a passthrough
  node that calls `nodes._otr_video_engines.wrapper_bridge.reclaim_idle_models(reason="humo bakeoff
  pre-sampler")` in its execute, then forwards its input. Wire it in the standalone graph BETWEEN the
  conditioning/encoder outputs and the `KSampler` model/latent inputs so topological order forces the
  TE/whisper eviction mid-graph, in ONE `/prompt`. The OTR pack `__init__.py` is NOT edited; this
  sibling package is removable and never ships in production.
NEW SCRIPTS (mirror the LTX-AV bakeoff): `scripts/build_humo_bakeoff_workflow.py` (emits the standalone
per-leg HuMo graph JSON under scripts/, reusing HuMoEngine._build_graph READ-ONLY for node templates) +
`scripts/run_humo_bakeoff.py` (boot-per-leg headless, external VRAM read, manifest, clips). Cold-import
`wrapper_bridge` by file path (LTX pattern, no `nodes` package side effects).

## 2. LEGS (minimal; fixed still+audio+seed = the LTX pair c02_466a19906ccb.png + c02_b002_line.wav)
(i)  `humo_14B_169` single-graph, 6-step distill, shift 8, cfg 1.0 = the 5/21 baseline (today's path --
     the "did we lose quality" control).
(ii) `humo_14B_169` TWO-STAGE (OTR_BakeoffReclaim between conditioning + KSampler), same settings = the
     CANDIDATE.
(iii)`humo_1.7B_169` (16:9 -- NOT portrait humo_1.7B; aspect must match the candidate) = current-ship control.
(iv) SENTINEL (no reboot): load a representative LTX-AV + Whisper render FIRST, then run leg (ii) in the
     SAME resident session -> the PRODUCTION-TRUE peak (clean boot-per-leg hides cross-engine residency).
CUT the no-LoRA ~25-step reference (not a promotion candidate; not needed to answer the gate).

## 3. METRICS + MANIFEST
- VRAM: render-window peak read EXTERNALLY by the harness (nvidia-smi poll / watchdog) -- so NO
  production VramPeakProbe / eng_humo edit is needed for the diagnostic. HARD-GATE the heavy engine at
  13500 MB; separately report the 14500 MB absolute box ceiling (do NOT let the copied LTX default to 14.5).
- SPEED: s/it + wall-clock per leg.
- QUALITY: side-by-side clips -> `otr/episodes/_bakeoff_humo/<leg>.mp4` for the eyeball (primary). PLUS
  a dependency-free BLUE-CAST DELTA in pure PIL+numpy (mean R/G/B of rendered frames vs the source
  still) -- directly tests the de-blue concern, needs no face libs. Face-detect / mouth-landmark / lip
  SSIM are SOFT-GATED: preflight cv2/dlib/mediapipe/ffprobe, mark each metric "skipped" if absent
  (they are NOT in requirements.txt) -- never fail the run.
- MANIFEST (fail-loud, mirror run_ltx_av_q_bakeoff.py:342-414): ASSERT (not just record) the resolved
  unet filename / LoRA present-absent / ModelSamplingSD3 shift=8 / steps / cfg / seed / dims / 4n+1
  frame length / terminal / output path AND the engine id that ACTUALLY ran (the LTX #1 risk).

## 4. OUTPUT encoding (byte-match production)
Standalone graph terminal = `SaveImage` (lossless PNG frame batch); the harness compiles frames into a
SILENT mp4 via the production `wrapper_bridge.encode_frames_to_silent_mp4` -- NOT CreateVideo/SaveVideo
(legacy render_humo_batch carried audio). Assets straight to otr/episodes/_bakeoff_humo/ (never tmp).

## 5. BOOT / RESET (CLAUDE.md S4)
Boot the server WITHOUT the `FLOOR` arg (FLOOR clears OTR_ENABLE_HUMO -> HuMoEngine.assert_usable fails
closed); manifest OTR_ENABLE_HUMO + OTR_HUMO_* + resolved class names. `reset_box` must SELECTIVELY CIM-kill
(by command line, excl. current PID): prior `run_humo_bakeoff.py` trees + the `cmd.exe` shell spawned by
SOAK_LAUNCH_CMD + the Comfy server + port 8000 owners -- NEVER a blanket python kill. Confirm port 8000
empty + nvidia-smi at desktop baseline before each boot. `--dry-validate` mode: build every leg JSON +
manifests, render nothing (mirror the LTX rail).

## 6. PROVABLY UNTOUCHED
eng_humo.py, otr_scifi_16gb_full.json, and the OTR pack __init__.py are NOT edited. Diagnostic output =
clips + manifest + a metrics table for the operator eyeball. The two production-candidate changes
(two-stage split + VramPeakProbe in eng_humo, profile flip) are DEFERRED to a later wiring round, gated
on the operator liking the bakeoff clips.

## Judgment log
ACCEPTED (grounded): sibling otr_bakeoff_helper/OTR_BakeoffReclaim for mid-graph eviction; control =
humo_1.7B_169; gate 13500 (report 14500); no-FLOOR boot; AV-resident sentinel leg; SaveImage->silent
encoder; fail-loud asserting manifest; orphan-bakeoff + cmd.exe kill in reset; blue-cast via PIL+numpy;
soft-gated face metrics; --dry-validate; cold-import wrapper_bridge. CUT (both): no-LoRA 25-step leg.
DEFERRED: all production edits (engine two-stage, profile flip) -> next round, operator-gated.
