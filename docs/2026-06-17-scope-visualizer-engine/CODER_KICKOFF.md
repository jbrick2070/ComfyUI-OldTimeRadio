# Coder kickoff -- Scope Visualizer Engine + verified-model 120w smoke sweep

START THIS ONLY AFTER you finish your current in-flight coding task and it is
committed + pushed green. Serialize via docs/GO_FORWARD_PLAN.md (one coder in the
code at a time).

## Read first (source of truth)

1. `docs/2026-06-17-scope-visualizer-engine/CODING_PLAN.md` -- the full spec (this
   is the build).
2. `docs/2026-06-17-scope-visualizer-engine/roundtable/pass01_judgment.md` -- the
   grounded decisions and why.
3. `CLAUDE.md` sections 0 (workflow JSON is the single source of truth), 3 (coding
   discipline), 4-5 (reset + headless boot), 6 (git policy).

## TASK 1 -- build the visualizer engine (per the plan)

Implement `engine_id = "visualizer"`, 16:9 only, as a NEW per-beat video-engine
adapter. Non-negotiables (all grounded in the plan):

- **Separation invariant (HARD):** do NOT import or call the floor node
  (`OTR_SignalLostVideo` / `video_engine.py render()`), do NOT flip its
  `draw_scopes`, do NOT touch `OTR_SceneAwareScopes` / `OTR_PostUpscaleProcgenBlend`.
  The engine is INERT unless a role's dropdown selects `visualizer`. When not
  selected, NOTHING about the render changes.
- **Mechanism:** EXTRACT the pure per-frame draw routines (ring / particles / grid
  / waveform / freq bars / CRT post) from `video_engine.py` into a torch-free
  `nodes/_otr_shared/scope_draw.py` that BOTH the floor and the engine import. The
  floor must call the same code with zero behavior change -- prove it with the
  floor's existing tests. (If extraction is too invasive for v1, the engine COPIES
  the routines into its own module -- still zero coupling.)
- **v1 = ONE engine** `visualizer` = full-colour procedural look. NO mode widget
  (the dropdown selects the engine; a single engine can't read a second-level
  mode). The strict green-only look is a SEPARATE follow-up engine
  `visualizer_green` AFTER v1 validates -- do NOT build it now. No new geometry.
- **render_clip (do ALL steps -- pass-02 caught the crashes):** resolve
  `audio_ref` path -> DECODE to numpy+sr via `soundfile` (NOT torchaudio) ->
  **mixdown to mono** `if audio_np.ndim>1: audio_np=audio_np.mean(axis=1)` ->
  `total_frames=int(timing.target_frame_count)`, `fps=25` ->
  `volume,freqs,waves=_analyze_audio_np(...)` -> **`signal,trig,loss=_dual_ema(volume)`**
  (REQUIRED; the draw helpers read `env["signal"]`) -> per frame build bounded
  windows `freqs[max(0,fi-_TRAIL_N+1):fi+1]` + an `env` dict
  (`fi/fps/key/vol/signal/loss/trig`) -> paint full-16:9 -> `_encode_silent_mp4(
  frames,total_frames,out_path,w,h,25,ffmpeg)` (w/h from `request["canvas"]`,
  ffmpeg path from assert_usable). Return `{out_path, frame_count}`. `has_audio`
  MUST be False (`test_audio_byte_identical` stays green).
- **Story-DNA (mood/palette/role) is DEFERRED** -- v1 is a faithful resurrection of
  today's green/cyan/amber + vol/signal/loss reactivity. Do not add tint/speed
  params to the shared routines in v1 (coupling risk to the floor's colours).
- **assert_usable:** fail LOUD, `fallback_engine=None` (547671d): flag gate ->
  ffmpeg resolvable -> non-empty audio_ref. No NVML/weights/node gate.
- **Registry:** `family="abstract"`, `roles=("announcer_visual","music_visual",
  "character_video")`, `default_roles=()`, `commercial_clean=True`,
  `requires_flag="OTR_ENABLE_VISUALIZER"` (default-OFF while dark),
  `render_aspect="wide"` (duck-typed, like eng_ltx_av). The CAPABILITIES["visualizer"]
  row already exists -- do not duplicate it.
- **Story-DNA mapping (minimal v1):** mood term -> one animation-speed scalar;
  `visual_palette` -> one deterministic phosphor-tint lookup seeded from
  request_seed (validate-later branch). No per-mood decision tree, no caption.
- **Wiring (CLAUDE.md sec 0):** any node/widget/wiring change lands IN
  `workflows/otr_scifi_16gb_full.json` in the SAME commit; re-validate with
  `OTR_WorkflowValidator` + JSON round-trip + link/widget audit.
- **Verify-at-build:** confirm `otr_video_director.py` builds the per-role combo
  from `validated_engine_names()`; confirm `soundfile` imports in the venv.

Tests (write them; run the FULL suite + Bug Bible after EVERY change, don't wait
to be asked): registration/roles/family/required_inputs; assert_usable LOUD paths;
render_clip produces a silent 25fps 16:9 mp4 of the expected frame count;
canonicalize shape + engine_id/family; determinism (same audio+seed -> byte-
identical); role_compat for all 3 roles; floor-unchanged regression; naming
conventions (no "dummy", UTF-8 no BOM). Commit + push per green chunk to
`v2.0-alpha`; verify HEAD==origin, no 0-byte/BOM, AST parse.

GPU validation gate: once it renders end-to-end through OTR_VideoRenderBatch + mux
with audio byte-identical, ADD `"visualizer"` to `VALIDATED_ENGINES` (and decide
default-ON for accessibility) so it appears in the per-role dropdowns.

## TASK 2 -- after Task 1 is green: verified-model 120-word smoke sweep

Goal: a set of RANDOM 120-word full-pipeline smokes across the IMAGE and VIDEO
models that are currently VERIFIED, to confirm none regressed.

- Enumerate the verified sets PROGRAMMATICALLY (do not hardcode): video =
  `validated_engine_names()` (currently ltx_video, ltx_av_music, ltx_av_talk,
  humo, humo_1.7B, humo_1.7B_169, humo_14B_169, wan_ti2v); image = the
  OTR_ImageDirector validated set (read its INPUT_TYPES / the image registry).
- Use the existing full-smoke harness (`scripts/queue_smoke.py` + `otr_api.py`),
  parameterized to 120 words with a RANDOM story seed per run (OS entropy, not the
  fixed widget seed -- per the true-randomization rule). Load the REAL
  `workflows/otr_scifi_16gb_full.json`, never a generated/stale copy.
- Sweep a representative matrix: each verified video engine once (forced via the
  force-map / per-role dropdown), and each verified image engine once for the
  still path. Include one run with `visualizer` selected for all three roles (the
  low-VRAM lane).
- RESET before EACH headless run (CLAUDE.md sec 4: selective CIM kill by
  CommandLine + port 8000 free + VRAM back to ~1.5 GB baseline). Use the watchdog
  (`scripts/otr_render_watchdog.ps1`) for long legs; UTF-8 boot env
  (PYTHONUTF8=1). Background long jobs to a log and poll (the ~60s MCP ceiling).
- For each run capture: engine_id, pass/fail, render time, VRAM peak vs ceiling,
  and `test_audio_byte_identical` result. Write a summary table to
  `docs/2026-06-17-scope-visualizer-engine/SMOKE_SWEEP_RESULTS.md`. Fail LOUD on
  any regression; do not paper over with fallbacks.

Hand back a short report: what passed, what regressed, the visualizer lane's
VRAM/time vs the model lanes.
