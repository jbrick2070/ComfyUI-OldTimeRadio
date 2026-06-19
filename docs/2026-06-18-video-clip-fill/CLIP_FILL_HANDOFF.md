# Video Clip-Fill + Dynamic-VRAM Frame Budget -- BUILD HANDOFF (2026-06-18)

Self-contained spec for the next (coder) window. Forward-only. Honor GO_FORWARD_PLAN.md
section 2 hard rules. UTF-8, no BOM, SFW. Commit per green chunk; do NOT push unprompted.

## 1. THE BUG (operator-reported, confirmed)
A wan_ti2v episode renders but the final video looks STATIC -- only the green procgen
CRT overlay + caption timer move; the underlying picture is frozen. Proven by pulling
two frames 1.6s apart INSIDE one beat (14.0s vs 15.6s): pixel-identical except the timer.

## 2. ROOT CAUSE (code-grounded)
- The shot calculator is INTACT and correct: `otr_shot_lock.py:314` computes the
  audio-derived per-beat `target_frame_count = round(dur_s*fps)` (~250-285 frames for a
  ~10s beat); `otr_shot_duration_calculator.py` is the helper. wan just never uses it.
- `nodes/_otr_video_engines/eng_wan_ti2v.py` `_floor_length()` (line ~303):
  `requested = min(int(target_frame_count or 17), floor_max)` where `floor_max =
  _TI2V_FLOOR_MAX_FRAMES = 17` (the old 8GB-card safety floor, raisable only by env
  `OTR_WAN_TI2V_MAX_FRAMES`). So wan renders **17 frames (0.68s @25fps)** regardless of
  the ~280-frame beat target. Telemetry: wan @17f @~1472x832 peaked **10.3 GB**.
- `nodes/otr_silent_composite.py` `plan_timeline_segments()` places each beat as a
  "clip" of `target_frame_count` frames and fills a SHORT clip by HOLDING THE LAST FRAME
  (tpad) -> 0.68s motion then ~9s freeze => looks static.
- LTX avoids this: `eng_ltx_video._ltx_frame_length(target_frame_count,...)` +
  `loop_via_reverse` mirror-extends its short render up to target. wan does NOT extend.
- Clips are written by `nodes/_otr_video_engines/_tmp.otr_engine_tmp_mp4()` into
  `episodes/_shared/tmp/` -- which the OH-3 janitor SWEEPS. There is NO persistent
  `clips/` folder (manifest verified: 6 wan clips, frames=17, paths in `_shared/tmp`).

## 3. DESIGN (3-model roundtable: ChatGPT+Gemini+DeepSeek, Claude judged + grounded)
Raw reviews: `scripts/_otr_rt_clips_out.txt`, `scripts/_otr_rt_vram_out.txt`.

**PREDICT the real-frame budget from a zero-cost VRAM read -- NEVER REACT (try-then-OOM).**
Unanimous + emphatic: a CUDA OOM inside ComfyUI's long-lived process corrupts the caching
allocator; catch-and-retry is unreliable without a subprocess restart. OOM is a bug to
AVOID, never a control signal. So:

1. **Cost model per engine/resolution:** `vram_mb ~= fixed_overhead_mb + cost_per_frame_mb * frames`.
   Seed from telemetry: wan_ti2v @1472x832 ~= **7000 MB overhead + ~185 MB/frame**
   (17f -> ~10.3 GB). Env-overridable; refine from observed peaks; new resolutions scale
   by pixel-area ratio (conservative).
2. **The "probe" is `torch.cuda.mem_get_info()`** -> free VRAM. 0 bytes, 0 time. NEVER a
   render-probe.
3. **Math:** `budget = min(free_vram_mb, 14500) * 0.85`;
   `F = min(target_frame_count, floor((budget - overhead) / per_frame))`; snap DOWN to the
   nearest valid 4n+1; clamp UP to a motion floor (17 for wan; 1 for LTX).
4. **Then the engine loop/ping-pong-extends** the real render to `target_frame_count`
   (generalize LTX's `loop_via_reverse`; ping-pong/mirror avoids a hard loop seam).

## 4. WHAT TO BUILD (5 pieces; pure fixes, no shims)
1. **`compute_real_frame_budget(free_vram_mb, target_frame_count, canvas_w, canvas_h,
   engine_name) -> int`** -- shared helper on `MotionEngineBase`
   (`nodes/_otr_video_engines/motion_common.py`). Cost-model + 0.85 margin + 4n+1 snap
   (reuse `wrapper_bridge.quantize_frames_4n1`) + per-engine motion floor. Constants in a
   small seed table (env-overridable). Reads free VRAM via the existing VRAM-levers reader
   (the `free=14775 MB` log path) or `torch.cuda.mem_get_info`.
2. **wan_ti2v:** delete the hard 17-frame cap in `_floor_length`; call
   `compute_real_frame_budget(...)` to get the real-frame count (honors the beat target,
   bounded by live VRAM).
3. **Loop/ping-pong-extend** the short render to `target_frame_count` -- a shared
   `wrapper_bridge` helper generalized from LTX's `loop_via_reverse`, used by wan (and any
   engine whose native render < target). LTX path must stay byte-identical (only refactor
   if the resulting frames are identical; else leave LTX's call site alone).
4. **Persist clips** to `episodes/<ep>/clips/<beat>_<role>_<engine>.mp4` (not swept
   `_shared/tmp`); manifest references the stable path. Diffusion scratch may stay in tmp;
   the FINAL clip lands in `clips/`. Janitor never touches `clips/`/`stills/`/`audio/`.
   (Operator directive: every rendered asset lives under `otr/episodes/<ep>/`, final only
   in `otr/obs/`.)
5. **Composite LOUD underrun guard** in `plan_timeline_segments` (or right after the
   ffprobe frame-count): if a clip's real frames << target_frame_count AND it is not a
   loop-fill, LOUD-WARN (never crash -- no-loud-fail rule) so a future short-clip engine
   is caught, not silently frozen.

## 5. ACCEPTANCE (pass/fail)
- A wan_ti2v all-roles episode: two frames ~1.5s apart WITHIN a beat DIFFER (continuous
  motion fills the beat, not a 0.68s flicker + freeze).
- `episodes/<ep>/clips/` contains the per-beat mp4s after the run (persisted, not swept).
- Render-phase VRAM stays < 14.5 GB (predicted; no OOM) on the 5080.
- LTX path byte-identical (regression: an LTX episode is unchanged).
- Composite emits the LOUD underrun warn when a clip is short.
- Suite + Bug Bible green; `test_audio_byte_identical` stays green.

## 6. GROUNDING (OTR already has the pieces -- not a new subsystem)
`quantize_frames_4n1` exists (wan calls it); VRAM peak + free telemetry exists
(`wan_ti2v VRAM render-phase peak 10277 MB`, `free=14775 MB`); `MotionEngineBase`
(motion_common.py) is the shared home; LTX `loop_via_reverse` is the loop pattern.

## 7. RULES
Single resident heavy <= 14.5 GB (host NVML); 100% local; determinism (same input -> same
frame count within a run; use max observed peak, conservative extrapolation); every
in-render fallback LOUD; UTF-8 no BOM; SFW. Edit the real source-of-truth workflow JSON in
the SAME change if any node/widget changes. Run the suite + Bug Bible after every code
change. Commit per green chunk; do NOT push unprompted. Update GO_FORWARD_PLAN.md + the
otr-build-tracker on wrap.
