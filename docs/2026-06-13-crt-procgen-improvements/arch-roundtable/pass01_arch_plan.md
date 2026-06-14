# v2 SCENE-AWARE SCOPES -- hardened architecture (arch round 1 synthesized)

Synthesized from arch-round-1 (gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro), grounded vs
the real manifest/blend/floor code. Invariants: no audio-spine touch; green-only;
deterministic; Pillow-only/CPU; no new model; the procgen floor's gap-fill/credits/
green-blend roles UNCHANGED; title card + CRT stay in the floor.

## The shape (4 parts)
1. **Scope-module refactor (PREREQUISITE = v1 §4C S2).** The scopes do NOT exist as
   reusable functions today -- the drawing is monolithic inside `_CRTRenderer.render()`
   (`_waveform_mirror`/`_freq_bars_wide`, full-width, and the centre ring), and it draws
   NON-green (red/cyan/amber). Extract the scopes into standalone GREEN-ONLY helpers
   `draw_fft_scope(draw, cx, cy, r, freq_window, env)` / `draw_scope(draw, cx, cy, r,
   wave, env)` callable by BOTH the floor (v1) and the new node (v2). v1 must land first.
2. **`OTR_SceneAwareScopes` (NEW node)** -- generates a `scopes_only.mp4`: a BLACK frame
   with GREEN-ONLY scopes, scene-aware, 1920x1080 @ the DELIVERY fps (25), silent,
   bt709/yuv420p/CFR, length == the source video. It draws scopes on BLACK only (it does
   NOT decode the master video -> no generation loss). INPUT_TYPES: `clip_manifest_json`
   (from `OTR_SilentComposite`), `audio` (the master, ANALYSIS-ONLY), `canvas_w/h`, `fps`,
   scope params. RETURN: `scopes_mp4_path`.
3. **`OTR_PostUpscaleProcgenBlend` EXTENDED** -- a 3rd OPTIONAL input `scopes_mp4`;
   double green-blend in ffmpeg: `[main][procgen]blend=screen[tmp];[tmp][scopes]
   blend=screen[out]` (R/B zeroed on both overlays, green-only), audio `-c:a copy`. Absent
   scopes_mp4 -> the existing single-blend (v1 compat). Stays pure-ffmpeg (no PIL added).
4. **`OTR_SignalLostVideo` (floor)** -- add `draw_scopes: bool` (default `True` = v1
   compat); v2 sets it `False` so the floor draws CRT + title ONLY. Floor roles unchanged.

## Beat map + eligibility (the scene-aware core)
- REUSE `plan_timeline_segments(manifest, floor_available=True, target_total_frames=
  <source_frame_count>, fps=25)` -> the SAME integer frame ranges the composite used.
  Walk cumulative `[cursor, cursor+n_frames)` per segment (handles missing `start_s` /
  SEQUENTIAL mode -- do NOT rely on `start_s` floats).
- Per-frame eligibility from the segment's `source`:
  - `source=="clip"` + PORTRAIT -> draw the two scopes in the real gutters.
  - `source=="clip"` + LANDSCAPE -> SUPPRESS (the b-roll is the subject).
  - `source in {"floor","black"}` (GAP -- head / inter-beat / credits tail) -> DRAW the
    scopes (centre is fine -- no portrait to protect) so the "signal-lost" GAPS STAY ALIVE
    (the floor no longer draws them). [arch-r1 catch: suppressing gaps would kill them.]
- **Aspect resolver** (the manifest has NO native-dims field): for a `clip` segment, probe
  the clip `path` native dims (`ffprobe`) -> portrait if `h>w`; fall back to a versioned
  `engine_id -> aspect` registry; un-probeable/unknown -> SUPPRESS (never risk drawing over
  an unknown subject). VERIFY-AT-BUILD: that `path` points at a pre-composite asset with
  meaningful native dims (not an already-pillarboxed canvas output).
- **Gutter geometry** through the upscale: `portrait_w = round(480 * 1920/1472) ~= 626`;
  `gutter = (1920 - portrait_w)//2 ~= 647`; `left_cx = gutter//2`, `right_cx = w-gutter//2`,
  `r = int(min(gutter*0.36, h*0.30))`; `amp <= r*0.35` (compute from `w,h`, never bake).

## Audio re-analysis (must match the timeline)
`total_frames` = `ffprobe` the SOURCE video's exact frame count (NOT audio duration --
rounding/padding differ). `_analyze_audio(audio_np, sr, total_frames, fps=25)`; use
`round(fi*sr/fps)..round((fi+1)*sr/fps)` chunk boundaries (the floor's `sr//fps` is not
fps-exact). Stable-hash seeded RNG (same as v1); same `_analyze_audio` => frame-identical
to a 25fps floor.

## Pipeline + wiring
`SilentComposite -> RTXUpscale -> [SceneAwareScopes(manifest, audio) -> scopes_only.mp4]
-> PostUpscaleProcgenBlend(source=upscaled, procgen=floor, scopes=scopes_only) ->
MasterAudioMux`. The manifest routes SilentComposite -> SceneAwareScopes (NEW wire); the
master audio is wired as an explicit analysis input (no path-discovery). SceneAwareScopes
runs parallel to the upscale (it needs only manifest+audio+canvas).

## v1 CUTS carried + arch CUTS
Beat-boundary crossfade -> CUT for v1 (hard on/off is enough to prove scene-awareness; add
a 2-4 frame fade later). Ultra-dim landscape edge variant -> CUT (suppress cleanly on
landscape). Do NOT maintain two scope paths long-term (temp `draw_scopes` compat flag, then
hard-switch).

## OPEN / VERIFY-AT-BUILD
- The manifest clip `path` native-dims probe (pre-composite asset, real dims).
- The `engine_id -> aspect` registry for the non-HuMo/LTX/Wan engines (still_*/visualizer/
  latentsync).
- Two green `screen` blends do not wash out (visibility of scopes over the floor CRT).
- 24(floor)/25(delivery) fps: scope visibility uses the 25fps integer segment plan ONLY.
- master-WAV-missing fallback: skip scope drawing (silent) rather than fail the node.
