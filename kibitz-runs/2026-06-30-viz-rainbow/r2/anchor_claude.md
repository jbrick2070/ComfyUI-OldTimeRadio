# Claude anchor review -- r2 (coding plan / implementability)

Grounded against `scope_draw.py` (analyze_audio_np / dual_ema / build_vignette / build_scanlines / _rng /
ring_geom / painters / encode_silent_mp4) + `eng_visualizer.py` render lifecycle.

## VERDICT: IMPLEMENTABLE -- the primitives all exist; 3 coding specifics must be pinned.

## CONFIRMED (the coding plan is buildable from existing pieces)
- `viz_mxc_cpu.render_clip` = (1) resolve canvas w/h/fps + `target_frame_count` from the request (same
  helpers the cheap families use), (2) if `audio_ref` present -> `analyze_audio_np` -> RMS/FFT/waveform +
  `dual_ema`; else procedural params from frame index + seed, (3) paint EXACTLY N frames numpy/PIL with the
  OTR-mystique motif + `build_vignette` + `build_scanlines` + seeded grain, (4) `encode_silent_mp4` ->
  the silent h264/yuv420p/bt709 CanonicalClip. Mirrors `eng_visualizer` 1:1. CONFIRMED buildable.

## MUST-FIX (coding specifics to pin THIS round)
1. **Frame-count exactness.** Paint EXACTLY `target_frame_count` frames (the per-beat timing contract) --
   never let audio duration set the count (that was the ffmpeg-filter trap r1 cut). Assert N frames before
   encode. Pin in the spec + a test.
2. **The two code paths share ONE painter.** audio-present and audio-absent must call the SAME frame
   painter with a uniform `params` struct (hue_phase, bloom, dial_angle, spectrum[]) -- the only difference
   is how `params` per frame is produced (audio-driven vs seed/time-driven). This keeps it one engine, not
   two render pipelines. Specify the `params` contract.
3. **Determinism (V-7).** All stochastic elements (grain, jitter) draw from `scope_draw._rng(seed, frame,
   salt)`; the same `request_seed` -> byte-identical frames. Test: render twice -> identical frame hash.

## SHOULD-FIX
1. **Reuse, don't fork, the painters.** `freq_bars_green` / `_waveform_mirror` / `ring_geom` already exist;
   the rainbow version is a HUE-MAPPED variant (map 32 FFT bins -> spectrum hue) + a new dial/magic-eye
   painter. Add the rainbow painters to `scope_draw` (shared) rather than a private copy, so the floor +
   both viz engines stay one source.
2. **Muted-rainbow grade.** Desaturate + apply the OTR amber/cyan rim + grain so it reads period-radio,
   not neon. A single post pass (numpy HSV shift + grain blend) keeps it one grammar.

## UNVERIFIABLE (verify-at-build)
- Exact `encode_silent_mp4` signature + whether it takes a frame iterator or a temp PNG dir (read it at
  build). The CONTRACT (silent, bt709, N frames) is confirmed; the call shape is a detail.
- Per-frame paint cost at 1472x832 x ~N frames -- the existing visualizer proves it is fast enough on CPU;
  confirm no per-beat regression in the soak.
