# Claude anchor review -- r1 (high-level arc / creative coherence)

Grounded against `nodes/_otr_shared/scope_draw.py`, `nodes/_otr_video_engines/eng_visualizer.py`,
`nodes/_otr_video_engines/registry.py`, `nodes/_otr_shared/role_compat.py`.

## VERDICT: SOUND ARC, but two high-level corrections + one scope-trim before it's coherent.

## CONFIRMED (grounding the plan's premises)
- `scope_draw.analyze_audio_np(audio_np, sr, total_frames, fps)` returns per-frame **RMS (volume)**,
  **32-bin FFT (freqs)**, and a **200-sample waveform** -- plus `dual_ema(volume)` for attack/decay
  envelopes. So "reuse the audio analysis" is real and sufficient. CONFIRMED.
- `scope_draw` ALREADY ships `build_scanlines(w,h)` + `build_vignette(w,h)` + `_small_font` +
  ring/bars/waveform painters, all **numpy + PIL, torch-free**. The OTR CRT/vignette mystique is
  reusable out of the box. CONFIRMED -- and this is the strongest argument for the CPU tier.
- `eng_visualizer` is `family="abstract"`, `accepts_still=False`, `required_inputs=("audio_ref",)`,
  `fallback_engine=None`, fits announcer/music/character by capability. CONFIRMED.

## MUST-FIX (high-level arc)
1. **Render PRIMITIVE mismatch (my own plan's misread).** The shipped visualizer **paints frames in
   numpy/PIL and uses ffmpeg only to ENCODE** -- it does NOT use ffmpeg filter-graph visualizers
   (`showcqt`/`showspectrum`). For consistency AND for the operator's bespoke OTR look (a radio dial /
   tuning-eye / spectrum sweep is a CUSTOM motif, not a stock ffmpeg filter), the CPU tier must stay on
   the **numpy/PIL paint -> ffmpeg encode** pattern. Demote `showcqt`/`showspectrum` to "reference look
   only." This is the single biggest correction.
2. **Creative coherence = OTR MYSTIQUE (operator, hard).** The arc must be a radio-themed, rainbow-
   COLORED viz: muted/desaturated spectrum (not neon), a glowing tube/dial/magic-eye that pulses with
   RMS, the rainbow as a tuning/SIGNAL-SPECTRUM sweep, over a dark noir field with the EXISTING
   `build_vignette` + `build_scanlines` + film grain. The plan now states this; the arc is only coherent
   if every effect is justified against this motif, not "plasma because plasma."

## SHOULD-FIX / SCOPE
3. **Trim or defer the GPU tier.** The operator's north star is "runs easily on AMD/Mac/any box." The
   numpy/PIL CPU tier delivers ~90% of the look with 100% portability and zero new deps. The GPU tier
   adds a headless-GL/EGL/torch dependency that is exactly the cross-vendor fragility the operator wants
   to avoid. Recommend: ship `viz_rainbow_cpu` FIRST as its own green chunk; treat `viz_rainbow_gpu` as a
   SEPARATE, later, opt-in, capability-gated experiment (or drop it if the CPU look satisfies). Don't let
   the GPU tier block the CPU tier.
4. **"onsets" is an overclaim.** The analysis gives RMS + FFT + waveform + EMA -- not an explicit onset
   detector. Reword to "transients via RMS/EMA delta." Minor, but keep the plan honest.

## UNVERIFIABLE (verify-at-build)
- The `no_audio` procedural mode's capability implications (an engine has ONE `required_inputs`; a
  per-mode change breaks role_compat). Defer the resolution to r3 (wiring). High-level: a no-audio mode
  is probably a SEPARATE engine id, not a mode flag -- but confirm against role_compat at wiring time.
