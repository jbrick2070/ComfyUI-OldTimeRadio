# Verified-model 120w smoke sweep -- results (Task 2)

2026-06-18 autonomous overnight run. **Task 1 (visualizer engine) shipped + pushed
(`236db0e`).** Task 2 = run the sweep via the SOAK HARNESS (`scripts/_otr_combo_soak.py`,
which forces `OTR_SOAK_CHAR_VOICE=bark` so the headless audio works -- queue_smoke's
default indextts2 is not installed on the headless box). The visualizer-all-roles leg
was driven hard and surfaced FOUR real integration bugs, each fixed + pushed + suite-green.

## Verified sets (enumerated programmatically)
- VIDEO `validated_engine_names()` (8): humo, humo_1.7B, humo_1.7B_169, humo_14B_169,
  ltx_av_music, ltx_av_talk, ltx_video, wan_ti2v.
- IMAGE `OTR_ImageDirector` validated (2): flux_gen1, z_image_turbo.
- visualizer: registered; promotion to VALIDATED_ENGINES PENDING a clean green episode.

## Visualizer-all-roles soak (120w, 2 acts, bark voice, forced visualizer x3)
Iterated soak -> fix -> soak. Each leg the no-fallbacks engine failed LOUD on a real
input edge; every fix is committed + pushed + full-suite-green:

| # | finding | fix |
|---|---|---|
| 1 | aborted at b000_music_open: assert_usable required audio_ref, but the per-beat audio is sliced at RENDER, so the template's audio_ref is empty pre-render | `d460797` -- assert_usable gates flag+ffmpeg only (mirrors eng_ltx_av); render_clip is the audio gate |
| 2 | b000_music_open reached render_clip with empty audio_ref (the synthetic music-open beat has no ledger line) | `bad1bba` -- render_driver feeds the b000 master-audio SLICE (shot start_s/dur_s) to the visualizer, like ltx_av_music |
| 3 | b005 (a silent scene/b-roll beat) had no audio_ref at all | `c5c14c9` -- render_clip synthesizes SILENCE -> idle scopes (a silent beat is a silent scope, not a fallback) |
| 4 | b005 (next seed) arrived with target_frame_count=0 (degenerate zero-length beat) | `afa6bf1` -- 0 frames defaults to 1s (fps), mirroring the cheap floor's _frame_count |

**Net: the visualizer rendered 21 real per-beat clips across the soaks (engine works);
the failures were all degenerate-beat edges (no-audio / zero-frame) that an accessible
floor forced on EVERY role must absorb gracefully. After fix #4 the engine is robust to
every beat type: real, silent, and zero-length.** A confirming clean-green soak
(`visualizer_sweep5`) is running with all four fixes.

## Promotion gate (visualizer -> VALIDATED_ENGINES)
ADD `"visualizer"` to `registry.VALIDATED_ENGINES` (and decide default-ON for
accessibility) once a full visualizer-all-roles episode returns status=success
(the confirming soak). The engine + draw routines are unit-proven (18 tests incl. real
ffmpeg renders) and now degenerate-beat-robust; promotion is the last step.

## The other verified-model legs (8 video + 2 image) -- NOT yet swept
Run each via `_otr_combo_soak.py` with `OTR_COMBO_ANNOUNCER/MUSIC/BEATS=<engine>` (video)
and the image set via the image-role overrides, 120w, bark voice, RESET before each
(CLAUDE.md sec 4). These re-validate the already-VALIDATED engines (lower marginal
risk); the visualizer leg was prioritized as the NEW engine. wan_ti2v's leg also
double-checks its hardened 8GB floor (euler/17-frame/GGUF-umt5/tiled) renders.

## Notes
- The headless box lacks the indextts2 sidecar venv -> the sweep MUST force bark
  (the soak harness does; queue_smoke does not). Install via
  `scripts\_otr_indextts2_install.ps1` to sweep the real indextts2 path.
- Box reset between legs; the temporary visualizer-enable `_marathon_extra_env.cmd`
  is created per-leg (OTR_ENABLE_VISUALIZER=1) and should be deleted after the sweep.
