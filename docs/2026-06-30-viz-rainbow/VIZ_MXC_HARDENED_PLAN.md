# viz_mxc -- HARDENED PLAN (kibitz r1+r2 converged, 2026-06-30)

OTR multi-colored ("mxc") audio-reactive visualizer, the creative replacement for the retired
`abstract` floor. Hardened by the local panel (Claude Code + Codex + Antigravity) + a Claude
code-grounded anchor; every fix below is verified against the real repo. Misreads discarded
(Claude Code "no label mechanism" -- `vd._label_for` exists; Antigravity `fallback_engine` -- violates
the no-fallback invariant). The panel converged hard at r2 (identical grounded must-fixes), so r3/r4
are folded in here rather than re-run for the same conclusions.

## LOCKED DECISIONS
1. **ONE engine: `viz_mxc_cpu`, `required_inputs=()` (audio-OPTIONAL).** Reacts to audio when present
   (announcer/music/character); renders a time/seed-driven OTR rainbow when absent -> also the no-image
   floor for scene_broll/background. `required_inputs=()` => fits all 5 roles (C2 capability); fills the
   `abstract` gap; `accepts_still=False` => mints NO still (kills the unwanted z_image-on-non-audio-slot).
   NO split `viz_mxc_gen` (panel converged: eng_visualizer already handles missing audio; one engine).
2. **CPU renderer = numpy/PIL paint -> `scope_draw.encode_silent_mp4` ONLY.** NO ffmpeg filter-graph
   visualizers (showcqt/showspectrum) -- they take audio, drive frame count, and break the silent /
   `target_frame_count` contract. Paint EXACTLY N frames.
3. **`viz_mxc_gpu` DEFERRED out of v1** -- separate later opt-in spike (lean: torch tensor ops on the
   active device; NVIDIA-first, capability-probe at `assert_usable`, FAIL CLOSED LOUD, NO `fallback_engine`).
4. **Labels auto-derived** (`vd._label_for` -> `viz_mxc_cpu (16:9)`); do NOT set a custom label (breaks the
   `" ("` round-trip in `_engine_id_from_pick`).

## WIRING MUST-FIX (grounded; each lands in the SAME chunk as the engine)
- **Register:** `@register` in new `nodes/_otr_video_engines/eng_viz_rainbow.py` + an import row in
  `nodes/_otr_video_engines/__init__.py` (else it never loads) + CAPABILITIES row in `registry.py`:
  `"viz_mxc_cpu": {"vram_class":"cpu","vram_estimate_mb":0,"required_toolchain":None,"requires_sidecar":False,"cpu_ok":True,"model_requirements":[]}`
  (test_capability_profiles asserts CAPABILITIES keys == all_engine_names -- must land together).
- **Ambient-master-audio gate:** add `viz_mxc_cpu` to `render_driver._uses_ambient_master_audio` (today it
  hardcodes `engine_id=="visualizer"`/`family=="audio_conditioned_video"`); without it the music-open / no-
  line beats are audio-STARVED and render the idle path. Add a regression for a music-open beat with
  `master_audio_path` and no line timing.
- **Family maps:** add `"viz_mxc_cpu":"abstract"` to `render_driver.ENGINE_FAMILY` AND
  `content_oracle._FAMILY_FALLBACK` (so the soak/oracle classify it when the registry isn't loaded;
  `abstract` is motion-EXEMPT in the oracle = correct for a visualizer).
- **node-87 promotion is a SEPARATE, operator-gated chunk:** default-off = registered + selectable but NOT
  the saved widget value. Promotion = edit node 87 in `workflows/otr_scifi_16gb_full.json` + update the
  pinned `tests/test_workflow_live_passes_validator.py` (asserts node-87 == visualizer/visualizer/
  humo_14B_169) + `config/profiles/16gb_full.json` (pins announcer/music/other_beats=visualizer) in the
  SAME commit, then re-validate. Do NOT conflate "registered" with "wired into the saved JSON".

## CODING PLAN
- `render_clip`: resolve canvas w/h/fps + `target_frame_count`; if `audio_ref` -> `analyze_audio_np`
  (RMS / 32-bin FFT / 200-sample waveform) + `dual_ema`; else seed/time-driven params. Build ONE per-frame
  `params` struct (hue_phase, bloom, dial_angle, spectrum[32], grain_seed); ONE painter consumes it for
  both paths. Paint EXACTLY N frames -> `encode_silent_mp4` (silent h264/yuv420p/bt709 CanonicalClip).
- **New shared painter** `scope_draw.paint_rainbow_frame(w,h,fi,total,fps,volume,freq,wave,signal,loss,
  scanlines,vignette,rng_key,font_small)` -- a HUE-MAPPED spectrum (32 FFT bins -> rainbow) + a glowing
  vacuum-tube / radio-dial / magic-eye needle that sweeps with RMS, on a dark noir field, graded with the
  EXISTING `build_vignette` + `build_scanlines` + seeded film grain, muted (desaturated) not neon.
- **Determinism (V-7):** all stochastic elements draw from `scope_draw._rng(seed,frame,salt)`; same
  `request_seed` -> byte-identical frames. `engine_version="1"`.
- **Diagnosability:** stamp `mode`/`audio_used` into the raw result qc so the content-oracle/soak can tell
  reactive vs idle clips apart.

## CREATIVE GRAMMAR (one look for v1)
OTR mystique, NOT party rainbows: muted rainbow as a RADIO SIGNAL-SPECTRUM sweep (ties to "Signal Lost");
a tube/dial/magic-eye that pulses with the audio; CRT scanlines + vignette + 35mm grain; dim amber/cyan
rim. Every effect justified against this motif.

## TESTS (suite + Bug Bible + B7 green; push per chunk)
- New `tests/test_video_viz_rainbow.py`: registration; `required_inputs=()` fits all 5 roles (capability);
  `accepts_still=False` (mirror the visualizer image-dispatch test in test_image_platform_c1); cold-import
  clean; render-contract with `encode_silent_mp4` MONKEYPATCHED (not "ffmpeg mocked") for audio-present AND
  audio-absent; frame-count exactness; seed-determinism (render twice -> identical hash); one real-ffmpeg
  test skipped like `test_video_visualizer`. Plus the ambient-audio + capability-consistency regressions.

## BUILD ORDER
C-mxc1 `viz_mxc_cpu` (engine + paint_rainbow_frame + CAPABILITIES + __init__ import + ambient-gate +
family-maps + tests; default-off, green+push) -> C-mxc2 OTR-mystique look pass + content-oracle/soak
validation -> C-mxc3 node-87 promotion (operator look-QA, pinned-test+profile+JSON in one commit) ->
[later/opt-in] `viz_mxc_gpu` spike. The GPU tier never blocks the CPU tier.
