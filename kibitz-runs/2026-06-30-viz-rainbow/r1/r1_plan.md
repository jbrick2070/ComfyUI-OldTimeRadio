# viz_rainbow -- HARDENED after kibitz r1 (high-level arc)

Panel (Claude Code + Codex + Antigravity) + Claude anchor CONVERGED. Grounded survivors folded in;
misreads discarded (Claude Code's "no label mechanism" -- `vd._label_for` exists; Antigravity's
`fallback_engine` -- violates the no-fallback invariant, rejected).

## DECISIONS (locked this round)

1. **CPU renderer = numpy/PIL paint -> ffmpeg ENCODE only.** DROP every ffmpeg filter-graph visualizer
   (`showcqt`/`showspectrum`/`showwaves`/`avectorscope`): they take an audio input, let ffmpeg drive
   frame-rate/count (breaks the per-beat `target_frame_count` contract), and the silent-clip contract
   forbids an audio stream. Follow the SHIPPED `eng_visualizer` exactly: paint frames in numpy/PIL using
   `scope_draw`, then `scope_draw.encode_silent_mp4`. ONE render path -- no dual numpy+ffmpeg layer.

2. **viz_mxc_cpu is AUDIO-OPTIONAL: `required_inputs=()`.** This is the single engine that BOTH
   reacts to audio (announcer/music/character) AND fills the no-image floor for scene_broll/background
   procedurally (when no `audio_ref`, render a time/seed-driven OTR rainbow). `required_inputs=()` makes
   it fit ALL five roles by capability (the C2 model) -- the true `abstract` replacement -- and it mints
   NO still (`accepts_still=False`), so it never triggers z_image on a non-audio slot (the operator's
   exact complaint about the visualizer leg). render_clip branches on audio-present (like
   `eng_visualizer` already does for a missing slice). NO third engine id needed for v1.
   - Tradeoff noted for r2: two code paths in one engine. If r2 deems that too complex, fall back to
     Codex/Claude-Code's split (audio-only `viz_mxc_cpu` + a separate `viz_mxc_gen`
     `required_inputs=()`). Primary = the single audio-optional engine.

3. **GPU tier DEFERRED out of v1.** Ship `viz_mxc_cpu` first as its own green chunk + soak. The GPU
   tier is realistically NVIDIA-first (the 5080) -- moderngl/EGL is not headless-cross-vendor, ffmpeg-GL
   needs a display, torch-compute is CUDA/MPS-ish not truly AMD. Re-scope `viz_mxc_gpu` as a SEPARATE
   later opt-in spike with ONE chosen stack (lean: torch tensor ops on the active device), a capability
   probe at `assert_usable` that FAILS CLOSED LOUD on a non-capable box ("select viz_mxc_cpu" -- the
   episode fails per the no-fallback rule; do NOT declare a `fallback_engine`).

4. **Labels are AUTO-derived; do NOT specify custom labels.** `vd._label_for` produces
   `viz_mxc_cpu (16:9)` and `vd._engine_id_from_pick` round-trips it by splitting on `" ("`. A custom
   label like "Rainbow visualizer (CPU...)" would parse back to "Rainbow visualizer" != the engine id and
   fail closed. Just register the engine id; the suffix is automatic.

5. **CAPABILITIES row: `required_toolchain=None`** for both tiers (a `"GL/torch"` toolchain would fail
   `capability_profiles` validation and disable the engine on every shipped profile). GPU gating lives in
   runtime `assert_usable`, never in the toolchain field.

## CREATIVE -- OTR MYSTIQUE (one visual grammar for v1, not a buffet)
Numpy-painted, muted rainbow (not neon) on a dark noir field: a glowing vacuum-tube / radio-dial /
magic-eye tuning indicator that pulses with RMS; the rainbow as a SIGNAL-SPECTRUM sweep (FFT 32-bin ->
spectrum colours); reuse the EXISTING `scope_draw.build_vignette` + `build_scanlines` + film grain for the
CRT/period look. Every effect justified against this motif. (showcqt etc. are reference-look only.)

## WIRING (grounded)
- Reuse `scope_draw.analyze_audio_np` (RMS + 32-bin FFT + 200-sample waveform) + `dual_ema` directly --
  the COPY-not-extract invariant in `scope_draw` applies to the FLOOR NODE only; other engines may import
  it (visualizer is the precedent). Keep `_rng` blake2s seed determinism + `engine_version="1"`.
- Register: `@register` in a new `eng_viz_rainbow.py` + an import row in
  `nodes/_otr_video_engines/__init__.py` (without it the engine never loads) + a CAPABILITIES row in
  `registry.py` (cpu / cpu_ok True / required_toolchain None).
- node-87 promotion is a SEPARATE phase: default-off = registered + selectable but NOT the saved widget
  value; promotion = set node 87 in `workflows/otr_scifi_16gb_full.json` + re-validate (AGENTS.md / the
  workflow-source-of-truth rule). Don't conflate "registered" with "wired into the saved JSON".
- Capability terminus: `required_inputs=()` -> fits all 5 roles; `accepts_still=False` -> no still minted.
- Tests: registry+CAPABILITIES consistency; capability matrix (fits all 5 by capability); accepts_still
  False; cold-import clean; an offline render-contract test (audio-present AND audio-absent paths) with
  ffmpeg mocked; seed-determinism. Suite + Bug Bible + B7 green; push per chunk.

## BUILD ORDER
C-rb1 viz_mxc_cpu (engine + CAPABILITIES + __init__ import + tests, default-off) -> C-rb2 OTR-mystique
look pass + content-oracle/soak validation -> C-rb3 node-87 promotion (operator look-QA) -> [later/opt-in]
viz_mxc_gpu spike. GPU tier does NOT block CPU.

## STILL OPEN (for r2/r3/r4)
- r2: confirm the single audio-optional engine (2 code paths) vs the split-engine fallback; the exact
  numpy rainbow geometry + how RMS/FFT/EMA drive the dial/spectrum/grain.
- r3: the node-87 promotion mechanics + widget_mapping; the no-audio floor's interaction with the
  scene/background still-pool path (does the dispatcher try to mint a still it won't use?).
- r4: residual no-fallback / cold-import / determinism checks.
