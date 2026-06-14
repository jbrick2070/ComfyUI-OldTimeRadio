# PROCGEN BUILD LOG -- §4C + §4D (CRT procgen upgrade + scene-aware scopes)

Built 2026-06-13 in a dedicated coder window from `docs/GO_FORWARD_PLAN.md` §4C + §4D
(now removed from that doc -- completed work lives here + git history + the tracker).
Branch `v2.0-alpha`. Base HEAD `a54df22`. Final HEAD `39aa6c9` (== origin).

## Commits

- **`336fb41`** -- `feat(procgen 4C): floor foundation + big-bold episode-title card`
  (`nodes/video_engine.py`, +515/-96).
- **`39aa6c9`** -- `feat(procgen 4D): scene-aware scopes node + 3-input blend`
  (`nodes/otr_scene_aware_scopes.py` NEW, `nodes/otr_post_upscale_procgen_blend.py`,
  `__init__.py`, `tests/test_video_scene_aware_scopes.py` NEW; +718).

## What landed

### §4C floor (`nodes/video_engine.py` `_CRTRenderer` + `render_video`)
- **S1 foundation.** New ctor `_CRTRenderer(w,h,title,volume,freqs,waves,fps,timing=None)`
  + `render(self, fi, draw_scopes=True)`. `volume`/`freqs`/`waves` converted to np up
  front. **Dual EMA precomputed in `__init__`:** `signal` (α 0.05) + `trig` (α 0.30),
  `loss = 1-signal`, `signal[0]=trig[0]=volume[0]`, read-only downstream. `import hashlib`
  + a `_rng(fi, salt)` blake2s-seeded generator **replaces the unseeded section-8 noise**
  (deterministic per `title|fi|salt`). Geometry from `w/h` (no baked 1920). **Draw order
  moved:** background (grid + gated scopes + bottom bar) -> section-8 vignette/scanlines/
  noise (numpy) -> THEN section-1 ident OR the hero card, so text draws AFTER the vignette
  multiply (kills the v1.5.1 dimmed-text bug). Caller L1822/L1825 updated; timing resolved
  via the new `_resolve_title_timing`.
- **S3 hero title card** on the b000 music-open window: decode -> reveal -> POP -> dock.
  Carrier-lock "=== SIGNAL LOST ===" decodes from a seeded scramble; a broken-phosphor
  carrier meter crawls to solid on `signal`; HERO title at 2-3x the title font, centre-
  anchored (EXEMPT from gutter sanctity), fake-bold by overstrike `{(0,0),(1,0),(0,1),
  (1,1)}`, `textbbox` wrap/scale to `~0.8w x 0.28h` with a font floor, decoded-fragment
  reveal on integer frames + a block cursor; 1-2 frame POP bloom; raster-collapse dock that
  lerps the hero into the section-1 ident coords (the docked target IS the normal ident).
  Window `[start, music_end + dock_frames)`, `dock_frames = min(fps*0.5, first_dialogue_f -
  music_end)`; never overruns the dialogue or `total`.
- **S4 envelope.** Grid brightness hierarchy (`grid_alpha *= 0.35 + 0.65*signal` -> dims
  first in the silent gaps); bounded gutter-clamped horizontal coordinate drift on the
  gated ring (`loss * w//120`, clamped on-frame; NOT `np.roll`, NOT a hue flash).
- **S5.** Conditional `music_close` outro bookend (reveal + POP, no dock; only inside
  `total_frames`). `draw_scopes` BOOLEAN (default True, appended LAST per the positional-
  widget rule) gates `render()` sections {2,3,5,6} as a SET (verified no undefined-var leak;
  section 3 reads section 2's `r`). Threaded `render_video -> _CRTRenderer -> render()`.
- **Timing extractor `_resolve_title_timing`**: music-open line by `speaker_role` in
  `("music_open","music_visual")` using `start_s`/`dur_s` (seconds); first dialogue from
  `("announcer","character")`; optional `music_close`. Fallback: volume-envelope intro
  window when `start_s` is unavailable (the wire ledger may carry `start_s=None`).

### §4D scene-aware (LOCKED new-node-only; floor NOT relocated)
- **S-v2a green-only helpers** (in `otr_scene_aware_scopes.py`): `draw_fft_scope(draw,cx,cy,
  r,freq_window,env)` (32 radial FFT spokes + bounded comet-tails, idle rotating radar
  sweep) + `draw_scope(draw,cx,cy,r,wave_window,env)` (wave traced around the circumference
  + electron sweep dot/trail, idle jittering baseline). Geometry by params (no `self`),
  `amp <= r*0.35`, line widths 1-2px. **CRT_CYAN/CRT_AMBER are deliberately not defined** ->
  green-only by construction (verified: 0 references; rendered R channel == 0).
- **S-v2b `OTR_SceneAwareScopes`** -> `scopes_only.mp4` (black + green, no master decode):
  `CATEGORY="OldTimeRadio/v2/video"`, `FUNCTION="render_scopes"`, `RETURN ("STRING",)/
  ("scopes_mp4_path",)`, registered in `_NODE_MODULES`. Required `clip_manifest_json`;
  optional `audio` (analysis-only; absent -> synth zero arrays, never calls the floor
  `_analyze_audio`), `out_w/out_h` (1920/1080), `ffmpeg`. Beat map via
  `plan_timeline_segments(floor_available=True, target_total_frames, fps=25)` -> integer
  ranges + `total` (no source-video probe for counts). Eligibility by `source`: clip+PORTRAIT
  -> gutters; clip+LANDSCAPE -> suppress; head/inter gap -> centre (keep the signal-lost gaps
  alive); TAIL/credits (gap at `start >= last-beat-end`) -> suppress. Aspect via `ffprobe`
  (h>w), memoized per path; un-probeable -> suppress + log. Empty manifest -> fail early.
  **Own SILENT `-an` encoder** (the floor `_encode_mp4` hard-requires audio): 25fps HARD-LOCK,
  yuv420p, CFR, bt709, nvenc-if-available else libx264. Deterministic stable-hash RNG.
- **S-v2c blend** (`OTR_PostUpscaleProcgenBlend`): optional `scopes_mp4_path` (appended LAST)
  switches to a NEW 3-input gbrp-throughout filtergraph -- `[0:v]format=gbrp[main];
  [1:v]<conform+crush+green>format=gbrp[pgn]; [2:v]<conform+setsar+zeroRB>format=gbrp[scp];
  [main][pgn]blend=screen[tmp]; [tmp][scp]blend=lighten[out]; format=yuv420p`. 2nd blend =
  **lighten (max)**, not a 2nd screen, so the green layers don't compound. `-map [v] -map
  0:a? -c:a copy` (C7-safe), bt709/CFR kept. Absent `scopes_mp4` -> the unchanged single-
  procgen path.
- **S-v2d wiring + tests:** node registered; `tests/test_video_scene_aware_scopes.py` (9
  tests) covers eligibility (portrait/landscape/head-gap/credits-tail), green-only,
  amp-clamp, frame determinism, empty-manifest fail-early, silent `-an` encode (ffmpeg-
  gated), and the 3-input vs 2-input filtergraph.

## Verification (this session)
- Floor harness (torch-stub-free; real venv): determinism PASS for `draw_scopes=True`
  AND `False`; `scopes on != off`; golden frames eyeballed for reveal / POP / dock / ident /
  outro / scopes-off (title decode->dock->ident is seamless); long-title overflow no crash;
  timing extractor correct; empty-timing renders.
- Node harness: eligibility classification PASS on a real 5-segment manifest with real
  ffprobe (portrait.mp4 / land.mp4); gutters + centre golden frames eyeballed (two asymmetric
  green scopes in the real gutters; concentric FFT+oscilloscope on the gap); silent
  `scopes_only.mp4` (no audio stream); 3-input filtergraph well-formed; empty manifest raises.
- Regression (Windows venv `...\ComfyUI\.venv`, py3.12 torch 2.10): `test_post_upscale_
  procgen_blend` (21) + `test_audio_byte_identical` (byte-identical GREEN) + `test_b7_
  forbidden_sweep` (5) + `test_forbidden_sweep_scope` (3) + `test_video_render_path_cw4`
  (16) + `test_video_ledger` (4) + the new scopes tests (9) all PASS. **Bug Bible** 16 passed.
  Full `tests/` **collects 4244 with zero import errors**. No BOM / AST-OK on all touched .py.

## Stubbed / CUT (per the panel, not regressions)
- The §4C-v1 **in-floor gutter scopes** -- SUPERSEDED by the node (§4D LOCK). The floor keeps
  its existing sections {2,3,5,6}, now gated off via `draw_scopes=False` for v2.
- Vignette-choke (v1.5.1 readable-text risk), telemetry micro-text, FFT peak-hold + noise-
  floor rings, oscilloscope free-running trigger seam, halation, the formal hierarchy layer-
  floor system -- all CUT from v1 (use per-element brightness scaling instead).
- The engine-registry aspect path -- CUT in favor of `ffprobe` on the clip `path`.

## v2 workflow wiring -- DONE (`eb64cd1`)
Wired into `workflows/otr_scifi_16gb_full.json` (the single source of truth): new node 94
`OTR_SceneAwareScopes`; `OTR_VideoRenderBatch`(92)`.clip_manifest_json` fan-out -> it
(link 271, alongside the existing composite consumer); `OTR_EpisodeAssembler`(7)`.episode_audio`
-> its `audio` (link 272); its `scopes_mp4_path` -> `OTR_PostUpscaleProcgenBlend`(93)'s new
`scopes_mp4_path` input (link 273); floor `OTR_SignalLostVideo`(12) `draw_scopes=False` + `fps=25`.
Validated: link referential integrity + NCM registration + INPUT_TYPES match + widget vectors;
`test_default_workflow_validator` + widget-vector + guardrails + core + capability_profiles +
audio-byte-identical + b7 + Bug Bible all green. (Originally the §4D code shipped DORMANT -- the
production JSON had the node absent, the blend 2-input, the floor `draw_scopes` defaulting True /
`fps` 24 -- caught by the operator; this commit makes it live.)

## What the eyeball still gates
- A live **full-episode render** on the 5080 (the floor title card on a REAL b000 music-open
  window + the scene-aware scopes blended over a real upscaled mp4) -- confirm the
  `lighten` scopes stay visible over the floor CRT and the title decode/dock reads well.
  prod/`main` remain GATED until this passes.
- Verify the REAL disk-ledger `speaker_role` string for the opening music on a live episode
  (resolver matches `music_open`/`music_visual`; a different string falls back to the
  volume-envelope window, which is safe but worth confirming).
