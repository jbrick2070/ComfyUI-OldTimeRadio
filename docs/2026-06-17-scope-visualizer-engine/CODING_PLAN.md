# Scope Visualizer Engine -- Coding Plan (2026-06-17)

**Status:** PLANNER draft, pre-roundtable. Planner window does NOT write production
code -- this is the coder-window kickoff once the roundtable converges.

**Goal:** a low-VRAM "video" lane. Audio-reactive CRT scopes rendered AS the
per-beat picture (not the AI-video render, not the existing overlay), selectable
per role in the `OTR_VideoDirector` dropdowns. **16:9 only** (operator directive
2026-06-17). When all three roles pick it, the episode renders at ffmpeg-only,
near-zero GPU.

---

## 0. Grounded current state (verified against the repo, do not re-derive)

- **The capability slot already exists.** `nodes/_otr_video_engines/registry.py`
  `CAPABILITIES["visualizer"] = {vram_class: cpu, vram_estimate_mb: 0,
  required_toolchain: None, requires_sidecar: False, cpu_ok: True,
  model_requirements: []}`. No registry table edit needed.
- **No engine adapter is registered for it.** There is no `eng_visualizer.py`;
  nothing calls `@register` with `name = "visualizer"`. This is the core gap.
- **It is NOT in `VALIDATED_ENGINES`** (registry.py ~L277). The per-role dropdown
  is built from `validated_engine_names()` (the live registry intersected with
  `VALIDATED_ENGINES`) per the 2026-06-17 tested-only display gate. So even once
  the adapter registers, it will NOT appear in the dropdown until it is added to
  `VALIDATED_ENGINES` after a green end-to-end validation.
- **The draw + encode code already exists** in `nodes/otr_scene_aware_scopes.py`
  as module-level, importable, torch-free functions: `draw_fft_scope`,
  `draw_scope`, `_analyze_audio_np`, `_dual_ema`, `_encode_silent_mp4`, `_green`,
  `_rng`. They are written geometry-by-params (no `self`), explicitly so callers
  other than the overlay node can reuse them.
- **The overlay node stays untouched.** `OTR_SceneAwareScopes` +
  `OTR_PostUpscaleProcgenBlend` are the late whole-episode gutter overlay (§4D).
  The new engine is a SEPARATE per-beat path. Do not conflate or rewire the two.
  The BUG-LOCAL-406 master-length padding logic lives in the overlay node and is
  irrelevant per-beat (each beat is its own clip; the mux concatenates).

## 0.1 IMPORTANT -- the procedural visualizer was NOT deleted (resurrect, don't rebuild)

Operator recollection ("we used to have procedural, it got ripped out with some
defunct modes") -- verified against git. The procedural renderer is **alive**:

- `nodes/video_engine.py` `render(self, fi, draw_scopes=True)` (L469+) draws a
  FULL-FRAME CRT audio-reactive scene: circular frequency ring + orbiting
  particles (green/cyan/amber -- **full colour, not green-only**) + geometric grid
  + mirrored waveform + frequency bars + CRT post (scanlines/vignette/noise) +
  the hero title card. This IS the procedural visualizer.
- It was never ripped out. What was lost is the ability to **select it as a
  beat's picture**. The CW-4 teardown (commit `3f8c486` "delete legacy
  OTR_VideoComposite + tombstone the type") removed the legacy composite/render-
  mode that exposed the floor as a first-class output. The renderer survived as
  the fallback floor + gap/credits fill, and its scopes were turned OFF
  (`draw_scopes=False`, commit `39aa6c9`) when the §4D gutter overlay landed.
- git confirms NO deleted standalone visualizer engine; the only deleted video
  engine is `eng_latentsync.py`, and the only deleted "procedural" node is the
  unrelated `OTR_BatchProceduralSFX` (audio SFX). The `7edc253` soak gate already
  references a "legit visualizer music beat" -- the concept still lives.

**Consequence for this plan:** the `eng_visualizer` adapter REUSES the *drawing
logic* of `video_engine.py`'s full-frame renderer, NOT the floor NODE. This is
cheaper, brings back the colour procedural look, and reuses battle-tested code.
The gutter `otr_scene_aware_scopes.py` helpers (section 2) become the GREEN-ONLY
alternative preset, not the primary.

### 0.2 SEPARATION INVARIANT (operator concern 2026-06-17 -- HARD)

"Resurrect" must NOT mean the procgen floor starts painting over the video, and it
must NOT touch `OTR_SceneAwareScopes`. Three things stay completely independent:

1. **The floor** (`OTR_SignalLostVideo` / `video_engine.py render()`) keeps its
   CURRENT job ONLY: gap/credits/fallback fill, with `draw_scopes=False` when the
   §4D overlay runs. Its behavior does NOT change. The visualizer engine does NOT
   call the floor node, does NOT flip `draw_scopes`, and does NOT make the floor
   run for non-floor beats.
2. **The §4D overlay** (`OTR_SceneAwareScopes` + `OTR_PostUpscaleProcgenBlend`)
   stays exactly as wired. The engine never reads or alters it.
3. **The visualizer engine** is a NEW, standalone per-beat code path. It is INERT
   unless a role's dropdown explicitly selects `visualizer`. When NOT selected,
   nothing about the render changes -- no procgen over the AI video, no overlay
   change. When selected for a beat, `OTR_VideoRenderBatch` calls its
   `render_clip` and it produces THAT beat's clip; that is the only time it runs.

**Implementation rule that guarantees this:** do NOT import or invoke the floor
node from the engine. Instead, EXTRACT the pure drawing routines from
`video_engine.py` (the ring/particles/grid/waveform/bars/CRT-post functions) into
a shared, torch-free helper module (e.g. `nodes/_otr_shared/scope_draw.py`) that
BOTH the floor and the new engine import. No behavior change to the floor (it
calls the same code it does today); the engine calls the same routines to paint a
standalone full-frame clip. If extraction is too invasive for v1, the engine
COPIES the routines into its own module -- still zero coupling. Either way the
floor, the overlay, and the engine share zero runtime state and zero triggering.

Re-scope sections 2-4 accordingly: the primary engine = the full-frame procedural
look, rendered by SHARED/COPIED draw routines, minus the title-card/gap logic that
only makes sense for the whole-episode floor.

## 1. The contract the adapter must satisfy

From `registry.py` `VideoEngine(Protocol)` + the reference adapter
`eng_ltx_av.py` (`_LtxAvBase`). Registry reads only the CORE members; the render
lifecycle is walked by `OTR_VideoRenderBatch`.

Core / identity:

```
name            = "visualizer"
family          = "abstract"          # CAPABILITIES "abstract"/"visualizer" cpu lane
roles           = ("announcer_visual", "music_visual", "character_video")
default_roles   = ()                  # never an auto-default; explicit pick only
commercial_clean= True                # own MIT code + ffmpeg only
requires_flag   = "OTR_ENABLE_VISUALIZER"   # default-OFF while dark; also display-gated by VALIDATED_ENGINES
required_inputs = ("audio_ref",)      # audio only; text_prompt optional, no init_image, no weights
render_aspect   = "wide"             # 16:9; no portrait geometry branch exists
declared_isolation = ISOLATION_IN_PROCESS
target_fps      = 25                  # HARD-LOCK, matches the overlay + mux
```

Lifecycle:

- **Extraction (the build mechanism -- do FIRST).** Do NOT instantiate the floor
  node or call its `render()`. `video_engine.py render(self, fi, draw_scopes=True)`
  is a method reading precomputed `self.volume/freqs/waves/_signal/_loss/_cards/
  total/w/h/_ring_*` -- it is stateful and owns title-card/gap logic. Extract the
  PURE per-frame draw routines (ring / particles / grid / waveform / freq bars /
  CRT post) into a torch-free `nodes/_otr_shared/scope_draw.py` that BOTH the floor
  and the engine import. The floor calls the same code it does today (zero behavior
  change -- verify with its tests); the engine calls those routines to paint a
  standalone clip, WITHOUT the title-card/gap branches. If extraction proves too
  invasive for v1, the engine COPIES the routines into its own module -- still zero
  coupling (section 0.2).
- `assert_usable(host_caps, profile, request_template=None)`: fail-LOUD, ordered:
  (1) `requires_flag` gate; (2) ffmpeg resolvable (`shutil.which` / configured
  path) else raise `EngineUnusable(... MISSING_MODEL ...)`; (3) a non-empty
  `audio_ref` path on `request_template` (None tolerated). No NVML, no weights, no
  node gate. Returns `self.name`. **NO FALLBACKS** (547671d):
  `fallback_engine = None`.
- `load()` / `unload()` / `prepare()` / `teardown()`: near-empty (no model
  residency). `load` confirms ffmpeg.
- `render_clip(request, prepared)` -- the EXACT steps (pass-02 panel caught the
  runtime crashes; do all of them):
  - (a) resolve the per-beat audio slice path from `request["audio_ref"]` (string
    or `{path}` -- reuse `_LtxAvBase._ref_path`).
  - (b) DECODE to numpy + sample_rate via `soundfile` (NOT torchaudio/torchcodec).
  - (c) **MIX DOWN TO MONO**: `if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)`
    -- `soundfile.read` returns 2D for stereo and `_analyze_audio_np` expects 1D
    (else `float(wave[j])` crashes). [gemini #2]
  - (d) `total_frames = int(request["timing"].target_frame_count)` (the BEAT
    length -- the floor analysis was whole-episode), `fps = 25`.
  - (e) `volume, freqs, waves = _analyze_audio_np(audio_np, sr, total_frames, 25)`.
  - (f) **`signal, trig, loss = _dual_ema(volume)`** -- REQUIRED: the draw helpers
    read `env["signal"]`; skip this and every frame renders permanent idle.
    [gemini #3 / deepseek #3]
  - (g) per frame `fi`, build bounded lookback windows
    `fwin = freqs[max(0, fi-_TRAIL_N+1):fi+1]`, `wwin = waves[...]` (the helpers
    take lists, not single frames) and an `env` dict carrying
    `fi/fps/key/vol/signal/loss/trig`. [gemini #4]
  - (h) paint each frame full-16:9 via the extracted routines -> RGB numpy iterator.
  - (i) encode SILENT: `_encode_silent_mp4(frames_iter, total_frames, out_path, w,
    h, 25, ffmpeg)` -- w/h from `request["canvas"]`, `out_path` a temp file,
    `ffmpeg` the path resolved in `assert_usable`/`load`. [gemini #6 / deepseek #6]
  - Return `{"out_path": path, "frame_count": n}`.
  Request fields (`canvas`, `timing.target_frame_count`, `seed_bundle.request_seed`,
  `asset_refs`, `audio_ref`, `text_prompt`) are GUARANTEED by OTR_VideoRenderBatch
  (`eng_ltx_av._build_render_request` reads exactly these). `render_aspect="wide"`
  is a duck-typed attr the director already reads (same as eng_ltx_av).
- `canonicalize(raw, request, profile)`: shape the CanonicalClip dict exactly like
  `_LtxAvBase._clip_from_raw`: `clip_id, type="video", path, container="mp4",
  codec="h264", pixel_format="yuv420p", fps=25, frame_count, has_audio=False,
  color/transfer/matrix="bt709", engine_id="visualizer", family="abstract"`.
  **has_audio MUST be False** -- only `OTR_MasterAudioMux` adds audio
  (`test_audio_byte_identical` invariant).

## 2. v1 = ONE procedural engine, NO new geometry, NO mode widget (pass-02)

There is no `_fullframe_geom`, no dual-wide layout, no preset table. The
`video_engine.py` renderer ALREADY composes a full-frame 16:9 look; inventing new
geometry contradicts "resurrect, don't rebuild."

**v1 ships exactly ONE engine** -- `visualizer` = the full-colour procedural look
(frequency ring, orbiting green/cyan/amber particles, geometric grid, mirrored
waveform, freq bars, CRT post), rendered by the EXTRACTED shared draw routines
(section 1, Extraction), NOT by instantiating the floor node. This is the look the
operator wants resurrected.

Why one engine and not a `mode` widget: the per-role dropdown selects the ENGINE;
there is no second-level mode widget, so a single engine can't know which look to
draw (pass-02 grok/deepseek). The strict GREEN-ONLY look becomes a SEPARATE
follow-up engine (`visualizer_green`, reusing `otr_scene_aware_scopes.py` helpers)
-- a clean second dropdown entry, zero mode plumbing -- AFTER v1 validates. This
also removes the green/centre overlap bug and the CRT-post-consistency question
from v1.

License stays clean (own code + ffmpeg encode only). Do NOT vendor the GPL nodes
(rhdunn, Yvann); RyanOnTheInside (MIT) is unnecessary.

## 3. CUT for v1 (panel consensus, grounded)

- **ffmpeg-native presets** (`showwaves`/`showspectrumpic`/`avectorscope`):
  `_encode_silent_mp4` is hardcoded `-f rawvideo -i -` (stdin RGB) and cannot also
  run an ffmpeg AV-filter graph (gemini, grounded). They would need a whole
  separate ffmpeg path. Cut.
- **`vectorscope`**: needs stereo; `_analyze_audio_np` is mono. Cut.
- **station-card / lower-third text**: text-gen complexity; cut.
- **`_fullframe_geom` + dual-wide**: cut (section 2).

## 4. Story-driven look (the roundtable's main creative question)

The per-beat `request` carries the full story DNA via `meta` (the
`_otr_story_brief_helpers` surface, verified):

- `get_story_brief_music_mood(meta)` -> in-vocab mood terms from a 16-term set
  (`tense, ominous, melancholic, hopeful, urgent, calm, eerie, sombre, playful,
  menacing, wistful, frantic, reverent, uneasy, stoic, yearning`).
- `get_era_tail` / `visual_palette` -> the per-episode palette colour (Mars=red,
  sci-fi=blue, period=warm). Today the overlay is GREEN-ONLY by deliberate
  constraint; as a PRIMARY picture (not an overlay) a constrained CRT-phosphor
  tint family (P1 green / P3 amber / blue) keyed to the palette is defensible --
  **roundtable decides whether to break GREEN-ONLY for the standalone lane.**
- per-beat `text_prompt`, `role`, `expression`, `motion`, `camera`.

**v1 = FAITHFUL RESURRECTION. Story-DNA mapping is DEFERRED to post-validation
(pass-02 consensus).** Ship the existing green/cyan/amber + vol/signal/loss
reactivity exactly as the floor draws it. Reasons the panel converged on deferral:
there is no tint/speed parameter on the routines today, and adding one to the
SHARED module risks the floor node's colours (coupling). Get the faithful look
green end-to-end first.

Post-validation backlog (does NOT change the adapter contract): mood term -> an
animation-speed scalar; `visual_palette` -> a deterministic phosphor-tint lookup
seeded from `request_seed`; `role` -> element-emphasis nudge. When wired, these
pass as a `visual_params` dict INTO the extracted draw routines (default values =
today's look, so the floor is unaffected). The appendix A/B/C story samples are
the design fodder for that pass.

Determinism (v1): seed the per-beat RNG from `seed_bundle.request_seed` via
`blake2s` `_rng`, so the same (audio slice, seed) is byte-identical
(regression-testable).

## 5. Wiring (CLAUDE.md section 0 -- same change as the code)

- The engine is registry-driven; selecting it is a dropdown value, so the only
  workflow-JSON consequence is that `visualizer` appears in the per-role combos
  once added to `VALIDATED_ENGINES`. Confirm `OTR_VideoDirector` rebuilds its
  combo from `validated_engine_names()` and that a saved `otr_scifi_16gb_full.json`
  with `a_video_model="visualizer"` round-trips through `OTR_WorkflowValidator`.
- Do NOT add `visualizer` to `VALIDATED_ENGINES` until the GPU/E2E smoke is green
  (it is CPU, but "validated" = proven end-to-end through the real batch +
  mux, audio byte-identical).
- Leave the §4D overlay nodes and their links exactly as-is.

## 6. Tests (run the full suite + Bug Bible after every change)

- registration: `is_registered("visualizer")`, roles/family/required_inputs,
  `default_roles == ()`.
- `assert_usable`: passes with no weights/NVML; raises LOUD when ffmpeg missing or
  audio_ref empty; no fallback.
- `render_clip` (OTR_TEST_MODE, CPU): produces a silent mp4 of the expected frame
  count at 25fps, 16:9 dims from canvas; `has_audio` False.
- `canonicalize`: exact CanonicalClip shape; `engine_id`/`family` correct.
- determinism: same (audio, seed, meta) -> byte-identical output.
- role_compat: usable for all three roles; `assert_usable` story holds.
- `VALIDATED_ENGINES` membership test (after promotion).
- naming conventions test (no "dummy"; UTF-8 no BOM); `test_audio_byte_identical`
  stays green.

## 7. Resolved decisions (roundtable pass-01 + judge)

1. **v1 = ONE colour `visualizer` engine** (full-colour video_engine look). No
   mode widget. Green-only is a SEPARATE follow-up engine `visualizer_green` after
   v1 validates (pass-02). (all 3 models)
2. **No per-role geometry, no story-DNA in v1.** Faithful resurrection; story-DNA
   (mood/palette/role) deferred to post-validation (section 4).
3. **Drop `vectorscope`** (mono analysis). (all 3)
4. **Per-story look is post-validation polish.** v1 ships the minimal mapping in
   section 4; appendix A/B/C is the design backlog, not a build blocker.
5. **No caption/lower-third** in v1. (gemini, deepseek)
6. **`requires_flag = "OTR_ENABLE_VISUALIZER"`**, default-OFF while dark (parity
   with triposr et al.). At promotion to `VALIDATED_ENGINES`, decide default-ON for
   accessibility (mirrors ltx_av). (gemini)

## 8. Verify-at-build (not blockers; check during coding)

- `otr_video_director.py` builds the per-role combo from `validated_engine_names()`
  (high confidence; read its INPUT_TYPES).
- `soundfile` imports in the ComfyUI venv for the decode (used by the sidecar audio
  path already).
