# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> Last updated 2026-06-30 | HEAD 37254f39 == origin/v2.0-alpha | branch v2.0-alpha.
> **FORWARD-ONLY / ACTION ITEMS ONLY.** Shipped/done work lives in `docs/GO_FORWARD_ARCHIVE.md` --
> do NOT record done work here. prod/main + tags remain operator-GATED.

## 1. CURRENT STEP -- ALL-ENGINES x ALL-SLOTS FIX: C0-C5 CODE SHIPPED; remaining = LIVE-GPU soak RUN

The slot-audit sprint (`docs/2026-06-30-slot-audit/SPRINT_PLAN.md`) C0-C5 is BUILT, green
(suite 5851/0 + Bug Bible + B7), and PUSHED to v2.0-alpha (8f701a73, 65c11bc1, 96aa54dc, ca2ac0e8,
f5b78ac5). What landed:

- **C0 SHIPPED (8f701a73):** RETIRED `station_card` + `abstract` engines (operator directive: both
  redundant -- visualizer covers it, visualizer_rainbow is the planned creative slot). Deregistered +
  CAPABILITIES rows removed + render_driver/soak dead-name constants repointed + profile JSONs
  (8gb_lite/cpu_floor announcer -> still_flat). The `abstract` FAMILY name survives (visualizer).
- **C1 SHIPPED (8f701a73):** `accepts_still=True` on StillPan + StillMotion families -> the image
  dispatcher MINTS their scene still (D2 BLACK fix). Bind set NOT generalized (HuMo excluded).
- **C2 SHIPPED (65c11bc1):** `VideoEngineRegistry(EngineRegistry)` overrides `engines_for_role` +
  `assert_usable` to use `role_compat.engine_fits_role` (capability); fail-soft to legacy `roles` ONLY
  when required_inputs missing/None or role unknown, NEVER for `()` (D1 drift killed).
- **C3 SHIPPED (96aa54dc):** `sfx` speaker_role (the only unmapped writer token) -> `scene_broll` in
  SPEAKER_TO_VIDEO_ROLE, so the scene_broll slot is routable (D4).
- **C4 SHIPPED (ca2ac0e8):** `tests/test_video_role_eligibility_matrix.py` -- all engines x 5 roles,
  eligibility == capability (not flat-True), + the `()`-fits-all and None-falls-to-legacy proofs.
- **C5 SHIPPED (f5b78ac5):** `nodes/_otr_shared/content_oracle.py` (per-beat luma floor + freeze
  detect, motion-required by family, statics exempt) + `nodes/_otr_shared/slot_matrix.py`
  (`build_all_five_role_profile` -- sets all 5 INDEPENDENT role keys via the applier, drops the legacy
  other_beats fallback) + `tests/test_slot_matrix_soak.py` (OFFLINE proof on the canonical JSON + real
  ffmpeg oracle fixtures). The gitignored `_otr_combo_soak.py` runner was converged locally to the
  all-5-role builder (imports the tracked builder).

REMAINING (GPU-gated, NOT code): RUN the live all-engines x all-slots soak -- boot headless ComfyUI,
load `otr_scifi_16gb_full.json`, apply the all-5-role profile (`slot_matrix.build_all_five_role_profile`),
render a leg per engine, and run `content_oracle.check_manifest` on the per-beat manifest clips. ALSO
optionally finish converging the other two gitignored runners (`scripts/otr_coverage_sweep.py` SLOTS is
still 3-role; `scripts/_otr_cov_runner.py`) onto `build_all_five_role_profile`. The accelerator for this
is S-F (the visual smoke fixture) so each leg is minutes not ~28 min.

ACCEPTANCE (met in CODE; live RUN proves empirically): every video+still engine eligible by capability
AND renders real content in all 3 slots (no black floor; static OK for static engines); audio
byte-identical; no shim. VERIFY-AT-BUILD checklist in the SPRINT_PLAN -- all green.

## 1A. CONSOLIDATED NEXT STEPS (post-soak sprints, priority order)

> **WIRE-READY hardened plan (kibitz r1-r4 CONVERGED 2026-06-29): `docs/2026-06-29-coverage-soak/SPRINT_PLAN.md`.**
> Grounded corrections that supersede the summary bullets below: BUG-411 already done (look-QA seed check
> only); no-fallback already enforced at render time (S-E = scaffolding cleanup + DEPRECATE
> `allow_auto_fallback`, not a behavior change); recipe-stamp extends `meta.render_engines`; S-F = a pruned
> API prompt + baked asset bundle (ComfyUI MCP executes it); HuMo cfg is NOT a regression (mush = S-A clip
> underrun); engine-retirement is separable/deferrable. The SPRINT_PLAN is the coder-window contract.

Every sprint: video/content-only where noted; single resident heavy <= 14.5 GB; seed-keyed
determinism; LOUD fallbacks; master audio byte-identical (`test_audio_byte_identical` GREEN);
suite + Bug Bible + B7 green AND push per green chunk to v2.0-alpha. Keep EVERY engine
user-selectable -- these are QUALITY FLOORS, never choice-limiting.

> ### BUILD-READY QUEUE (for a fresh coder window -- refreshed 2026-06-30 post-item2-ship)
> These have coder-ready contracts; a new window can pick them up in this order via the otr-handoff skill:
> 1. ~~viz_mxc_mandala~~ -- **SHIPPED `8d90562a`** (2026-06-30): `eng_viz_mandala.py` + scope_draw
>    `paint_mandala`/`mandala_surface_to_rgb`/`apply_crt_post_rgb` + full wiring
>    (`__init__`/`ENGINE_FAMILY`/`_uses_ambient_master_audio`/`content_oracle`/`otr_video_soak.py`
>    ENGINE_FAMILY/`registry.CAPABILITIES`) + 13 new tests (`tests/test_video_viz_mandala.py`). Also fixed
>    the pre-existing `viz_mxc_cpu` latent gap in `otr_video_soak.py` ENGINE_FAMILY (never added when THAT
>    engine shipped). Suite 5884 passed/35 skipped/0 failed + Bug Bible 16/0 green; pushed, HEAD==origin.
>    No workflow-JSON change needed (opt-in selectable, no node-87 default -- operator may set it later).
> 2. ~~still_parallax 100% rip-out~~ -- **SHIPPED** (2026-06-30, same commit as item 3): deregistered by
>    dropping the `@register` decorator in `eng_still_parallax.py` -- NOT just the `__init__.py` import; a
>    bare direct import re-registers a class that still carries `@register`, which is what the dark-scaffold
>    pattern actually needs and what caught 2 test regressions mid-ship (the plan text below undersold this).
>    Also dropped: its CAPABILITIES row, `render_driver.ENGINE_FAMILY` entry, and the
>    `otr_video_dep_pilot.py` PROBE_ENGINES row (a no-drift contract test asserts every probe entry is
>    registered). Fixed the two dangling `fallback_engine="still_parallax"` refs (`eng_mesh_stage.py` /
>    `eng_triposr.py` now point directly to `still_motion`, a 1-hop chain). Rewrote
>    `tests/test_video_still_parallax.py` to direct-instantiation (triposr precedent) + fixed ripple in 6
>    more test files the first pass missed (`test_capability_profiles` / `test_image_platform_c1` /
>    `test_route_a_14b_promotion` / `test_slot_matrix_soak` / `test_video_dep_pilot` /
>    `test_video_render_driver_perbeat_audio`). Source file kept on disk (dark scaffold, like
>    triposr/character_3d). `content_oracle._FAMILY_FALLBACK` + `otr_video_soak.ENGINE_FAMILY` KEPT their
>    still_parallax entries ON PURPOSE (historical-manifest audit tables, not live dispatch).
> 3. ~~visualizer -> viz_green rename~~ -- **SHIPPED** (2026-06-30, same commit as item 2): engine `name` +
>    CAPABILITIES key + `render_driver` ENGINE_FAMILY/`_uses_ambient_master_audio` + `content_oracle`
>    FAMILY_FALLBACK + `otr_video_dep_pilot.PROBE_ENGINES` + all code/test refs renamed to `viz_green`.
>    `otr_video_soak.py` ENGINE_FAMILY was ALSO missing a visualizer/viz_green entry (a second latent gap,
>    same shape as item 1's viz_mxc_cpu miss) -- closed in this same chunk. `_LEGACY_ENGINE_ALIASES` gained
>    `"visualizer": "viz_green"`. The 3 profile JSONs + node-87 in `otr_scifi_16gb_full.json`
>    (`widgets_values[0:3]`) updated together (workflow-JSON change in the SAME commit as code). Python
>    module/file/class names (`eng_visualizer.py` / `VisualizerEngine`) intentionally UNCHANGED -- only the
>    registered engine_id was in scope. Suite 5878 passed/35 skipped/0 failed + Bug Bible 16/0 + B7 green;
>    pushed, HEAD==origin.
> 4. ~~HuMo improvements~~ -- **SHIPPED 2026-06-30** (`docs/2026-06-30-humo-improve/HUMO_IMPROVE_PLAN.md`
>    BUILD-READY items 1/2/4/5 -- see the detail bullet below for the full receipt; phrase-chunking (item
>    1b, "the deepest change") and the HuMo-isolation smoke (portrait-knob A/B prerequisite) are explicitly
>    SPLIT OUT as a follow-up per the plan's own suggestion, not bundled here). Suite 5889 passed/35
>    skipped/0 failed + Bug Bible 16/0 + B7 green; pushed, HEAD==origin. No workflow-JSON change: the
>    canonical `otr_scifi_16gb_full.json` node-87 announcer/music defaults were ALREADY `viz_green` (not
>    `humo`) going in, so nothing there violated the new policy -- the fix is a structural CODE guard
>    against a future dropdown pick or `OTR_FORCE_ENGINE_MAP` override, not a default-value change.
>    Whether to promote `ltx_audio_in` (a HEAVY ~13.7GB engine) over the free `viz_green` as the shipped
>    aesthetic default is a SEPARATE, GPU-smoke-gated decision, deliberately not made here.
> 5. ~~mesh_stage MIN-ACCEPT~~ -- **SHIPPED 2026-06-30** (KIBITZ r1 `docs/2026-06-30-mesh-improve/
>    MESH_STAGE_IMPROVE_PLAN.md`, r1_judgment.md's 4 LOCKED points, r1-only scope). (1) radio subject
>    minted in PROMPT-GEN: `_mesh_fodder_subject` (`otr_meta_brief_image_prompt.py`) gained a
>    `role=="music_visual"` branch returning a vintage-radio phrase instead of the old generic "an
>    emblematic object representing X" (an arbitrary, unrelated prop). (4) identity continuity: EVERY
>    music_visual fodder beat (open/inter/close) now shares ONE canonical `MESH_RADIO_HOST_SUBJECT_ID =
>    "radio_host"` mesh-cache id (keyed on the always-present video ROLE, never a line/speaker_role
>    lookup -- robust to lineless synthetic bookend beats); character/announcer fodder ids UNCHANGED
>    (verified via a live debug run, not guessed -- announcer's pre-existing `obj_<beat>` id is a
>    separate, out-of-scope gap). (2) MEASURABLE headroom contract: the Blender camera's vertical FOV is
>    now PINNED (`sensor_fit='VERTICAL'`, fixed lens/sensor-height -- Blender's own factory numbers, a
>    determinism pin not a behavior change) so `adaptive_camera_radius(mesh_height, target_frac=0.62)`
>    (new pure fn, `scripts/otr_mesh_stage_blender.py`) has a well-defined trig basis; `_normalize_meshes`
>    now also returns the mesh's post-normalize height so the camera distance is SHAPE-AWARE (a tall
>    character and a squat radio get different, correct distances) instead of the old flat
>    radius=2.5/elevation=0.35 (which over-framed a longest-dim-1.0 tall mesh to ~84% of frame height --
>    verified via `test_adaptive_camera_radius_old_flat_default_was_over_tight`). `--radius`/
>    `--elevation`/`--target-height-frac` default to `None` (adaptive) with an explicit-override escape
>    hatch threaded through `eng_mesh_stage.build_blender_cmd`/`render_clip`
>    (`OTR_MESH_STAGE_RADIUS`/`_ELEVATION`/`_TARGET_HEIGHT_FRAC` env vars, mirroring the existing
>    start_angle/arc_degrees pattern) -- byte-identical legacy invocation when unset. Also added a pure
>    `alpha_bbox_stats` fn (the MEASURABLE contract's bbox/margin math, kibitz r1 point 2's "proof-frame
>    bbox test") operating on a top-origin alpha buffer (PIL `getchannel("A")` convention) so a future
>    GPU-rendered-frame proof script can reuse it; the bpy-dependent wiring (camera pin, `_build_turntable`/
>    `main()` changes) and the actual before/after GPU proof render are OPERATOR-GATED (no GPU/Blender
>    access this session) -- only the pure trig/bbox math got unit coverage here. (3) Routing: confirmed
>    via a live JSON probe that `otr_scifi_16gb_full.json` node-87 selects NONE of mesh_stage's roles
>    today (announcer/music/other_beats=`viz_green`, character_video=`humo_14B_169`) and
>    `OTR_FORCE_ENGINE_MAP` is unset -- so NO workflow-JSON change was needed; mesh_stage stays reachable
>    only via an operator dropdown pick or a `OTR_FORCE_ENGINE_MAP` override. Kibitz r1 SCOPE explicitly
>    CUT Trellis/WorldMirror + broad material/lighting/turntable exploration (separate, operator-gated) and
>    the r2-r4 rounds (r1-only per operator). 17 new/updated tests across `tests/test_video_mesh_stage.py`
>    + `tests/test_3d_image_streams.py`. Suite 5904 passed/35 skipped/0 failed + Bug Bible 16/7/3 green;
>    pushed, HEAD==origin.
> 6. **S-A..S-F coverage-soak sprint** -- KIBITZ r1-r4 CONVERGED `docs/2026-06-29-coverage-soak/SPRINT_PLAN.md`. <- NEXT
> Invariants for all: single resident heavy <= 14.5 GB; audio byte-identical; no-fallback (hard-fail LOUD);
> UTF-8 no BOM; SFW; workflow-JSON edited in the SAME change as code; suite+BugBible+B7 green + push per chunk.

- **S-A [HIGH] DELIVERY-QUALITY FLOOR (clip-fill + legibility).** NOT a routing bug -- a
  short/dead generated clip is allowed to ship. Grounded + REPRODUCED on two episodes
  (`weight_of_the_blueprints_163656`, `steel_against_skin_170522`): the announcer portraits
  ARE present at render (`[portrait_ledger] still_b001/b005 ... recorded via ledger['images']`),
  but `humo_1.7B` UNDERRUNS -- `CLIP UNDERRUN: shot_b005 rendered 177 frame(s) for a 434-frame
  target (41%); the composite will HOLD the last frame for the rest of the beat`. The held
  static last-frame IS the murky/dead plate (177 = HuMo per-clip frame ceiling vs long 405-434f
  announcer beats). Completion gates (obs ships, audio byte-identical) PASS regardless. Fix,
  priority: (1) **clip-fill** -- a motion engine that underruns LOOPS / ping-pong-extends to the
  target (the composite's OWN recommendation), never holds the last frame; (2) **legibility
  guard** after each clip (sharpness RATIO vs source -- relative/catastrophic only; motion via
  freezedetect; face-presence = phase 2); (3) on failure composite the clear still + subtle
  parallax; (4) record `attempted_engine`/`delivered_engine`/`fallback_reason` via the EXISTING
  humo->still_parallax LOUD restamp. SECONDARY/forensic (aids diagnosis, NOT the cause):
  preserve `ledger['images']` durably (`production_ledger._merge_with_disk` drops top-level
  `images`) + stamp per-beat `init_image_used`/`init_source`. (HuMo phrase-chunking in S-C
  attacks the same underrun root.) Detail: `docs/2026-06-29-coverage-soak/RETEST_LIST.md` B2.
- **S-B ltx_audio_in VRAM FIT (regression).** Breaches the 14.5 GB ceiling (~15.9 GB,
  `eng_ltx_av.py:687`) and hard-fails in all 3 slots. Regression: `7bbce1d8` "bakeoff-winner
  quality upgrade (PROVISIONAL)" + `fd9edc28` switched to dev-Q3_K_M + SHARP LoRA (~15.5 GB);
  last-good = `c4d7815b` base recipe @ 512x288 = 13688 MB. FIX-FIRST = observability (per-beat
  recipe / unet / quant / LoRA / canvas / frames / audio-source / phase-marker / peak VRAM),
  THEN re-fit via recipe/quant/offload (`OTR_LTX_AV_RECIPE` / `distilled_native` / lighter
  quant) -- NOT higher resolution. Quality/resolution tiers LAST, probe-gated. Replace the
  stale 13688 comment with "see runtime logs / bakeoff manifest".
- **S-C AUDIO-IN CONDITIONING SPRINT.** Shared per-beat `audio_motion_profile` (rms / peak /
  onset / silence / brightness / dynamic-range / speech-vs-music / duration) driving EVERY
  engine -- audio-in engines get real audio, non-audio engines get prompt / camera / parallax /
  light from the profile; normalized conditioning WAVs (model-input only, master untouched);
  HuMo phrase-chunking for long dialogue (vs mirror-extending the 49-frame cap, `eng_humo.py:61`);
  probe-gated HQ tiers last.
- **S-D gemma normalize_length wrapper-key drift.** Every gemma episode: the model returns the
  RadioEditPlan nested under a top-level `RadioEditPlan` key -> `projected_word_total` "missing"
  -> retry ladder exhausts -> length normalization skipped (warn-only). Fix the LEVER-1
  tolerant-unwrap to peel a top-level schema-name wrapper; retest on a gemma leg.
- **S-E NO-FALLBACKS + ENGINE-MENU + UX CLEANUP (operator directives 2026-06-29).** Detail +
  receipts in `docs/2026-06-29-coverage-soak/RETEST_LIST.md` (Section 0 + B3 + B5 + 0c). Bundle:
  - **NO FALLBACKS / hard-fail:** a selected engine RENDERS or raises a LOUD hard error -- never a
    silent degrade to stills (the ltx_audio_in "looks like stills" + the black-floor carrier bug).
    Rip out the fallback chains / `resolve_fallback_chain` / `SYNTH_FALLBACKS`. (S-A's legibility floor
    becomes detect-and-fail/flag, NOT a still-swap.)
  - **RETIRE engines:** `still_motion` (the fallback-floor twin of still_pan -- falls away with the
    rip-out), `station_card` (broken black card, missing `accepts_still`), and `abstract` (redundant
    with the real `visualizer`). Unregister + workflow-JSON dropdowns + ripple tests (the C3 pattern).
  - **`viz_mxc_cpu` (rainbow visualizer) -- SHIPPED this session** (b01d2363; green + pushed). The
    creative rainbow replacement for retired `abstract`. Operator was under-whelmed by the PIL look ->
    the pycairo MANDALA is the upgrade (next bullet).
  - **ADD `viz_mxc_mandala` (Cosmic Radio Mandala, pycairo) -- BUILD-READY, KIBITZ r1-r4 CONVERGED
    2026-06-30.** Coder contract: **`docs/2026-06-30-viz-rainbow/MANDALA_ENGINE_PLAN.md`** (per-round
    judgments in `kibitz-runs/2026-06-30-mandala/`). Separate engine (keeps viz_mxc_cpu as zero-dep
    alternate); `fallback_engine=None` + fail-loud (no-fallback contract, render_driver.py:1531);
    pycairo lazy-imported + isolated (NOT in requirements.video.txt); assert_usable probes cairo AND
    ffmpeg; `render_aspect="wide"` + `declared_isolation=ISOLATION_IN_PROCESS`; surface.flush()+get_stride
    ->rgb24; new `apply_crt_post_rgb(rgb,scanlines,vignette,fi,rng_key,vol)` helper in scope_draw; full
    _DECL_KEYS CAPABILITIES row; opt-in selectable (NO node-87 default -- operator may set the music/title
    bookend widget later). Ring/band coefficients = a build-time LOOK pass WITH the operator (denser look
    approved 2026-06-30), not a frozen constant.
  - **(historical) `viz_mxc` locked decisions** (operator: must run on
    AMD/Mac/any box -- NO GPU shaders for v1): ONE engine `viz_mxc_cpu`, `required_inputs=()`
    (audio-OPTIONAL -> reactive when audio present, procedural OTR rainbow when not = ALSO the no-image
    floor for scene/background; `accepts_still=False` so it never triggers z_image on a non-audio slot);
    CPU **numpy/PIL paint -> `scope_draw.encode_silent_mp4`** ONLY (NO ffmpeg showcqt/showspectrum --
    breaks the silent/frame-count contract); reuse `scope_draw.analyze_audio_np`; new shared
    `paint_rainbow_frame`. OTR MYSTIQUE look (radio dial / tube / magic-eye / signal-spectrum sweep, muted
    not neon, reuse `build_vignette`+`build_scanlines`+grain). WIRING must-fixes (same chunk): add to
    `_uses_ambient_master_audio` + `ENGINE_FAMILY` + `content_oracle._FAMILY_FALLBACK`; CAPABILITIES row;
    auto-derived label (NO custom label -- breaks `_engine_id_from_pick`); node-87 promotion is a separate
    operator-gated chunk that updates the pinned `test_workflow_live_passes_validator` + 16gb_full profile
    in one commit. `viz_mxc_gpu` (torch-compute, NVIDIA-first, fail-closed) DEFERRED to a later opt-in
    spike -- never blocks the CPU tier. Build order C-mxc1..C-mxc3 in the plan. SCHEDULE: after the current
    all-engines all-slots sweep/soak completes.
  - ~~RETIRE `still_parallax`~~ -- **SHIPPED 2026-06-30** (operator verdict: "kinda weird but sucks").
    See BUILD-READY QUEUE item 2 above for the full receipt (the actual retirement also had to drop the
    `@register` decorator itself, not just the `__init__.py` import -- the plan text here undersold that
    part; corrected in the queue entry for the next reader).
  - ~~RENAME `visualizer` -> `viz_green`~~ -- **SHIPPED 2026-06-30** (same commit as the rip-out above).
    See BUILD-READY QUEUE item 3 above for the full receipt.
  - **KEEPER-ENGINE LOOK-QA IMPROVEMENTS (operator soak eyeball 2026-06-30) -- plans drafted, build
    post-soak:**
    - **HuMo (KEEP; 1.7B class + same VRAM):** `docs/2026-06-30-humo-improve/HUMO_IMPROVE_PLAN.md` --
      **SHIPPED 2026-06-30** (see BUILD-READY QUEUE item 4 above for the full receipt): clip-underrun "all
      mush" fix (`otr_silent_composite._should_loop_fill` now EXCLUDES `audio_driven_face` rows from
      loop-fill -- looping a talking face desyncs the mouth from its audio, worse than a held frame; the
      LOUD `held_last_frame` legibility guard still fires); announcer/music bookend = RADIO not a face
      ("radio is the host", reversing Route-A 2026-06-28) via a NEW `render_driver._enforce_radio_is_host`
      guard that wires the previously-dormant `_otr_speaker_role.is_never_humo_role` into real dispatch and
      redirects any HuMo-family pick on those 2 roles to the existing `ltx_audio_in` engine (proven-good in
      a live check: it flows through the pre-existing role-driven LTX motion-prompt path and produces an
      animated radio-CONSOLE prompt, not a generic scene); `eng_humo.HuMoEngine.roles` narrowed to
      `("character_video",)` (self-description only -- role_compat eligibility is capability-based and
      unaffected); HuMo dropdown labels now auto-append a VRAM-tier suffix (see DROPDOWN LABELS below).
      STILL OPEN (deliberately split out, per the plan's own note): phrase-chunking (the deepest/real fix
      for clip-underrun -- render long dialogue in speech-aligned chunks within the 177-frame cap) and the
      HuMo-isolation smoke (prerequisite for portrait-quality-knob A/B tuning only, not for the above).
    - **mesh_stage (KEEP -- "one must-have"):** `docs/2026-06-30-mesh-improve/MESH_STAGE_IMPROVE_PLAN.md`
      -- MIN-ACCEPT: opening/music beat = a 3D RADIO (not a body) + MORE HEADROOM (subject centered, not
      full top-to-bottom). Optional r1-only kibitz for further quality. Trellis is a musing-stage
      alternative 3D backend (3D-path decision still operator-gated).
  - **DROPDOWN LABELS:** every option states model + variant + recipe + VRAM tier. **VRAM-tier suffix
    SHIPPED 2026-06-30** (BUILD-READY QUEUE item 4): `registry.vram_tier_label` auto-derives ``' (~N.NGB)'``
    from the CAPABILITIES table's `vram_estimate_mb` (never hand-maintained), appended in
    `otr_video_director._label_for` after the existing aspect suffix -- e.g. `humo_1.7B (portrait)
    (~6.8GB)` / `humo (portrait) (~13.7GB)` (the earlier "~3.3 GB / ~15.9 GB" figures floated in this doc
    were STALE placeholders -- these registry-grounded numbers are the real ones; KEEP BOTH tiers, a real
    low/high split, operator 2026-06-29). STILL OPEN: which LTX / Wan i2v/ti2v / image model each label
    should additionally spell out beyond the auto-derived id+aspect+VRAM (e.g. recipe/quant detail); the
    "abstract/visualizer(now viz_green) = audio-reactive, no scene image" wording still needs its own
    label copy pass. Grounded label copy in RETEST_LIST B5.
  - **STAMP THE RECIPE IN THE LEDGER:** per-beat `delivered_engine` (video) + `image_engine` +
    `recipe/quant`, durable through `_merge_with_disk`, so every episode self-documents what made it
    ("what did I use?" is unanswerable from saved files today). Durable twin of no-fallbacks + labels.
  - **ANNOUNCER + MUSIC = always a radio-themed still** (vintage radio in the scene) -- never a black
    card / abstract / fallback; the sane on-brand bookend default. Image-prompt change (+ B4 framing).
- **S-F [ACCELERATOR] VISUAL SMOKE FIXTURE -- bake story+audio+ledger, run stills+video only (operator
  2026-06-29).** Today every coverage leg re-runs the WRITER (gemma, minutes) + the full AUDIO path (TTS +
  music) just to test a VISUAL engine -- that is most of the ~28 min/leg, and a DIFFERENT story each run
  (not apples-to-apples). Build a smoke that BAKES one good 30-word episode's master audio + story ledger
  (cast / brief / beat structure / per-beat durations / portrait hashes) ONCE into a fixture, then INJECTS
  it and runs only the stills -> video -> composite -> mux tail. Each engine test then swaps ONLY the
  image/video engine and re-renders the cheap visual tail -> minutes not tens; identical story+audio every
  run = a clean per-engine eyeball; master audio byte-identical for FREE (same baked WAV). Seam = the MIRROR
  of the existing audio-only soak (which PRUNES the graph at node-7 `EpisodeAssembler` to skip video) -- this
  starts FROM that boundary (the frozen audio + `/otr/latest_ledger`) and skips the writer + audio nodes.
  ACCELERATES the current soak + the GATE-A sweep + every future engine test -- schedule EARLY (before
  grinding the rest of the matrix by hand). Keep it a TEST harness only (no production-path change; the real
  workflow still writes a fresh story each episode).

**DEFERRED BACKLOG** (still pending, lower priority -- detail in sections 3-5 + the bug logs):
GATE-A punch list incl. BUG-411 (flux lost its cinematic tint -- restore FluxGuidance ~3.5 +
the cinematic/radio suffixes + bookend seed) and look-QA; Wan i2v back-burner (drifts off the
still; LTX holds -- Path B two-expert only if revisited); 3D path decision (WorldMirror
multi-view vs TripoSG object-mesh -- BLOCKED on operator pick); switchable distribution S3-S6.

Older ACTIVE/SUPERSEDED step history -> `docs/GO_FORWARD_ARCHIVE.md`.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (operator, hard):** `workflows/otr_scifi_16gb_full.json` IS the
  production workflow. (1) ANY node / wiring / widget change MUST be made IN that file in the SAME
  change as the code -- code that is not wired into this JSON is DORMANT and does nothing (the §4D
  miss, 2026-06-13: node + blend input shipped + tested but unwired -> ran dead in production). After
  editing, re-validate via `OTR_WorkflowValidator` + a JSON round-trip + the link/widget audit.
  (2) EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot (the sandbox mount lags file writes; always
  read/write the Windows path + verify via Desktop Commander).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8) -- not story-spine, not
  story-pipeline, not the broader audio stack, not other ROADMAP items.
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream character-voice
  "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline; determinism
  seed-keyed (per-seed within a render, NOT run-to-run); every in-render fallback LOUD; UTF-8 no BOM;
  SFW; V-12 dep isolation; no new widgets in the static workflow shell (V-11).
- GIT (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green chunk; the
  operator eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no 0-byte / no BOM /
  AST parse on touched .py. prod/`main` is GATED until operator work is done (a `v2.0-alpha-stable`
  tag on `v2.0-alpha` is fine).
- EVERY session updates this doc + the `otr-build-tracker` dashboard (content; keep the gauge+lanes
  styling).
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs must log
  `cast RNG seed=... (OS entropy)`. Do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in sequence)

> **Two tracks, parallel.** Item 1 (punch-list audit) is OPERATOR-GATED (look-QA -- section 5); the
> ENGINE track (items 3-4, Wan + sweep GREEN) proceeds NOW. "In sequence" applies WITHIN a track, not
> across the operator gate.

1. **Punch list (GATE A) -- OPERATOR APPROVED 2026-06-21, proceed.** Captions DONE (node 86
   `OTR_CaptionBurn` in `otr_scifi_16gb_full.json`, profile resolves `burn_captions=True`). REMAINING:
   node-level audit of LTX radio-open + procgen rolling credits -- baked into the headless path but maybe
   NOT into the saved JSON; prove a render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED 2026-06-21 (operator: "we ripped it out").** Verified: NO engine file under
   `nodes/_otr_video_engines/`, 0 references in `otr_scifi_16gb_full.json`, only a few stray comment/env
   strings remain (`OTR_LSYNC_BASE_ENGINE`). Not a live lane -- dropped from the forward order. (A trivial
   code-comment scrub of the stray strings can ride any future cleanup; not a roadmap item.)
3. **Wan 2.2 video engine (section 4) -- OPERATOR APPROVED 2026-06-21 ("100% approved"): proceed with the
   eyeball + acceptance.** BOTH engines BUILT + validated live (2026-06-14, `bcbe05a`):
   wan_i2v (14B, post mixin-refactor) + the new wan_ti2v (5B/GGUF 8GB tier). REMAINING = the operator
   WEBM EYEBALL (14B vs 5B) + the optional formal full-episode `--acceptance` GREEN exit (slow
   wan-music-bed leg, run attended) + the M9 CS-3 instrumented proof. Code-complete; gates are the
   operator's.
4. **Coverage sweep GREEN (GATE A acceptance).** Re-run the permutation matrix after the soak fixes.
   Matrix (additive, not cross-product): a visual-engine leg-set (varies each of music/announcer/
   other_beats), a writer-LLM leg-set (varies node-1 `creative_writing_model`/`technical_model`), and a
   curated voice-variation leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no
   seed pins). **Wan is a CORE/BLOCKING engine** -- the sweep is NOT green until `wan_i2v` (and
   `wan_ti2v`) pass, so it stays RED until item 3 lands; that is expected. This re-run also answers the
   one open R2 question: whether `humo_1.7B` renders NATIVE char beats at 70w once its enable flag is on
   (the soak floored it only because the flag was off). **GATE-A precondition: harden the
   sweep FIRST (section 4A M1-M4) -- DONE 2026-06-13: the M1-M5 acceptance gate landed
   (`scripts/otr_coverage_sweep.py --acceptance`), so a silent fallback / empty-results
   run / missing VRAM measurement now scores RED, not GREEN.**
   **S6 harness reality:** `otr_coverage_sweep.py` enumerates ONLY the visual-engine
   leg-set today (the dropdown rotation). The writer-LLM leg-set (node-1
   `creative_writing_model`/`technical_model`) and the curated voice-variation leg-set
   are NOT yet wired into a runnable harness -- TODO: point them at a real driver
   (e.g. a `run_combo_matrix.py`) or run them as separate parametrized soak legs.
   "Coverage sweep GREEN" today means the visual-engine set only.

   **SOAK READINESS AUDIT (2026-06-13).** Walked the registry + harness. Conclusion:
   **clear to run a wan_i2v-only soak today** (no wan_ti2v hard prereq for validation).
   Verified live: `wan_i2v` enumerates `ok`/runnable under `16gb_full` (legs
   `music_visual=wan_i2v` + `other_beats_visual=wan_i2v`) -- the old "add wan_i2v to the
   enable-set" note is STALE/resolved. 27 legs enumerate; the only skips are
   `hunyuan3d_talk`/`trellis_talk` (missing cu128 toolchain, expected darks). Wan models
   on disk + `OTR_ENABLE_WAN_I2V=1` env known. **Two limitations to know:**
   (i) `--acceptance` exit is RED-by-construction until `wan_ti2v` is built (M2 requires
   BOTH Wan engines) -- expected; read the per-leg verdicts in `coverage_sweep_summary.json`,
   the wan_i2v leg PASS/FAIL is the meaningful signal.
   (ii) **The M1 no-fallback (CS-1) gate is bound to `--acceptance`** (`forbid_fallback=
   args.acceptance`); the capstone CLI does not expose it. So re-running the NON-Wan
   permutation soak (the set that originally false-greened) WITH the M1 fix active and a
   clean GREEN/RED exit needs either `wan_ti2v` built OR a small **`--strict-fallback`**
   flag that decouples M1 from the Wan-engine requirement (~10 lines; RECOMMENDED, optional
   -- operator's call). Until then: `--acceptance --only wan` exercises M1 on the wan_i2v
   legs (overall RED expected), and a non-acceptance sweep runs but with M1 OFF
   (informational). No half-built code, no missing capability rows beyond the deferred
   `wan_ti2v`, no broken tests (the 2 `test_model_catalog_scan` reds are pre-existing /
   environmental, tracked separately).
5. **3D sprints.** s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the `character_3d` family
   (image-routing must-fixes already landed). Detail in the 3D plan (pointers).
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing phase).

**0-E parallel track:** `ltx_orbit`/`still_parallax`/`mesh_stage` CPU side shipped + all three GPU-green;
Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the `scripts/_otr_0e_gpu_go.txt` GO file.

**Audio parallel track (own window, never blocks video):** the character-voice "whiny" fix (upstream TTS
only; frozen spine untouched). Operator note: may have self-resolved -- verify before scheduling work.

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only (lip-sync stays SEPARATE
on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ wrapper (KJ drags in SageAttention + a numpy<2 pin
this box violates). Phase 1 + the 5 code-gap fixes are DONE (`2fbc2f3`); the full grounded spec is in
that commit + git history of this file.

- **Phase 2 -- 16GB engine leg.** Drive `eng_wan_i2v.render_clip` via the real path
  (`scripts/otr_run_leg.ps1` / `coverage_sweep --only ...`). ASSERT `wan_i2v` is the final_engine in the
  trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <= 14.5 GB + byte-identical audio mux + silent
  mp4 (h264/yuv420p/bt709, fps 25, `has_audio` False). Kill/reset the Phase-1 server first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF (Q6/Q5_K_M) + the wan2.2 VAE into
  `C:\ComfyUI-Models\` (record HF repo + sha256 + license, fail-closed). Define a NEW `wan_ti2v` engine
  (own flag/model/VAE env, registry registration, `_node_candidates` incl. the 5B latent node, loader
  mode, `canonicalize`, profile hook + tests) -- do NOT alias `WanI2VEngine`.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar = real camera motion, still preserved, no warp.
  **S3 motion risk to watch:** the wired I2V-14B fp8 is a SINGLE low-noise expert (the
  two-expert HIGH/LOW MoE handoff, Path B, is NOT wired -- see `eng_wan_i2v` header). If
  the "real camera motion" bar FAILS (motion too subtle / static), the Path B two-expert
  HIGH/LOW handoff is the mitigation, not a knob tweak. Call this out at the eyeball.
- **Risk CS-3 (reframed):** sequential-residency, NOT co-residency -- see section 4A M9
  and the section-5 CS-3 entry. The supervised Wan batch proves the inter-beat reclaim,
  it does not "decide if they co-stage."

---

## 4A. WAN + GATE-A SWEEP HARDENING (roundtable 2026-06-13, grounded vs HEAD 134f8e2)

Folded from a 3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude's
grounding against the real code. Full judgment + raw reviews:
`docs/2026-06-13-goforward-wan-hardening/`. These gate item 3 (Wan) and item 4 (sweep
GREEN). MUST-FIX -- until M1-M4 land, a GREEN sweep is meaningless:

> **STATUS 2026-06-13 (autonomous build) -- LANDING LEDGER:**
> - **M1 + M4** `9b2294b` -- no-runtime-fallback gate + VRAM fail-closed (12 tests).
> - **M2 + M3 + M5** `0ab55bc` -- sweep `--acceptance`: empty/required-engine exit
>   code + Wan enable-flag / OTR_TEST_MODE / --exclude preflight (17 tests).
> - **M6** `ec91a3c` -- `assert_usable` preflights UNET + umt5 CLIP + VAE (8 tests).
> - **M7** `f71edaa` -- render_clip ffprobe-PROVES the silent-clip contract (13 tests).
> - **S1 + S5** `dfe9ab5` -- wan_i2v vram_estimate 14500 + real wan2.2-i2v asset id.
> - **S7 + S10** `f3a529f` -- per-shot/seed init staging + Pillow-required fail-loud.
> - **S3 / S6 / S8** -- folded into this doc (MoE eyeball risk, sweep-harness reality,
>   the exact acceptance invocation below).
>
> **M8 + S2 -- LANDED 2026-06-14 (`bcbe05a`).** The `wan_ti2v` engine is built: its 5B core
> node class (`Wan22ImageToVideoLatent`) was captured from a live `/object_info` first; M8 raises
> `EngineUnusable` when the resolved VAE basename is empty or is the 2.1 VAE; S2 added the
> `medium`/8000 CAPABILITIES row (registry-consistency invariant holds -- the row + the registered
> engine landed together). Validated live (5B bare-graph smoke PASS). **STILL OPEN:** **M9** (CS-3
> sequential residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are live-GPU
> proof obligations -- partial evidence only. A full multi-leg `--acceptance` GREEN exit is gated on
> the slow wan-music-bed leg (run it attended/selectively), not on missing code.
>
> **S8 -- exact acceptance invocation** (ComfyUI venv python; live server on :8000;
> `OTR_TEST_MODE` UNSET; `OTR_ENABLE_WAN_I2V=1` (+ `OTR_ENABLE_WAN_TI2V=1` once built);
> Wan UNET + umt5 CLIP + VAE on disk):
> `python scripts\otr_coverage_sweep.py --acceptance --only wan`
> (`--only wan` matches the `sweep_<slot>_wan_i2v` / `_wan_ti2v` legs; drop `--only`
> for the full visual set. `--exclude` of a core Wan engine is REFUSED in acceptance.)

- **M1 -- the sweep is BLIND to silent fallback.** `otr_coverage_sweep.py` runs every
  leg with `expect_engine=""`, which `_otr_soak_capstone.py:464` treats as
  informational (no assert), so a leg that silently falls back to `still_kenburns`
  scores PASS (this is exactly CS-1). FIX (NOT per-leg `expect_engine=engine` -- that
  false-fails a slot that gets 0 beats at 30w): in acceptance mode assert ZERO runtime
  fallbacks across the whole trace -- fail any shot where `final_engine != attempts[0]`
  -- with an opt-out only for known-degrade experiment legs. (Verify the trace field is
  a stable requested-id, not an alias.)
- **M2 -- the sweep returns GREEN on EMPTY results.** `return 0 if passed ==
  len(results)` makes `0 == 0` pass when `--only`/`--exclude` filter everything out or
  `wan_ti2v` is unregistered. FIX: fail on empty results; for GATE-A, fail unless BOTH
  `wan_i2v` AND `wan_ti2v` are present in results with PASS.
- **M3 -- acceptance preflight (closes the R2 trap).** `availability()` is pure
  profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so a gated-off Wan leg enumerates
  "run", `assert_usable` fails it closed, it falls back, and (pre-M1) passes -- the same
  `gated_by_flag` mechanism that floored HuMo-1.7B (commit 5231d31). FIX: the acceptance
  run preflights `OTR_ENABLE_WAN_I2V=1` (+ future `OTR_ENABLE_WAN_TI2V=1`) and the model
  files, and FORBIDS `--exclude` of the core Wan engines.
- **M4 -- the V-3 VRAM gate fails OPEN.** `driver_peak = int(report.get("vram_peak_mb")
  or -1)` then fails only if `> ceiling`, so a missing/0/negative measurement (`-1`)
  PASSES -- the `<=14.5GB` invariant can read GREEN with no measurement. FIX: fail
  closed when `vram_peak_mb` is absent or `<= 0`.
- **M5 -- the Wan render-phase VRAM assert is skipped under `OTR_TEST_MODE`** (`if not
  os.environ.get("OTR_TEST_MODE"): ... assert_peak_within_ceiling`). Phase-2 acceptance
  MUST run with `OTR_TEST_MODE` UNSET; the harness preflight fails if it is set.
- **M6 -- `assert_usable` preflights only the ckpt.** The umt5 CLIP + the VAE are
  required graph loaders. FIX: verify UNET+CLIP+VAE present + matching the sha/license
  manifest before any forward (offline / no-runtime-fetch invariant).
- **M7 -- the Phase-2 clip contract is SELF-DECLARED, not asserted.** `_clip_from_raw`
  hardcodes `has_audio=False`/h264/yuv420p/bt709/fps25 in a dict; the soak only inspects
  the obs final's audio. FIX: ffprobe the emitted silent Wan mp4 (or a real-path test)
  to PROVE those fields before mux.
- **M8 -- `wan_ti2v` VAE fail-closed.** `eng_wan_i2v` defaults the VAE to
  `wan_2.1_vae.safetensors`; the 5B needs the Wan2.2 VAE. Give `wan_ti2v` its own VAE
  env; raise `EngineUnusable` if the resolved VAE basename is empty OR equals the 2.1
  basename. Do NOT inherit `_loader_names()` unchanged.
- **M9 -- CS-3 = sequential residency (see section 5).** Prove per-beat peak <= 14.5GB +
  the inter-beat reclaim drains the prior heavy engine (incl. the retained Wan unet
  patcher) before the next loads; that is the real risk, not co-residency. Unblocks
  Phase-2 scoping.

SHOULD-FIX: **S1** raise `CAPABILITIES["wan_i2v"].vram_estimate_mb` 14000 -> the measured
Phase-2 peak (or 14500); the 14499 smoke figure was WITHOUT `free_after_use`, which is
load-bearing -- document it as mandatory. **S2** add a concrete `wan_ti2v` CAPABILITIES
row (`medium` / ~8000 DRAFT -- the 5B VAE decode may push higher, verify on the 8GB
probe / `["wan2.2-ti2v-5b"]`). **S3** surface the single-expert (low-noise) MoE motion
risk on the eyeball gate -- Path B two-expert HIGH/LOW handoff is the mitigation if the
"real camera motion" bar fails. **S4** sweep leg isolation -- reclaim/restart between
legs that swap heavy engines (one resident server, no teardown -> residue corrupts the
next leg's peak; ties to CS-2 + the CLAUDE.md reset directive). **S5** fix the stale
`["wan2.1-i2v"]` label -> the real Wan2.2 I2V asset id. **S6** point item-4's writer-LLM
+ voice-variation leg-sets at their real harness (run_combo_matrix.py?) or mark TODO --
`otr_coverage_sweep.py` enumerates ONLY the visual-engine set today. **S7** stage the
init image under a shot/seed/uuid name (`otr_wan_init_WxH.png` is fixed -> same-dim
renders overwrite; low risk, driver is sequential). **S8** spell `scripts/otr_coverage_sweep.py`
+ the exact `--only` Wan substring + required env. **S9** Phase-2 post-reset verify
(PID/start-time changed, Sage NOT active, `OTR_TEST_MODE` unset, env visible) before
submitting. **S10** `_materialize_init_image`: require Pillow + fail loud (the no-Pillow
path leans on `WanImageToVideo` cover-resize -- N9 risk).

CUTS (panel consensus -- do NOT over-engineer): no broad VRAM-budget-aware scheduler to
close CS-3 (the reclaim assertion suffices; wait for a measured failure); do NOT subclass
all of `WanI2VEngine` for `wan_ti2v` (share only pure dims/aspect/materialize/canonicalize
helpers; keep loaders + node candidates + graph SEPARATE); keep the GATE-A sweep ADDITIVE,
not a visual x writer x voice cross-product. VERIFY-AT-BUILD: capture TI2V-5B's exact core
node class from `/object_info` before coding (the "5B latent node" is underspecified).

---

## 4B. WAN PHASE 1 -- DONE (pointer)

Phase 1 PROVEN: a real Wan b-roll clip (wan_i2v 14B fp8 in-process, ~14.5 GB; commits `2fbc2f3` +
`8eaf058`). Phase 2 is the ACTIVE next step (section 1); remaining Wan work = sections 4 + 4A. The
overnight-soak companion findings (R1 GPU-proven, R2 harness fix unexercised, R3 landed) live in git +
`scripts/FABLE_SOAK_REVIEW.md`; the not-done remainder (R2 verify) is in section 5.

---

## 5. OPEN TICKETS

- **SCHEMA-ADHERENCE (2026-06-25 -- LEVER-1 LOAD-BEARING SHIPPED; see the CURRENT STEP block at the top):**
  LEVER 1 tolerance (`pass04_plan.md` C0-C6, refined by the nested-fork + c4-scope roundtables) SHIPPED in 2
  green chunks `516644eb` (C0+C1+C2+C5+C6: `apply_field_aliases`/`__otr_field_aliases__` before-validator +
  `validate_tolerant_data` core; proven nested Opus `normalize_length` failure fixed) + `d4ca6cd4` (C3:
  JSON-syntax-only structural rung). C4 (schema-in-repair) DEFERRED -- proven failure already fixed, would test
  dead code; OPTIONAL `_build_schema_snippet`-shim recipe ready in c4-scope/, reopen on a real captured drift.
  LEVER 2 binary lane `docs/2026-06-25-schema-adherence/binary/pass01_plan.md` still GATED on **G1** (offline
  abstain-residual count -- the cheap first move; may DROP the lane) + **G2** (byte-identity of abstain).
  **G1 DONE -> Lever 2 (binary lane) DROPPED (genuine residual ~0; `binary/G1_RESULTS.md`); SCHEMA-ADHERENCE
  SPRINT COMPLETE.** NO workflow-JSON change.
- **LOOK-QA BUGS (NEW 2026-06-14 eve — operator look-QA pass; all in `BUG_LOG_2026-06.md`):**
  - **BUG-408 default MUSIC sounds non-musical (SA3).** **IMPLEMENTED 2026-06-14 (`3a4f71d`).** Path B:
    SA3-shaped prompt + real negative + per-cue `seconds_start` within a 30s `seconds_total` context (latent
    stays `dur` → length+determinism unchanged), env-overridable sampler knobs. Suite 4261/0. **OPERATOR-GATED:**
    restart Desktop, A/B listen (tune `OTR_SA3_CFG/STEPS/CONTEXT_S`), then RE-BASELINE the `test_audio_byte_identical`
    golden (intended music-bytes change). Plan: `docs/2026-06-14-sa3-music-improvement/roundtable/pass01_plan.md`.
  - **BUG-409 title card scrambles the WHOLE window** — **FIXED 2026-06-14 (`9e0b658`).** New
    `_title_reveal_progress` resolves the reveal in the first ~40% of the window then holds solid (env
    `OTR_TITLE_REVEAL_FRACTION`); close card stays bounded to the main video (no credits overlap). Suite 4259/0.
  - **BUG-410 closing ROLLING CREDITS** — **CLOSED 2026-06-14 (operator-verified on flux_still).** Credits
    scroll over the held last clip to the end again (silent after the theme). Detail in `BUG_LOG_2026-06.md`
    + `docs/2026-06-14-credits-tail-fix/`. (HuMo backdrop not yet eyeballed — low risk, engine-agnostic path.)
  - **BUG-411 flux BOOKEND / image lost its "lush" cinematic tint (NEXT — HANDOFF FOCUS).** The 6/5 image
    pipeline (`visual/batch_flux_render.py` + `flux_prompt_extractor`) was WHOLLY REWRITTEN into
    `_otr_image_engines/flux_gen1.py` + `otr_meta_brief_image_prompt.py` (pure insertions after `e4cb3ac`).
    Model/steps/cfg/sampler IDENTICAL (flux1-dev-fp8, 20, 1.0, euler/simple), but the rewrite DROPPED the look
    levers: **(1) FluxGuidance = 3.5** (flux_gen1 has NO FluxGuidance node — biggest factor), **(2) the
    cinematic style suffix** `"cinematic, 35mm film, anamorphic lens, volumetric lighting, heavy vignette,
    muted color grade, sharp focus"`, **(3) the radio broadcast-distress suffix** + retrofuturistic radio
    fallback (`35mm film grain ... dim amber and cyan rim lighting`), **(4) bookend seed 4242**, **(5) portrait
    style line**. 6/5 workflow widgets inspected + confirmed (no other hidden hardcodes). FIX = restore those in
    the new pipeline (FluxGuidance node @ ~3.5 + the suffixes + seed). Full forensic in `BUG_LOG_2026-06.md`
    BUG-411. CODER-READY (the next window's task).
  These are GATE-A look-QA items (operator-gated track), parallel to the engine forward order — NOT a
  reordering of section 3.

- **IMPROVED 3D INPUT -- BLOCKED on a PATH DECISION (operator look-QA 2026-06-14; GROUNDED this session).**
  The 3D rotating output looked like a "blobby plaster-of-paris" block. GROUNDING (checked logs + disk):
  the ONLY 3D system actually installed/active is **HunyuanWorld-Mirror / WorldMirror 2.0**
  (`custom_nodes/ComfyUI-HunyuanWorld-Mirror`, model `C:\ComfyUI-Models\WorldMirror-V2\HY-WorldMirror-2.0`)
  -- NOT Blender, NOT OTR's deferred character_3d/TripoSG. Recent episode ledgers used NO 3D engine; the
  server log only shows HWM loading (no episode rendered 3D). **WorldMirror is a MULTI-VIEW SCENE
  reconstructor** (image SEQUENCE -> point cloud / Gaussian splat): per its docs 1 frame = "depth/normals
  only"; good 3D needs **8-24 FEATURE-RICH frames, orbital/forward parallax, well-lit, 50-70% overlap**. A
  single flat/low-feature image -> the plaster blob. So the earlier "clean / object-free single image"
  idea is the OPPOSITE of what WorldMirror needs -- object-free helps only single-image-to-OBJECT-mesh
  tools (TripoSG / Hunyuan3D-2 / TRELLIS), which are NOT installed. **OPEN DECISION (operator, next window)
  -- the prompt strategy is opposite per path:** (A) WorldMirror scene/world -> improved input = GENERATE
  an orbit/multi-view sequence + rich-textured scene prompts (NOT a plain bg); or (B) single-image ->
  object mesh -> INSTALL TripoSG/Hunyuan3D-2 + clean isolated-subject prompts. Do NOT draft the improved
  3D prompts (and do NOT wire character_3d) until the path is picked. A roundtable can harden the chosen
  path's prompt set. (Note: the live roundtable launcher stalled this session -- the panel blocked with no
  output; budget a retry or a smaller panel.) Example obs finals to eyeball the EPISODE look (these do NOT
  contain 3D): `output\otr\obs\signal_lost_plunging_depths_20260614_185229_silent_procgen_blended_final.mp4`
  (a pre-fix render -- shows the closing FREEZE + the skinny flux_still portrait, both now fixed at HEAD).
- **HuMo full-frame TEST (operator 2026-06-14 -- future experiment, NOT now).** Operator wants to
  eventually SEE HuMo rendered full-frame (not the 480x832 portrait pillarbox). For now portrait stays
  HuMo's REQUIREMENT -- BUG-407 shipped "full frame everything EXCEPT HuMo". Future: a HuMo full-frame /
  16:9 smoke to evaluate whether the talking-head holds at a wider aspect before changing the default.
- **Look-QA the 5 overnight 120-word episodes (NEW 2026-06-14).** The default-lane soak ran 5/5 SUCCESS
  (LTX + humo_1.7B); the episode outputs (`...\output\otr\episodes` + obs finals) are NOT yet eyeballed.
  Check audio sync, burned captions, procgen scopes/credits, character look. This is the operator's
  "analyze the soak" item; verdicts in `scripts/_otr_120word_soak_summary.json`.
- **Wan WEBM EYEBALL -- DONE 2026-06-14 (operator + Claude live smoke).** RESULT: **Wan i2v 14B
  DRIFTS** -- holds the input still ~1 frame then re-interprets the scene into its own subject (a
  generic tube close-up). NOT fixable by easy input knobs: cfg3.5->2.0 + a locked-tripod prompt STILL
  drifts; cfg1.5 COLLAPSES into incoherent abstraction. **LTX (2B v0.9) HOLDS** the composition with
  subtle motion in all 3 modes tested (ksampler 30-step, distilled, AND 1216x704 hires -- hires
  answers the "low-res" note). => **RECOMMEND: Wan i2v 14B -> BACK-BURNER for the music/announcer
  OPENER role** (keep selectable; revisit only with Path B two-expert handoff, GO_FORWARD 4A S3); LTX
  stays the opener engine; **PROMOTE LTX-REGR (below) to the active thread.** Evidence:
  `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. AWAITS
  operator confirm on the re-prioritization.

- **Non-Wan soak = ENOUGH (operator call 2026-06-13).** The non-Wan permutation coverage sweep
  (`--strict-fallback --exclude wan/latentsync/triposg`) has run sufficiently; do NOT keep grinding it.
  The non-lip-sync FLOORS (`still_kenburns` / `still_parallax` / Ken-Burns / `station_card`) render fine
  and are acceptable for the 8GB tier, but they are NOT the target experience -- the operator wants real
  audio-driven lip-sync, not a still with motion. Focus the remaining runway on **getting the Wan lane
  bug-free** (section 1 + 4 + 4A). A new sweep, if ever needed, should add `--exclude-engine humo` (the
  exact-match flag added `ca10b63`: skips the 14B `humo` that TIMES OUT per CS-4, KEEPS `humo_1.7B`).
- **LTX-REGR — SUPERSEDED 2026-06-15 by the LTX 22B-GGUF splice** (`docs/2026-06-15-ltx-splice/SPLICE_PLAN.md`).
  LTX-REGR's recommended fix was to bake the **2B** v0_9 recipe into `eng_ltx_video.py`; the splice instead swaps
  `LtxVideoEngine` to the **22B GGUF** mini recipe (verified-working). **Do NOT do the 2B bake.** Original entry kept
  below for history only:
  **LTX-REGR (operator 2026-06-13; PROMOTED to active 2026-06-14 pending operator confirm)** -- LTX
  clips no longer animate like the **2026-05-30..06-05** era (motion lost / too static). `BUG-LOCAL-113b`
  (`8115c72`: ksampler 30-step euler cfg3.0 as the LTX default, distilled 8-step = the
  `OTR_LTX_SAMPLER=distilled` rollback) was the prior fix, but the operator STILL sees the regression.
  **2026-06-14 eyeball update:** the Wan-vs-LTX smoke proved LTX HOLDS the still composition cleanly
  (good) -- so the open question is narrowed to **MOTION AMOUNT** (5/30-6/5 read as more dynamic; the
  current ksampler/distilled holds are subtle). With Wan i2v back-burnered for openers, this is the
  recommended NEXT thread. Probe = an LTX **--strength / sampler-mode / step-count / cfg / frame-cap**
  sweep (otr_ltx_motion_smoke.py exposes all of them; --strength is the prime motion lever, 1.0=max
  freedom) against the 5/30 baseline + the 169 decode floor from look-QA round 5.
  **FORENSIC DONE 2026-06-14 (BUG-LOCAL-412, `BUG_LOG_2026-06.md`):** diffed the GOOD 5/09 `l001` + 5/28
  `b001` LTX bookends vs the current engine (ledgers + the DELETED `batch_ltx_render.py` @ `70d379b^` + the
  old workflow JSON widgets). The good recipe = **v0_9 / sampler `euler_cfg_pp` / 8 distilled steps / cfg
  1.0 / 832×480 / I2V strength 0.75 / `loop_via_reverse` boomerang / audio-length**; the cleanbreak
  `70d379b` DELETED that node and `eng_ltx_video.py` shipped **ksampler / `euler` / 30-step / cfg 3.0 /
  768×512-or-1472×832 / strength 1.0 / 169-cap / no boomerang** (the code comment itself admits `euler_cfg_pp`
  is the documented dynamic-motion sampler but the default was left on `euler`). The old WORKFLOW JSON baked
  in NOTHING but seed/method/cap — the recipe lived in code. **ENV-TESTABLE A/B FIRST (no code change):** at
  832×480 set `OTR_LTX_SAMPLER=distilled` + `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`,
  re-render a bookend, A/B vs `l001`/`b001`; if it matches, bake those defaults + the boomerang + audio-length
  back into `eng_ltx_video.py` (coder chunk; no JSON change implicated).
- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep. (Non-Wan -> deprioritized per the operator's "non-Wan soak = enough" call.)
- **CS-2** -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase attribution reads
  ~3 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase peak is a partial answer).
- **CS-3 (reframed 2026-06-13)** -- NOT a co-residency budget: wan_i2v (~14GB) +
  humo_1.7B (~7GB) cannot co-reside under 14.5GB by construction, so they must render
  SEQUENTIALLY. The real proof obligation = per-beat NVML peak <= 14.5GB AND the
  inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`, BUG-291) fully drains the
  prior heavy engine -- incl. the retained Wan unet patcher -- before the next beat
  loads. A mixed Wan+HuMo episode is the test. This UNBLOCKS Phase-2 scoping (no
  open "decision" needed). See section 4A M9.
- **CS-4-open** (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN 14B HuMo lane so it
  fits 14.5 GB. The default char tier is `humo_1.7B` (`955f134`); the 14B is opt-in.
- **R2 verify** -- confirm `humo_1.7B` renders native char beats at 70w with its enable flag ON (the
  soak floored it only via `gated_by_flag`); answered by the item-4 re-run.
- **README "what to expect per video model" (operator 2026-06-14).** Once the opener model bake-off
  settles (interactive render bench artifact `otr-render-bench` + `docs/2026-06-14-wan-ti2v/
  EYEBALL_FINDINGS.md`), add a user-facing "what to expect from each video engine" section to the
  README (newbie audience -- folds into S6/closing): Wan i2v 14B = drifts off the still (b-roll only,
  NOT openers); LTX = holds composition + subtle motion (opener default); TI2V-5B = 8GB tier, lower-res.
  Source the verdicts from the operator's bench ratings (export button).
- **Ship defaults (release)** -- proposed: announcer + character = `flux_still`, music = `visualizer`
  (selectable: station_card, still_parallax, abstract — `ltx_orbit` ripped 2026-06-15 in the LTX splice Phase 0). Keep HuMo/latentsync/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish** (minor) -- output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR`
  (fail LOUD on mismatch); run the OH-3 janitor sweep at server boot; widen the heartbeat cadence.
- **OH-4** -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator "go OH-4"
  (`docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`).
- **0-E Phase B** -- tickets E-1..E-7, gated on the sweep GO file; coder-window ready.
- **Operator gates** -- ComfyUI Desktop relaunch (look-QA), fresh-render acceptance, whiny-voice P0 matrix
  + reel, S-3D-0 green light, `v2.0-alpha-stable` tag decision. (latentsync demos REMOVED 2026-06-21.)

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
path gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9:
S-3D-0 spike -> T2b keystone GO/NO-GO (timeboxed ~1wk) -> T4 driver + LOOK gate -> W7 production wiring +
soak ("v1-usable") -> S3-S6 distribution. SHORTCUT FORK: S-3D-0 or keystone NO-GO -> `character_3d`
defers (HuMo-2D stays) -> collapses to ~2-3 sprints (0-E + closing). Done splits: "v1-usable" (one
engine, one real episode) vs "B-parity ship" (>=2 engines bind at SHIP).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard: `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Soak review (R1/R2/R3 detail + roundtable): `scripts/FABLE_SOAK_REVIEW.md`.
- Wan/sweep hardening (grounded QA + 3-model roundtable judgment, 2026-06-13):
  `docs/2026-06-13-goforward-wan-hardening/` (pass00 plan+QA, pass01/pass01b raw
  reviews, pass01_judgment.md).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` +
  `otr-sweep-monitor`; digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (forward item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug log (this repo): ACTIVE = `BUG_LOG_2026-06.md` (epoch BUG-LOCAL-400+, started 2026-06-14);
  ARCHIVE = `BUG_LOG.md` (BUG-LOCAL-001..~305, through 2026-06-12, reference only).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
(**LTX-AV audio-input lane MOVED OUT of PARKED 2026-06-17** -- operator revived it as the CURRENT STEP
(section 1): M1+M3 are shipped, the remaining work is recipe-align + M4 GPU smoke. It already uses the good
Q3_K_M GGUF unet; do NOT rebuild from scratch.)

(**STORY-ENGINE quality roundtable (2026-06-21) -- PARKED side campaign.** A 4-pass live roundtable converged a
sprint-ready plan for 8 content-only story-engine fixes (length tail / costly-choice binding / ending-aware outro
/ gender-pronouns / speech register / narration hygiene / arc-shape variety; F9 reorder + F10 anti-repeat list
deferred). Docs: `docs/2026-06-21-allnight-864-frontier/` -- `SPRINT_READY_PLAN.md` + `STORY_ENGINE_KICKOFF.md` +
`roundtable/pass0{1,2,3,4}_judgment.md`. All content-only inside the FIXED ledger, ZERO workflow-JSON edits
(verified vs the real consumers). NOT active -- the visual fixes (section 1) + the forward order win. Resume only
on an explicit operator green light.)
