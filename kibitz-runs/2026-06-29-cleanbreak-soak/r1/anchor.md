# Claude anchor review -- r1 (arc / coherence) -- CLEANBREAK_SOAK_REENGINEER_PLAN

VERDICT: Arc is sound; 3 must-fix structural gaps + 2 should-fix. Grounded vs the real repo.

## MUST-FIX
M1. PRUNE BOUNDARY conflation (CONFIRMED). S-F's `build_pruned_prompt`
    (`scripts/otr_visual_smoke.py`) prunes to node-92 ONLY and bakes its 3 forceInput sockets
    (patched_ledger_json / master_audio_path / image_done); the stills are already committed in the
    baked ledger's `images.images`. "Start at image gen" is a DIFFERENT, upstream boundary: it must
    re-run `OTR_ImageGenDispatcher` per image engine. The arc must commit to: the baked fixture is the
    FROZEN story+cast+audio+beat-timing ledger WITH the committed stills CLEARED, so image-gen
    re-mints per engine; the prune set becomes {image-director(s) + image-gen-dispatcher + node-92},
    baking the writer/audio-derived inputs and leaving the image+video ENGINE selection live. (S-F's
    node-92-only prune stays the fast path for a pure video-engine swap.)

M2. A-SHIP GATE SEMANTICS (CONFIRMED). The old soak's value is the "not all-procgen / the selected
    heavy engine actually ran" assertion via `engine_histogram` + the degradation trail. Gutting it
    drops that gate. The re-engineered soak MUST replace it with a concrete per-combo acceptance:
    "the selected image+video engine appears in the histogram, OR the leg hard-fails LOUD with a
    NAMED reason" -- nothing silently floored. `build_clip_manifest` already emits `engine_histogram`
    (the signal); the new gate reads it per leg (mirror `_otr_cov_runner`'s silent-fallback detector,
    now repurposed as a silent-FLOOR detector since fallbacks are gone).

M3. CLEAN-BREAK SCOPE: do NOT rip `engine_family` (CONFIRMED). `render_driver.engine_family` (~177)
    + `ENGINE_FAMILY` feed `build_clip_manifest`'s `family` field + the recipe-stamp. Only the
    FALLBACK CHAIN (`make_fallback_of`/`FLOOR_NAMES`/`UNIVERSAL_FLOOR`/`SYNTH_FALLBACKS`/`FamilyInputGap`)
    + the soak constants (`_PROFILES`/`_CHAR3D`/`OOM_ENGINES`/`EXPECTED_OOM_TRAIL`/`build_soak_fixture`)
    + `OTR_VideoRenderBatch` mode="soak"/`run_gpu_soak`/`assert_soak_ok` are removed. Keep a pure
    engine->family lookup. The arc must state this boundary so the rip-out is surgical, not a blanket
    delete of everything that says "fallback" or "soak".

## SHOULD-FIX
S1. PIN THE MATRIX (CONFIRMED). "~15" = the ADDITIVE shape (5 image engines x 3 video lanes), NOT the
    image x video cross-product (~25+). Matches the existing `_otr_cov_runner` fill model (one engine
    filled across its valid slots). Confirm the exact image-engine list (flux_gen1 / flux2_klein /
    z_image_turbo / qwen_image / lumina_image) + the 3 video lanes per the operator.
S2. REUSE, DON'T REBUILD (CONFIRMED). Extend `scripts/_otr_cov_runner.py` (already merges + resumes
    into `_otr_coverage_matrix.json` + drives the `otr-coverage-soak` dashboard); swap its per-leg
    full-episode `soak.run_leg` for the bake-and-replay prune. Don't author a parallel harness.

## UNVERIFIABLE (verify-at-build)
U1. Whether `OTR_ImageGenDispatcher` can be driven from a baked ledger with its stills cleared without
    re-pulling the writer node -- needs the dispatcher's INPUT graph inspected (its forceInput sockets
    + which upstream nodes feed it). This is the r3 wiring crux.
U2. Whether gutting mode="soak" breaks a non-soak consumer (watchdog, CI gate, keystone). Grep at build.
