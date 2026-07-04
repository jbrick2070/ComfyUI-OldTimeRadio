# CLEAN-BREAK FALLBACK RIP-OUT + IMAGE-GEN-START COMBO SOAK -- PLAN (for kibitz)

> Operator directive 2026-06-29: "i need clean break from rip out -- i expect you do a
> re-engineer of the soak so it starts at image gen and story is baked in ledger, baked cast
> in, so we can soak the 15 or so video/image combos." Clean break, NO shims. Kibitz-harden
> this FIRST (r1 arc -> r2 coding -> r3 wiring -> r4 convergence), then implement.

Context: S-E E5 (ledger recipe-stamp) shipped. S-F shipped a visual smoke fixture
(`scripts/otr_visual_smoke.py`) that bakes ONE reference episode (ledger + master audio + every
referenced asset) into a bundle and submits a PRUNED ComfyUI API prompt of ONLY node-92
(`OTR_VideoRenderBatch`, mode=episode) so the writer + audio never re-run. This plan (a) does the
clean-break removal of the production fallback scaffolding and (b) GENERALIZES the S-F fixture into
the coverage soak: start at IMAGE GEN, story/cast/audio baked into the ledger, soak the ~15
video x image engine combos.

## PART A -- CLEAN-BREAK FALLBACK RIP-OUT (E1 + E3, no shims)

Runtime fallback is ALREADY disabled: `render_shot` raises LOUD (render_driver.py ~1526; a selected
engine renders or the episode fails). What remains is DEAD scaffolding that must be removed cleanly:

A1. PRODUCTION fallback machinery in `nodes/_otr_video_engines/render_driver.py`:
    - `make_fallback_of` (~142), `FLOOR_NAMES` (~46), `UNIVERSAL_FLOOR` (~51), `SYNTH_FALLBACKS`
      (~58), the fallback half of `ENGINE_FAMILY` (~65; KEEP a pure engine->family lookup if still
      used by `engine_family`), `FamilyInputGap` (~127).
    - The `fallback_of` threading through `run_real_episode` (~1692/1735), `render_single` (~2236),
      `run_episode` (~1561). Callers stop passing/àaccepting a fallback resolver.
    - `eng_character_3d.py` fallback refs (~55,257,326).
    - The fallback-specific tests (`test_video_character_3d.py:363-369`,
      `test_video_render_driver_additive.py:77-82`, plus any `make_fallback_of` / `SYNTH_FALLBACKS`
      coverage). Migrate/retire in the SAME commit.

A2. ENGINE RETIREMENT (E3): `still_motion`, `station_card`, `abstract` -- unregister + remove from
    `cheap_families.py` (~165-190), the capability rows (`registry.py` ~127-133), node-87 / director
    dropdown options, soak fixtures, and tests. (Saved node-87 values are
    visualizer/flux_gen1/humo_14B_169 -- none retired -- so no widget-value rewrite; but the workflow
    JSON dropdown OPTION lists + any validator that pins the option set must be re-baselined.)

A3. OPEN: confirm what STILL legitimately consumes `engine_family` / `ENGINE_FAMILY` /
    `classify_failure` after the rip-out (the recipe-stamp + histogram read engine_id, not family).
    Keep ONLY what a non-fallback path needs; delete the rest. NO back-compat gate, NO shim.

## PART B -- RE-ENGINEER THE SOAK (gut the forced-OOM demo; bake-and-replay combo soak)

The OLD A-S7.5 soak (`build_soak_fixture` / `run_gpu_soak` / `assert_soak_ok` / `EXPECTED_OOM_TRAIL` /
`_PROFILES` / `_CHAR3D` / `OOM_ENGINES`, render_driver.py ~83-204,2056-2230, plus `OTR_VideoRenderBatch`
mode="soak") exists ONLY to demonstrate the forced-OOM fallback degradation trail -- behavior that no
longer exists. GUT it (clean break). REPLACE with a bake-and-replay COMBO soak:

B1. BAKE ONCE: a reference fixture = the story + cast + per-beat audio + beat timing baked into the
    ledger (reuse the S-F bundle: `otr_visual_smoke.bake_bundle` + the node-92 `node_episode_input.json`
    capture). Cast baked in (CastLock frozen), master audio byte-identical, ONE story for every combo
    (apples-to-apples).

B2. START AT IMAGE GEN: each combo leg re-runs ONLY image-gen -> video -> composite -> mux. The PRUNE
    BOUNDARY moves UPSTREAM of S-F's video-only prune: include the image-gen dispatcher node
    (`OTR_ImageGenDispatcher`) + node-92, BAKE the writer + audio inputs, and CLEAR/re-mint the stills
    per image engine so the image engine is genuinely exercised. (S-F's video-only prune stays the
    fast path for a pure video-engine swap.)

B3. SOAK THE ~15 COMBOS: image engines {flux_gen1, flux2_klein, z_image_turbo, qwen_image,
    lumina_image} x the video lanes, swapping ONLY the image + video engine per leg, re-rendering the
    cheap visual tail (minutes, not the ~28 min full episode). Resumable + merged matrix (reuse the
    `_otr_cov_runner` merge/resume + the `otr-coverage-soak` dashboard). ACCEPTANCE per leg: the
    selected engines RENDER or hard-fail LOUD with a NAMED reason; nothing silently dropped (no
    fallbacks to assert anymore); audio byte-identical; the recipe-stamp records delivered engine +
    recipe.

B4. The old `mode="soak"` on `OTR_VideoRenderBatch` + `run_gpu_soak` are removed or repointed; the
    A-ship gate becomes "every registered combo renders or fails LOUD", not "the OOM trail degrades".

## OPEN QUESTIONS FOR THE PANEL (ground against the real repo)
1. Exact prune boundary for the image-gen-start combo soak: which node ids feed `OTR_ImageGenDispatcher`,
   and which of its inputs must be baked vs left live so image-gen re-mints per engine without re-running
   the writer/audio? (S-F baked only node-92's 3 forceInput sockets.)
2. How to CLEAR baked stills per leg so the image engine actually re-generates (vs the ledger's committed
   `images.images` short-circuiting it)?
3. Does gutting `mode="soak"` / `run_gpu_soak` break any non-soak consumer (the keystone assertions, the
   watchdog, the CI gate)? What replaces the A-ship "is-not-all-procgen" assertion?
4. After A1, what remains the canonical `engine_family` source, and is `classify_failure` still needed
   (no fallback to classify toward)?
5. The workflow-JSON re-baseline for the retired-engine dropdown options (E3): which validator/test pins
   the option set, and does removing options shift any positional `widgets_values`?
6. Combo count + matrix shape: is it image x video cross-product (~25) or the additive ~15 the operator
   means? Confirm the exact lane list.

## INVARIANTS (unchanged)
Workflow JSON is the source of truth (node/widget changes go IN it, same commit, re-validate); single
resident heavy <= 14.5 GB; 100% local; seed determinism; master audio byte-identical
(`test_audio_byte_identical` GREEN); UTF-8 no BOM; SFW; suite + Bug Bible + B7 green AND push per green
chunk to v2.0-alpha; clean break = no shims, no runtime back-compat gates.
