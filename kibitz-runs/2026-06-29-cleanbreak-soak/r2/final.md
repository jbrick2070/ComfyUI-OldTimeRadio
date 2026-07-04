# CLEANBREAK + COMBO-SOAK -- r2-HARDENED CODING PLAN (Claude-synthesized, grounded)

Panel r2 (Claude Code + Codex + Antigravity), all claims grounded vs the real repo. Clean break, NO
shims. Build as ordered chunks; suite + Bug Bible + B7 green AND push per chunk.

## CORRECTED KEEP/REMOVE (grounded -- supersedes r1)
KEEP (production, NOT scaffolding):
- `FamilyInputGap` + `_assert_family_inputs_satisfiable` -- GROUNDED: `_render_one` calls it
  UNCONDITIONALLY (render_driver.py:1448) on EVERY render; the no-fallback `render_shot` (1544) relies
  on `classify_failure` mapping it -> DEPENDENCY_MISSING (170). KEEP the class + that classify case;
  only REWORD the docstring to drop the "fallback candidate / chain" framing (it is now a hard
  input-contract guard that fails LOUD).
- `engine_family()` (10+ non-fallback callers) -- only PRUNE the `ENGINE_FAMILY` dict (drop
  `soak_oom_3d`); `classify_failure()` + `RenderError` stay.
REMOVE:
- `make_fallback_of`, `FLOOR_NAMES`, `UNIVERSAL_FLOOR`, `SYNTH_FALLBACKS`; the `fallback_of` param
  threading (run_episode/run_real_episode/render_single) + the dead `prune_orphaned_groups`-on-fallback
  path in run_episode.
- `OomSignal` + the `force_oom` plumbing in `_render_one` (1440-1447) + its `classify_failure` OOM
  case -- GROUNDED soak-only (raised only when `force_oom=True`).
- The whole soak: `build_soak_fixture`/`run_gpu_soak`/`assert_soak_ok`/`_PROFILES`/`_CHAR3D`/
  `OOM_ENGINES`/`EXPECTED_OOM_TRAIL`/`SoakError`/`RenderFloorError` + `OTR_VideoRenderBatch` mode="soak".
- `nodes/_otr_shared/fallback.py` (whole module: resolve_fallback_chain/chain_terminates_at_floor/
  FallbackChainError/FallbackCycleError) + its DEAD import at render_driver.py:35 (imported, never
  called) + `scripts/otr_video_soak.py` + `scripts/otr_video_gpu_smoke.py:demonstrate_humo_fallback`.
- Every engine `fallback_engine` attr -> remove (eng_humo:120,504; eng_ltx_video:281;
  eng_still_parallax:186; eng_mesh_stage:317; eng_triposr:120; eng_character_3d:258,327,398).
- `retry_taxonomy.py` fallback bits (`escalate_to_fallback`, `build_fallback_decision`,
  `append_runtime_fallback_decision`) -- audit + remove if dead after the rip-out.

## CHUNK ORDER (each green+pushed)
C0. PRODUCTION-CALLER SWEEP (do first, no code): grep PRODUCTION (non-test) for the REMOVE symbols to
    confirm no hidden consumer beyond those grounded above.

C1. Fallback rip-out (render_driver + engines + the dead module + tests, one commit).
    TEST MIGRATION/DELETE (grounded, same commit): delete the fallback-chain/soak tests --
    test_video_fallback_chain_additive.py, test_video_retry_taxonomy.py(+_additive),
    test_video_soak_fixture.py, and the named fallback tests in test_video_render_driver.py:23-45,
    test_video_survival_guide_vectors.py:63-82, test_video_character_3d.py:354-369,
    test_video_gpu_smoke.py:68-78, test_video_render_driver_additive.py:414,486; migrate the
    `resolve_fallback_chain` importers (test_video_humo.py:23/205-215, test_video_still_parallax.py:182,
    test_video_mesh_stage.py:323, test_video_render_driver.py:20). KEEP the survival-guide
    `test_no_ghost_fallback_engine_references` GREEN by removing ALL `fallback_engine` attrs (none
    remain -> no ghost). GROUNDED: cheap_families.py declares NO `fallback_engine` (only eng_* do).

C2. Gut the soak: remove the soak symbols + `OomSignal`/`force_oom`; `OTR_VideoRenderBatch` mode combo
    `["soak","single","episode"]` -> `["single","episode"]` default "episode" + node-92 JSON rebaseline
    (saved value already "episode"; re-validate the widget vector via otr_api + OTR_WorkflowValidator) +
    its tests. Delete `scripts/otr_video_soak.py`.

C3. Engine retirement -- REASSIGN DEFAULTS FIRST (grounded gap): `abstract` is the
    `background_abstract` default + `still_motion` the `scene_broll` default + `station_card` the
    `announcer_visual` cheap default (cheap_families.py:170,179,189). Set
    `StillPanFamily.default_roles = ("scene_broll","background_abstract")` and reassign
    `announcer_visual`'s cheap default, THEN unregister abstract/station_card/still_motion + registry
    capability rows + the director dropdown option-set + tests. UPDATE `config/profiles/8gb_lite.json`
    + `config/profiles/cpu_floor.json` (they pin `announcer_visual: "station_card"` -> still_pan/flat)
    so `cross_validate_profile` passes. VERIFY no role default-less.

C4. `allow_auto_fallback` clean-delete (one commit): OTR_VideoDirector (216,282-286,342) +
    `VideoPolicy` (schemas.py:128) + `scripts/run_otr_30word_smoke.py:193-197` +
    test_route_a_14b_promotion.py:127 + test_still_aspect_and_labels.py:168 +
    test_video_platform_aseam.py:248,305,329 + node-87 `widgets_values` + `otr_api` widget-slot
    expectation -- coordinated, with the JSON widget-vector rebaseline.

C5. COMBO SOAK (offline parts now; live run in the GPU batch). New TRACKED script (`_otr_cov_runner.py`
    is UNTRACKED -- commit a tracked combo runner). Pieces:
    - DISPATCHER-INPUT CAPTURE: `OTR_ImageGenDispatcher.dispatch` writes
      `state/node_image_input.json` = {script_json, image_policy_json, image_prompts_json, gate_in,
      episode_id} (the node-92 capture lacks these). Best-effort, default-on forensic (twin of the S-F
      node-92 capture).
    - BAKE: merge the node-92 capture (ledger + master_audio_path) with the dispatcher capture; assert
      cast rows + voice refs + cast-lock revision + a STABLE master PCM hash ONCE at bake (per-leg
      re-hash is redundant with `test_audio_byte_identical` -- these legs never touch audio). CLEAR
      `ledger["images"]` + the dispatcher cache so image-gen re-mints per leg (r3 nails the exact cache
      key/storage).
    - REPLAY (image_gen boundary): a NEW multi-node subgraph builder (the S-F single-node prune does
      NOT generalize) -- a separate function/script. The real tail is multi-node (92 -> 84 ->
      86 -> 93 -> 85 with deps on 12/94/7 per the saved links); r3 captures the EXACT ids + each node's
      forceInput sockets from `/object_info` + the JSON, bakes the externals (base_video_path, audio
      gates) as literals, and defines the executed-node assertion. Verify node-91 ran via its
      observable outputs (image_done / stills_manifest / `made>0` / `reused==0` /
      `meta.image_engines.by_role`), NOT `outputs.keys()` (the dispatcher is not an OUTPUT_NODE).
    - MATRIX: additive ~15 = 5 image-engine legs (fill image slots; still carrier) + ~10 video-engine
      legs (fill valid slots; neutral image), NOT a 5x10 cross-product. + 1 explicit cache-reuse leg.
    - ACCEPTANCE per leg: executed set == the boundary set (writer+audio ABSENT); image leg proves
      node-91 ran + engine named + stills on disk; video leg proves the histogram + recipe receipt name
      the engine; NO silent floor; NVML <= 14.5 GB; deterministic 2-pass (same engine_histogram +
      clip metadata) on >=1 leg. Dashboard/merge plumbing AFTER the verdict works (deferred).

## OPERATOR DECISION GATE (resolve before C3 build)
`still_motion`: retire entirely (operator's stated "twin of still_pan"; still_pan absorbs scene_broll) --
ASSUMED YES. If instead it stays a selectable Ken-Burns option, keep it registered but strip only its
floor role. Plan proceeds on RETIRE; flag for a one-word confirm.

## INVARIANTS / REJECTED / VERIFY-AT-BUILD
Invariants unchanged (workflow JSON source of truth, <=14.5GB, audio byte-identical, no shims, push per
chunk). REJECTED: no-op fallback stub; deleting FamilyInputGap/engine_family/classify_failure; 5x10
cross-product; per-leg audio re-hash. VERIFY-AT-BUILD (-> r3): exact replay node-id graph + forceInput
sockets; dispatcher cache key/storage + the clear mechanism; the exact ~10 renderable-video-engine list.
