# CLEANBREAK + COMBO-SOAK -- r1-HARDENED PLAN (Claude-synthesized, grounded)

Panel: Claude Code + Codex + Antigravity (r1). Every folded claim grounded against the real repo;
misreads + shim-suggestions discarded. Operator invariant: CLEAN BREAK, NO shims, no runtime
back-compat gates; workflow JSON changes ride the SAME commit as the code.

## PART A -- CLEAN-BREAK FALLBACK RIP-OUT (surgical keep-list, NOT a blanket delete)

REMOVE (dead after runtime fallbacks were disabled):
- `make_fallback_of`, `FLOOR_NAMES`, `UNIVERSAL_FLOOR`, `SYNTH_FALLBACKS`, `FamilyInputGap` (class +
  its `classify_failure` case) -- render_driver.py ~46-159,127.
- The `fallback_of` parameter threading in `run_episode` / `run_real_episode` / `render_single`.
- EVERY engine's `fallback_engine` attribute -> set to None / remove (GROUNDED targets:
  eng_humo.py:120,504; eng_ltx_video.py:281; eng_still_parallax.py:186; eng_mesh_stage.py:317;
  eng_triposr.py:120; eng_character_3d.py:258,327,398). visualizer + ltx_audio_in already declare
  `fallback_engine=None` -- that is the target state for all.
- The soak: `build_soak_fixture`, `run_gpu_soak`, `assert_soak_ok`, `_PROFILES`, `_CHAR3D`,
  `OOM_ENGINES`, `EXPECTED_OOM_TRAIL`, `OTR_VideoRenderBatch` mode="soak", and the exception classes
  `OomSignal` / `SoakError` / `RenderFloorError` (zero non-soak consumers).

KEEP (grounded -- these have MANY non-fallback callers; deleting them breaks production):
- `engine_family()` + its registry branch -- 10+ live sites (render_driver.py
  ~236,737,841,1126,1190,1422,1490,1819,2251: canvas selection / lipsync dispatch / per-beat request
  build / apply-engine-override). Only PRUNE the `ENGINE_FAMILY` dict (drop `soak_oom_3d` + any
  retired engine).
- `classify_failure()` -- used by the no-fallback `render_shot` (render_driver.py:1546) to name the
  `FailureKind` in `RenderError`. Remove ONLY the `FamilyInputGap` case.
- `RenderError` (the no-fallback production exception) stays.

A2 ENGINE RETIREMENT (ordered; still_motion is NOT pure scaffolding):
- `abstract` (redundant with `visualizer`) + `station_card` (broken black card) -- retire cleanly:
  unregister + `cheap_families.py` + capability rows (`registry.py` ~127-133) + node-87/director
  dropdown options + workflow-JSON option-list re-baseline + tests. VERIFY no role's default-engine
  list becomes empty after removal (`default_roles` on the retired classes).
- `still_motion` -- GROUNDED LOAD-BEARING: it is `UNIVERSAL_FLOOR` AND the declared `fallback_engine`
  of humo/ltx_video/still_parallax AND a registered selectable scene_broll engine (Ken-Burns motion
  still, distinct from still_pan/still_flat). Retire it ONLY AFTER the floor + fallback_engine
  removal above strips its scaffolding role, THEN migrate its remaining selectable refs + tests to
  still_pan/still_flat. **OPERATOR FLAG:** retiring it removes the Ken-Burns motion-still option
  (operator stated intent: "twin of still_pan"); confirm before unregistering vs keeping it
  selectable-but-no-longer-a-floor.

A3 TEST MIGRATION -- SAME COMMIT (clean break, NO stub). GROUNDED breakage:
- Calls into `make_fallback_of` / `run_episode(..., fallback_of=)` / asserts `SYNTH_FALLBACKS`:
  test_cs3_inter_beat_reclaim.py:55, test_ltx_av_driver_wiring.py:25-28, test_video_mesh_stage.py:323,
  test_video_render_driver.py:24, test_video_render_driver_additive.py:414,
  test_video_still_parallax.py:182.
- DELETE fallback-chain tests: test_video_fallback_chain_additive.py, test_video_retry_taxonomy.py,
  test_video_retry_taxonomy_additive.py.
- MIGRATE still_motion asserts -> still_pan/still_flat: test_video_cheap_render.py, test_video_humo.py,
  test_ltx_open_health.py (+ the soak tests, removed with the soak).

A4 `allow_auto_fallback` (E2): dead config on OTR_VideoDirector (otr_video_director.py:216,342) +
   `VideoPolicy.allow_auto_fallback` (schemas.py:128). Under CLEAN BREAK -> clean-DELETE + a
   coordinated workflow-JSON re-baseline (the hard rule permits the JSON change in the same commit),
   with `otr_api` widget-vector + `OTR_WorkflowValidator` validation. (r3 confirms the positional
   `widgets_values` shift is absorbed by the rebaseline -- BUG-LOCAL-097 guard.)

## PART B -- IMAGE-GEN-START COMBO SOAK (replace the gutted soak)

B1 BAKE-ONCE FIXTURE (a NEW combo fixture, NOT raw S-F reuse). Capture, from a CLEAN reference run
   (the live reference episode writes `state/node_episode_input.json` -- bootstrap prerequisite):
   the PRE-IMAGE ledger (post-CastLock, audio-frozen) + `image_policy_json` + `image_prompts_json` +
   `script_json` + `master_audio_path` + `episode_id` + the image/audio gate tokens. Bake-manifest
   INVARIANTS (assert at bake): cast rows + voice refs + a cast-lock revision present; per-line audio
   or master-audio fallback; stable master PCM hash. CLEAR `ledger["images"]` + the dispatcher
   cache/index + any pre-resolved still paths so image-gen RE-MINTS per leg (the cache-reuse path,
   otr_image_gen_dispatcher.py ~478-517, must not short-circuit the engine under test).

B2 PRUNE BOUNDARY (one generator, two boundaries). Extend `scripts/otr_visual_smoke.py` with a
   `--start-boundary {video|image_gen}` param (keep S-F's node-92-only `video` boundary intact). The
   `image_gen` boundary replay graph = the full visual tail:
   OTR_ImageGenDispatcher -> OTR_VideoRenderBatch(92) -> OTR_SilentComposite -> OTR_MasterAudioMux,
   with the dispatcher's `script_json` / `image_policy_json` / `image_prompts_json` BAKED as literals
   (their upstream producers are writer-phase nodes 90/88/89 -- bake to avoid pulling the writer in),
   `master_audio_path` baked, and the IMAGE + VIDEO engine selections LEFT LIVE. **r3 must capture the
   exact node ids + each node's forceInput sockets from the live `/object_info` + the workflow JSON
   before coding** (types confirmed present: ImageDirector / ImageGenDispatcher / VideoRenderBatch /
   SilentComposite / MasterAudioMux). Verify no dispatcher forceInput socket transitively pulls a
   writer/audio node into the executed graph (mirror S-F's executed-node-set assertion).

B3 MATRIX -- ADDITIVE ~15 (NOT a cross-product; pinned). Reuse `scripts/_otr_cov_runner.py`'s fill
   model: (a) 5 IMAGE-engine legs -- each of {flux_gen1, flux2_klein, z_image_turbo, qwen_image,
   lumina_image} filled across the 3 image slots with a still carrier; (b) ~10 VIDEO-engine legs --
   each renderable video engine filled across its valid slots with a neutral image. ~15 additive legs
   total (a full 5x10 cross-product = 50, explicitly CUT). Resumable + merged into
   `_otr_coverage_matrix.json` + the `otr-coverage-soak` dashboard (plumbing detail -> r2).

B4 ACCEPTANCE -- a STRUCTURED matrix verdict (replaces the old A-ship gate). Per leg:
   - executed-node set == the pruned boundary set (writer + audio nodes ABSENT);
   - IMAGE leg: node 91 executed, `made > 0`, `reused == 0` (unless an explicit cache-reuse leg),
     `meta.image_engines.by_role` names the selected image engine, still files exist on disk;
   - VIDEO leg: node-92 `engine_histogram` + the E5 recipe receipt name the selected video engine;
   - NO silent floor substitution (fallbacks are gone -> a miss is a LOUD named hard-fail, not a still);
   - master PCM hash EQUAL before/after (audio byte-identical); render-window NVML <= 14.5 GB;
   - deterministic two-pass replay for >= 1 representative leg.
   Add ONE explicit cache-reuse leg AFTER the fresh-render legs (so cache behavior is tested, never
   masking the engine test).

## INVARIANTS (unchanged)
Workflow JSON = source of truth (node/widget/option changes go IN it, same commit, re-validate via
OTR_WorkflowValidator + JSON round-trip + widget audit); single resident heavy <= 14.5 GB; 100%
local; seed determinism; master audio byte-identical (`test_audio_byte_identical` GREEN); UTF-8 no
BOM; SFW; suite + Bug Bible + B7 green AND push per green chunk; clean break = no shims.

## REJECTED (this round)
- A no-op `make_fallback_of` stub / `SYNTH_FALLBACKS={}` shim (Antigravity) -- VIOLATES clean-break;
  migrate/delete the tests in-commit instead.
- "Delete `engine_family` / `classify_failure`" -- both have many non-fallback callers (grounded).
- A 5x10 image x video cross-product -- out of the operator's ~15 scope.

## VERIFY-AT-BUILD (carried to r2/r3)
Exact node ids + forceInput wiring for the image_gen boundary; whether the dispatcher can run without
re-pulling the writer; the node-92 mode-combo + allow_auto_fallback widget-vector rebaseline; the
exact renderable-video-engine list for the ~10 video legs; the cache/stills-clear mechanism.
