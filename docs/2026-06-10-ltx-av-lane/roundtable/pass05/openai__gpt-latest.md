<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Pass05 is not build-ready because the plan still says “tests pass05 enumerates” instead of pinning exact CPU/GPU tests, existing-test edits, and byte-identical gating.

MUST-FIX BEFORE BUILD:
1. [M1/M2/M3 tests] Add the definitive new CPU pytest files/cases; do not leave this as “pass05 enumerates.”
   Concrete list:
   - `tests/test_av_dims.py` — mirror pure-stdlib style in `test_video_fallback_chain_additive.py`.
     - `test_next_8n1_boundaries`: `T=25 -> 25`, `T=26 -> 33`.
     - `test_assert_ltx_dims_accepts_valid_canvas`: `1472x832`, `frames % 8 == 1`.
     - `test_assert_ltx_dims_rejects_bad_width_with_nearest_hints`: `1450x832` raises with nearest valid width hints.
     - `test_assert_ltx_dims_rejects_bad_height_with_nearest_hints`.
     - `test_render_frames_snap_up_and_cap`: snap-up plus cap behavior, including cap at configured `LTX_AV_MAX_FRAMES`.
     - `test_dims_fail_before_gpu_lease`: no AS-3 lease taken on bad dims. Mirror `test_humo_prepare_releases_lease_on_load_failure` lease assertions.
   - `tests/test_video_ltx_av.py` — mirror `test_video_humo.py`.
     - `test_ltx_av_registered_and_dark`: both engines registered, `default_roles == ()`, flag names, roles, family, required_inputs, isolation, fallback engines, dropdown presence via `vreg.all_engine_names()`.
     - `test_ltx_av_required_inputs_match_family_schema`: talk matches `audio_driven_face`; music matches new `audio_conditioned_video`.
     - `test_ltx_av_role_fit_audio_inputs`: `ltx_av_music` fits `music_visual` with `audio_ref`; `ltx_av_talk` fits `announcer_visual` and `character_video`; wrong roles rejected. Mirror `test_humo_role_fit_audio_driven_face`.
     - `test_ltx_av_assert_usable_flag_first`: flag unset raises `GATED_BY_FLAG` before Sage/node/weight/dims checks.
     - `test_ltx_av_assert_usable_sage_gate`: flag set + mocked `sageattention` resident raises the documented BUG-070 classified failure before node/weight checks.
     - `test_ltx_av_node_availability_gate_missing_class`: mocked `NODE_CLASS_MAPPINGS` missing required class raises classified failure; no GPU.
     - `test_ltx_av_weights_missing_after_nodes`: nodes present but fake weight path missing raises `MISSING_MODEL`.
     - `test_ltx_av_request_template_dims_checked`: valid/invalid `request_template.canvas` reaches `av_dims`; invalid raises before lease.
     - `test_ltx_av_ref_extraction`: tolerant `audio_ref` string/dict/None and `asset_refs["init_image"]`; mirror `test_humo_ref_extraction_handles_str_dict_and_none`.
     - `test_ltx_av_build_render_request_deterministic`: same request builds identical graph/request; seed changes alter only seed-dependent field. Mirror `test_humo_build_render_request_deterministic`.
     - `test_ltx_av_canonicalize_silent_identity`: returns `has_audio=False`, `yuv420p`, `bt709`, `fps=25`, integer `frame_count`, `engine_id`, `family`. Mirror `test_humo_canonicalize_silent_bt709`.
     - `test_ltx_av_fallback_chains_converge`: talk chain exactly `["ltx_av_talk","humo","humo_1.7B","latentsync","still_kenburns"]`; music chain exactly `["ltx_av_music","ltx_video","still_kenburns"]`; mirror `test_humo_fallback_chain_converges_on_radio_floor`.
     - `test_ltx_av_cold_import_no_heavy_libs`: import `nodes._otr_video_engines.eng_ltx_av` without `torch`, `transformers`, `diffusers`. Mirror `test_cold_import_humo_no_heavy_libs`.
     - `test_ltx_av_source_is_ascii_no_em_dash`: mirror `test_humo_source_is_ascii_no_em_dash`.
   - `tests/test_video_ltx_av_schema.py` — mirror `schemas.py` model round-trip style.
     - `test_audio_conditioned_video_family_added`: `audio_conditioned_video` in `sc.FAMILIES` and `sc.FAMILY_REQUIRED_INPUTS`.
     - `test_audio_conditioned_video_requires_text_prompt_and_audio_ref`: missing either fails.
     - `test_audio_conditioned_video_round_trip`: `VideoRequest.model_dump()` / `model_validate()` preserves family and audio ref.
     - `test_family_required_inputs_guard_still_in_sync`: preserve `assert set(FAMILIES) == set(FAMILY_REQUIRED_INPUTS)`.
   - `tests/test_video_ltx_av_driver.py` — mirror existing render-driver tests; VERIFY-AT-BUILD exact helpers.
     - `test_dark_lane_golden_existing_engine_requests_bit_identical`: with LTX-AV registered but flag off, existing-engine render requests/hash fixtures are byte-identical.
     - `test_flag_off_render_time_degrades_not_abort`: force/select `ltx_av_*`, flag unset, render walks fallback chain and completes.
     - `test_degradation_trail_retains_origin_and_hops`: trail includes `ltx_av_*` origin and every hop.
     - `test_force_map_role_guard_ignores_incompatible`: incompatible `(role, engine)` ignored with LOUD warning.
     - `test_announcer_portrait_alias_only_for_ltx_av_talk`: empty `char_id` + `announcer_visual` resolves shipped non-cast portrait only for `ltx_av_talk`; missing portrait classifies pre-render failure.
     - `test_render_one_passes_request_template_to_assert_usable`: fake adapter records `request_template is request`.
     - `test_render_one_legacy_assert_usable_typeerror_guard`: fake legacy adapter without kwarg does not crash with `TypeError`.
     - `test_synthetic_timing_audio_slice_only_ltx_av_music`: line with no `start_s/dur_s` gets shot synthetic slice only when `engine_id == "ltx_av_music"`.
   - `tests/test_video_ltx_av_io.py` — mirror `test_video_humo.py` canonicalize style; use existing ffmpeg/ffprobe helper pattern if present, otherwise VERIFY-AT-BUILD.
     - `test_fake_av_mp4_strip_removes_audio`: fake input with audio stream is stripped with `-map 0:v:0 -an`.
     - `test_encoded_clip_ffprobe_zero_audio_streams`: assert zero audio streams.
     - `test_pad_tail_marker_emitted_when_padding_gt_2s`: marker text exactly includes `[ltx_av] pad-tail rendered=<n> target=<T>`.
     - `test_canonical_clip_identity_stamps_manifest`: `CanonicalClip.engine_id/family` and manifest per-clip identity column agree; manifest writer path VERIFY-AT-BUILD.
   - `tests/test_video_ltx_av_ast.py` — mirror `test_brief_prompt_finishing.py` AST style.
     - `test_eng_ltx_av_imports_no_brief_helpers`: no brief/prompt-composer imports in adapter.
     - `test_b7_forbidden_import_sweep_includes_eng_ltx_av`: if touching existing B7 sweep, AST loop variable must be named `imp` per repo gotcha.
   - `tests/test_ltx_av_m0_artifact.py` — only after M0 artifact is committed.
     - Assert `docs/2026-06-10-ltx-av-lane/M0_RESULTS.md` exists and parses required checklist rows: Desktop nodes, headless nodes, pip-freeze before/after, scratch render, audio hash probe, NVML/wall-time tiers.

2. [schemas.py / tests] The plan adds family `audio_conditioned_video` but grounding `schemas.py` currently has no such family. Existing `assert set(FAMILIES) == set(FAMILY_REQUIRED_INPUTS)` will hard-fail unless both are edited.
   Concrete fix: add `"audio_conditioned_video"` to `FAMILIES` and `FAMILY_REQUIRED_INPUTS` with `("text_prompt", "audio_ref")`; add tests above.

3. [registry.py / tests] Do not add a new `EngineUsabilityReason` for missing Comfy nodes. Grounding `test_video_engine_registry_base_additive.py::test_usability_reason_has_the_six_codes` pins exactly six values.
   Concrete fix: classify missing `NODE_CLASS_MAPPINGS` as one existing reason, probably `MALFORMED_CONFIG` or `MISSING_MODEL`, and assert that in `test_ltx_av_node_availability_gate_missing_class`.

4. [Existing-test fallout] Create an explicit touch-list matrix before coding; current plan names categories but not files/cases.
   Required detector matrix:
   - Missing `__init__.py` guarded import -> `tests/test_video_ltx_av.py::test_ltx_av_registered_and_dark` fails.
   - Missing `schemas.py` family/input map -> `test_audio_conditioned_video_family_added` / round-trip fails.
   - Missing `role_compat.py` `MUSIC_VISUAL += "audio_ref"` -> `test_ltx_av_role_fit_audio_inputs` fails.
   - Missing registry docstring correction -> docstring test if one exists; otherwise no pytest detector. VERIFY-AT-BUILD.
   - Missing render-driver canvas tuple edit -> driver request/canvas test fails. VERIFY-AT-BUILD helper.
   - Missing prompt branch edit -> prompt fixture test fails. VERIFY-AT-BUILD existing prompt tests.
   - Missing synthetic timing slice fallback -> `test_synthetic_timing_audio_slice_only_ltx_av_music` fails.
   - Missing `_render_one(... request_template=request)` -> pass-through test fails.
   - Missing `ENGINE_FAMILY` entries -> exact map assertion fails. Existing file VERIFY-AT-BUILD.
   - Missing `SYNTH_FALLBACKS` entries -> guarded-import fallback test fails. Existing file VERIFY-AT-BUILD.
   - Missing force-map guard -> force-map role test fails.
   - Missing announcer portrait alias -> alias test fails.
   - Missing fake AV strip -> zero-audio ffprobe test fails.
   - Missing identity stamps -> canonical/manifest test fails.
   - Missing pad-tail marker -> marker test fails.
   - Missing AST no-brief rule -> AST test fails.
   - Missing cold-import discipline -> cold-import test fails.
   - Existing exact engine-count/dropdown tests must be updated to additive expectations. Files VERIFY-AT-BUILD.
   - Existing fallback-chain sweeps, likely `tests/test_video_retry_taxonomy.py` per grounding comment in `test_video_fallback_chain_additive.py`, must include both LTX chains.
   - Existing B7 forbidden import sweep must include new file and keep loop variable `imp`.

5. [GPU-gated vs CPU tests] The split is currently not pinned.
   Concrete fix:
   - CPU pytest covers all structural items above: registry, dropdown presence, role compat, schemas, `av_dims`, fallback termination, dark-lane fixtures, flag-off degrade via fakes/mocks, force-map guard, announcer alias, fake AV strip if ffmpeg/ffprobe test dependency exists, AST, cold import, identity fields, pad marker, `_render_one` kwarg.
   - M0/M4 operator GPU scripts only: real HuMo/LTX forwards, real Comfy graph render, NVML ceiling, wall time, real lip-sync eyeball, output visual quality, Desktop-vs-headless actual node availability.
   - No pytest may require network, CUDA, model weights, or real Comfy forward.

6. [test_audio_byte_identical.py] Do not claim a CPU-only dedicated byte-identical LTX-AV forced variant unless the “prune-to-node-7” path is grounded. Grounding shows the actual hash test requires `OTR_REGRESSION_RUNTIME=1`, ComfyUI, GPU, and `tests._run_baseline`.
   Concrete fix: dedicated forced-`ltx_av_*` master-audio hash is M4 GPU/operator-gated unless an existing CPU prune helper is verified. Add only a structural CPU test that LTX-AV dark/registered does not alter audio request hashes. Mark any prune-to-node-7 implementation VERIFY-AT-BUILD.

7. [Desktop-vs-headless node gate] Add both levels of test; one alone is insufficient.
   Concrete fix:
   - CPU: `assert_usable` unit with mocked `NODE_CLASS_MAPPINGS` missing one required class, plus flag-first ordering case where missing nodes are not inspected while flag is off.
   - M0 artifact: checklist row for Desktop build and headless build node presence.
   - No pytest should import real Comfy node packages at module scope; mock/inject only.

8. [Bug Bible discipline] Add regression pins for known at-risk bugs and decide the new row now.
   Concrete fix:
   - BUG-070 Sage: CPU test verifies Sage check order and classified fail-closed behavior.
   - BUG-291 reclaim: CPU lease test verifies prepare/load failure releases AS-3 lease, mirroring `test_humo_prepare_releases_lease_on_load_failure`.
   - BUG-265 family/schema: [ASSUMPTION] add schema round-trip tests above; verify actual BUG-265 meaning before naming it.
   - New Bug Bible row at ship for “silent rounding dims trap”: `1450x832` raises with nearest-valid hints; no silent rounding.

SHOULD-CONSIDER:
1. [test_video_ltx_av_io.py] If ffmpeg/ffprobe are not guaranteed in CI, split pure command-construction tests from an integration ffprobe test with an existing project skip pattern. Do not introduce a new framework.
2. [M0 artifact] Check in `docs/2026-06-10-ltx-av-lane/M0_RESULTS.md`, but add the parser test only after M0 lands; otherwise M1 commits will fail before the artifact exists.
3. [Dark-lane golden fixtures] Store minimal JSON/request fixtures, not rendered media, for CPU determinism. The media hash belongs to M4 runtime.
4. [Existing exact enumerations] Search before coding for exact literals of `FAMILIES`, `ENGINE_FAMILY`, `all_engine_names`, dropdown choices, and fallback chains. Current grounding only shows some patterns, not all files.
5. [Trail retention] Test pure fallback resolution and render-time trail separately; `resolve_fallback_chain` success does not prove the runtime trail is preserved.

OPEN-QUESTIONS:
1. VERIFY-AT-BUILD: exact render-driver test helper names/files for `_render_one`, `apply_engine_override`, prompt branch, synthetic timing slice, and manifest write.
2. VERIFY-AT-BUILD: exact classified reason to use for missing Comfy node classes under the six-value `EngineUsabilityReason` enum.
3. VERIFY-AT-BUILD: whether ffmpeg/ffprobe are mandatory test dependencies or need existing skip guards.
4. VERIFY-AT-BUILD: whether a CPU “prune-to-node-7” audio hash path actually exists; grounding does not show it.
5. VERIFY-AT-BUILD: shipped non-cast announcer portrait object id in `ledger["images"]`.
6. VERIFY-AT-BUILD: exact BUG-265 scope before attaching LTX-AV schema tests to that Bug Bible ID.

CUT THESE:
1. Do not add a new pytest framework or GPU pytest lane. Existing patterns are stdlib/pytest/subprocess only.
2. Do not add real LTX/HuMo render tests to pytest. They belong in M0/M4 operator scripts.
3. Do not add a new usability enum for node availability; existing tests pin six reasons.
4. Do not make the M0 result parser validate machine-specific numeric performance beyond presence/parse and documented rows; NVML/wall-time pass/fail is operator-gated.