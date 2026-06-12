<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. The plan lacks concrete test-file mappings, the "forgot-it detector matrix" is absent, the byte-identical guard for the new lane is unspecified, and the desktop-vs-headless node gate is not tested.

MUST-FIX BEFORE BUILD:
1. [pass05 §1] TEST ENUMERATION INCOMPLETE: The plan says "enumerate the full test list" but provides zero test-file names. MUST produce a definitive list mapping each claim to a specific test file + function name. Example: `test_video_engine_registry_base_additive.py::test_register_returns_adapter_and_records_instance` mirrors the existing pattern. For the new lane, create `test_video_ltx_av.py` with at minimum:
   - `test_ltx_av_talk_registered_and_dark` (mirrors `test_humo_registered_and_dark`)
   - `test_ltx_av_music_registered_and_dark`
   - `test_ltx_av_talk_role_fit` / `test_ltx_av_music_role_fit` (mirrors `test_humo_role_fit_audio_driven_face`)
   - `test_ltx_av_required_inputs_match_family_schema` (mirrors `test_humo_required_inputs_match_family_schema`)
   - `test_ltx_av_assert_usable_flag_then_install` (mirrors `test_humo_assert_usable_flag_then_install_tolerates_sage`)
   - `test_ltx_av_fallback_chain_converges_on_radio_floor` (mirrors `test_humo_fallback_chain_converges_on_radio_floor`)
   - `test_ltx_av_cold_import_no_heavy_libs` (mirrors `test_cold_import_humo_no_heavy_libs`)
   - `test_ltx_av_source_is_ascii_no_em_dash` (mirrors `test_humo_source_is_ascii_no_em_dash`)
   - `test_ltx_av_build_render_request_deterministic` (mirrors `test_humo_build_render_request_deterministic`)
   - `test_ltx_av_canonicalize_silent_bt709` (mirrors `test_humo_canonicalize_silent_bt709`)
   - `test_ltx_av_dims_unit_cases` (NEW: covers W/H/frames, nearest-valid hints, 1472x832 passes, 1450x832 raises, frames snap-up cases T=25, T=26, cap)
   - `test_ltx_av_fake_mp4_zero_audio_stream` (NEW: ffprobe assert)
   - `test_ltx_av_ast_no_brief_import` (NEW: AST check)
   - `test_ltx_av_identity_stamps` (NEW: CanonicalClip/manifest engine_id)
   - `test_ltx_av_pad_tail_marker` (NEW: LOUD marker emission)
   - `test_ltx_av_request_template_pass_through` (NEW: TypeError guard)
   - `test_ltx_av_chain_termination_5hop_talk` (NEW: 5-hop talk chain)
   - `test_ltx_av_chain_termination_3hop_music` (NEW: 3-hop music chain)
   - `test_ltx_av_dark_lane_golden_fixtures` (NEW: existing-engine requests bit-identical)
   - `test_ltx_av_flag_off_degrade` (NEW: flag-off render-time degrade)
   - `test_ltx_av_force_map_role_guard` (NEW: force-map role guard)
   - `test_ltx_av_announcer_portrait_alias` (NEW: announcer portrait alias)
   - `test_ltx_av_cold_import_with_new_module` (NEW: V-12 with the new module)

2. [pass05 §2] EXISTING-TEST FALLOUT MATRIX MISSING: The plan says "Name the files from grounding where possible" and "What existing test would FAIL TODAY if the coder forgets each touch-list edit" but provides zero entries. MUST produce a table. Example:
   - Edit: `nodes/_otr_video_engines/__init__.py` (guarded import). Forgot-it detector: `test_video_engine_registry_base_additive.py::test_all_engine_names_sorted` would fail because the new engine name is missing from the dropdown.
   - Edit: `nodes/_otr_video_engines/schemas.py` (family + required-inputs map). Forgot-it detector: `test_video_ltx_av.py::test_ltx_av_required_inputs_match_family_schema` (new test) would fail.
   - Edit: `nodes/_otr_video_engines/registry.py` (docstring: family list + correct ShotLock-assert claim). Forgot-it detector: `test_video_engine_registry_base_additive.py::test_register_returns_adapter_and_records_instance` would fail if the registry rejects the new engine.
   - Edit: `nodes/_otr_shared/role_compat.py` (MUSIC_VISUAL += "audio_ref"). Forgot-it detector: `test_video_ltx_av.py::test_ltx_av_music_role_fit` (new test) would fail.
   - Edit: `nodes/_otr_video_engines/render_driver.py` (a-g). Forgot-it detectors: (a) `test_video_ltx_av.py::test_ltx_av_canvas_tuple` (new test) would fail; (b) `test_video_ltx_av.py::test_ltx_av_prompt_gate` (new test) would fail; (c) `test_video_ltx_av.py::test_ltx_av_synthetic_timing_slice` (new test) would fail; (d) `test_video_ltx_av.py::test_ltx_av_request_template_pass_through` (new test) would fail; (e) `test_video_ltx_av.py::test_ltx_av_engine_family` (new test) would fail; (f) `test_video_ltx_av.py::test_ltx_av_force_map_role_guard` (new test) would fail; (g) `test_video_ltx_av.py::test_ltx_av_announcer_portrait_alias` (new test) would fail.
   - Edit: `nodes/_otr_video_engines/render_driver.py` SYNTH_FALLBACKS. Forgot-it detector: `test_video_ltx_av.py::test_ltx_av_synth_fallbacks` (new test) would fail.
   - Edit: `nodes/_otr_video_engines/render_driver.py` b7 sweep AST loop var `imp`. Forgot-it detector: `test_video_ltx_av.py::test_ltx_av_b7_sweep_loop_var` (new test) would fail.

3. [pass05 §4] BYTE-IDENTICAL GUARD UNSPECIFIED: The plan asks "Does the new lane need a DEDICATED variant... and can that run CPU-only via the existing prune-to-node-7 trick?" but provides no answer. MUST specify: the new lane requires a DEDICATED byte-identical test `test_audio_byte_identical_ltx_av.py` that captures a baseline with `ltx_av` forced for a shot, then asserts the master audio hash is unchanged from the default-config baseline. This test is M4-GPU-only because it requires a real render. The prune-to-node-7 trick is NOT applicable because the audio path is unchanged; the test proves the new lane's audio path is bit-identical to the existing audio path.

4. [pass05 §5] DESKTOP-VS-HEADLESS NODE GATE UNTESTED: The plan asks "Where is that gate TESTED?" but provides no answer. MUST add a CPU test `test_video_ltx_av.py::test_ltx_av_desktop_vs_headless_node_gate` that mocks `NODE_CLASS_MAPPINGS` missing one class and asserts `assert_usable` raises `MALFORMED_CONFIG` (or the appropriate reason). Also add an M0 checklist row per build (Desktop + headless) verifying node presence.

5. [pass05 §6] REGRESSION DISCIPLINE MISSING: The plan asks "which existing BUG-IDs are at risk of regression" but provides no answer. MUST add a section to the test plan listing BUG-070 (Sage), BUG-291 (reclaim), BUG-265 (family) and the new silent-rounding dims trap as a NEW Bug Bible row. Each must have a dedicated regression test in `test_video_ltx_av.py`.

6. [pass05 §3] GPU-GATED VS CPU TESTS SPLIT UNSPECIFIED: The plan asks "Define the exact split" but provides no answer. MUST specify: all tests in `test_video_ltx_av.py` are CPU-provable except the M0/M4 GPU scripts (real render, NVML ceiling, wall time, lip-sync eyeball). The M0 sheet should be a CHECKED-IN artifact (`docs/2026-06-10-ltx-av-lane/M0_RESULTS.md`) that a later test (`test_video_ltx_av.py::test_m0_sheet_exists_and_parses`) asserts exists and parses.

SHOULD-FIX:
1. [pass05 §1] The plan says "mirror existing patterns" but `test_video_ltx_av.py` is not listed in the touch list. Add it to the M1 milestone task list explicitly.
2. [pass05 §1] The plan says "cover at minimum: registry/dropdown additive presence" but the existing `test_video_engine_registry_base_additive.py` tests a LOCAL registry, not the global one. The new lane's dropdown presence must be tested against the global registry in `test_video_ltx_av.py`.
3. [pass05 §2] The plan says "the b7 sweep's AST loop var must be `imp`" but the grounding does not show the b7 sweep code. VERIFY-AT-BUILD: the AST test must parse the b7 sweep's loop and assert the variable name is `imp`, not `i` or `engine`.
4. [pass05 §2] The plan says "What existing test would FAIL TODAY if the coder forgets each touch-list edit" but the grounding does not show the existing test files for `role_compat.py` or `render_driver.py`. VERIFY-AT-BUILD: the forgot-it detector matrix must be cross-referenced with the actual test files.

OPEN-QUESTIONS:
1. [pass05 §4] The plan asks "can that run CPU-only via the existing prune-to-node-7 trick (audio path without video cost)?" The answer is NO. The byte-identical test for the new lane must render a real clip with `ltx_av` forced, which requires GPU. The prune-to-node-7 trick only works for audio-only tests.
2. [pass05 §5] The plan says "PR #13111 nodes may exist in one build and not the other." VERIFY-AT-BUILD: which specific nodes are in PR #13111? The test must mock the exact missing class name.
3. [pass05 §6] The plan says "does any new lane behavior deserve a NEW Bug Bible row at ship (e.g. the silent-rounding dims trap)?" YES. The silent-rounding dims trap (av_dims rounding) is a new behavior that can cause silent failures. Add a Bug Bible row for it.
4. [pass05 §3] The plan says "M0 sheet as a CHECKED-IN artifact" but does not specify the format. VERIFY-AT-BUILD: the M0 sheet must be a Markdown file with a parseable table format so the assertion test can read it.