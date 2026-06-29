VERDICT: no. The plan fails to unregister all NotImplementedError image adapters, misses several required requires_flag deletions across surviving adapters, and breaks the test suite's protocol parity and fallback/soak verification tests.

MUST-FIX BEFORE BUILD:
1. [Confirmed mechanics lines 42-45 / C3] Dark scaffold list is incomplete. Image engines `hidream_i1` and `sd35_large` both raise `NotImplementedError` in their `render_image()` methods: `nodes/_otr_image_engines/hidream_i1.py` lines 91-95 and `nodes/_otr_image_engines/sd35_large.py` lines 96-100. Both are registered by default in `nodes/_otr_image_engines/__init__.py` lines 61-79 and have `CAPABILITIES` rows in `nodes/_otr_image_engines/registry.py` lines 114-116 and 128-130.
   - Concrete fix: Include `hidream_i1` and `sd35_large` in the C3 unregistration work. Remove their imports from `nodes/_otr_image_engines/__init__.py`, remove their `@register` decorators, remove their rows from `CAPABILITIES` in `nodes/_otr_image_engines/registry.py`, and remove/update any tests targeting their registration.

2. [C6 lines 71-73] Protocol parity test breaks. Removing `requires_flag` from `EngineCore`, `VideoEngine`, and `ImageEngine` will break the protocol parity tests because `AudioEngine` is frozen and still declares `requires_flag`. The test suite asserts that video/image platforms are structural supersets of `AudioEngine` by iterating over `AudioEngine.__annotations__`:
   - `tests/test_video_platform_aseam.py` lines 95-99: `assert name in v_ann` and `v_ann[name] == typ`
   - `tests/test_image_platform_c1.py` lines 116-120: `assert name in i_ann` and `i_ann[name] == typ`
   - Concrete fix: Keep the `requires_flag` field in `EngineCore`, `VideoEngine`, and `ImageEngine` annotations as a deprecated/unused `Optional[str]` (set to `None` on all surviving adapters), OR modify `test_protocol_parity()` and `test_image_protocol_parity()` to skip checking `requires_flag` and document the divergence.

3. [Confirmed mechanics lines 46-48 / C2 / C6] Stale `requires_flag` usage on surviving/validated engines will cause `AttributeError` at runtime. The plan only lists `humo`, `wan_i2v`, `wan_ti2v`, `still_parallax`, `flux2_klein`, and `z_image_turbo` for flag-check deletion. However, other surviving engines also define and check `requires_flag` in their custom `assert_usable()` methods:
   - `mesh_stage` in `nodes/_otr_video_engines/eng_mesh_stage.py` lines 301, 392-397 (`OTR_ENABLE_MESH_STAGE`)
   - `ltx_video` in `nodes/_otr_video_engines/eng_ltx_video.py` lines 284, 373-377 (`OTR_ENABLE_LTX_VIDEO`)
   - `ltx_audio_in` in `nodes/_otr_video_engines/eng_ltx_av.py` lines 273, 386-389 (`OTR_ENABLE_LTX_AV`)
   - `visualizer` in `nodes/_otr_video_engines/eng_visualizer.py` lines 43, 80-83 (`OTR_ENABLE_VISUALIZER`)
   - `qwen_image` in `nodes/_otr_image_engines/qwen_image.py` line 84 (declares `requires_flag`)
   - `lumina_image` in `nodes/_otr_image_engines/lumina_image.py` line 68 (declares `requires_flag`)
   - Concrete fix: Remove `requires_flag` definitions and their corresponding `os.getenv` checks from all of these surviving video and image engine classes, while keeping their file-on-disk validation logic intact.

4. [C3 lines 66-67] Unregistering `triposg_talk` and `hunyuan3d_talk` breaks test suite fallbacks, fixtures, and granularity checks. These engines are hardcoded in `nodes/_otr_video_engines/render_driver.py`'s `SYNTH_FALLBACKS` (line 58), `ENGINE_FAMILY` (line 66-67), `OOM_ENGINES` (line 96), and `EXPECTED_OOM_TRAIL` (line 106).
   - Furthermore, `three_d_locked_slots` in `nodes/otr_image_director.py` will raise `ValueError` (lines 112-119) because `triposg_talk` is named in the test policies but is not registered.
   - Tests like `test_triposg_talk_locks_via_real_registry` (`tests/test_image_platform_c1.py` L357-366) and the entire soak/render_driver suite (`tests/test_video_soak_fixture.py` L55-81, `tests/test_video_render_driver.py` L25-31, `tests/test_video_render_driver_additive.py`) will crash/fail because they expect `triposg_talk` to be registered.
   - Concrete fix: Either (a) register a synthetic/mock `character_3d` engine specifically for tests inside the test suite, or (b) refactor `three_d_locked_slots` and all affected tests/fixtures to use a test-registered stub instead of expecting the real registry to hold `triposg_talk`.

5. [C4 lines 68-70] `VALIDATED_ENGINES` clean up misses direct test assertions. Dropping `VALIDATED_ENGINES` and `validated_engine_names()` from registries will break multiple tests that assert on them directly:
   - `tests/test_still_aspect_and_labels.py` line 132: `assert parsed == set(vreg.validated_engine_names())`
   - `tests/test_video_triposr.py` lines 45-46: asserts `"triposr" not in vreg.VALIDATED_ENGINES`
   - `tests/test_ltx_audio_in_engine.py` line 71: asserts `ltx_audio_in` is in `vreg.VALIDATED_ENGINES`
   - `tests/test_video_cheap_render.py` line 99: refers to `validated_engine_names()`.
   - Concrete fix: Update C4 to explicitly clean up/remove these direct assertions in the same chunk that deletes the dropdown filter.

SHOULD-FIX:
1. [Sequencing line 63] "no push" violates the repository operating rules. `CLAUDE.md` lines 136-137 require pushing every green commit immediately to `v2.0-alpha`.
   - Concrete fix: Change sequencing to "suite green + commit + push per chunk".

2. [C2/C5 lines 49-54] Dep-pilot manifest tests break without `requires_flag`. The pilot tests assert that the engine's `requires_flag` matches the manifest `flag` keys:
   - `tests/test_video_dep_pilot.py` line 109: `assert adapter.requires_flag == spec["flag"]`
   - Concrete fix: Decouple the pilot scripts and tests from `requires_flag` entirely (e.g., check against a list of known names or check the environment variables directly).

3. [Confirmed mechanics lines 34-37] Dead enum documentation. `GATED_BY_FLAG` is kept as a dead enum member, but comments describing active flag gating in `nodes/_otr_shared/engine_registry_base.py` lines 67-83 and 191-204 will mislead future maintainers.
   - Concrete fix: Update comments in `engine_registry_base.py` to clarify that `GATED_BY_FLAG` is deprecated and unused for video/image.

OPTIONAL / NICE-TO-HAVE:
- Add a test verifying that no registered engine has `requires_flag` (or that it is always `None`) and that no registered engine's render path raises `NotImplementedError`.
- Add a test checking that all engines in `CAPABILITIES` are registered, preventing orphaned metadata rows.

CUT THESE (over-engineering):
1. Cut `VALIDATED_ENGINES` and `validated_engine_names()` completely instead of keeping them as aliases. Once the dropdowns migrate to `all_engine_names()`, keeping dead variables invites accidental reuse.
2. Cut `triposg_talk` and `hunyuan3d_talk` from `render_driver.SYNTH_FALLBACKS` once the tests/fixtures are refactored, as keeping fallbacks for unregistered engines is dead runtime code.

[ASSUMPTION] We assume that `AudioEngine` is frozen and must not be edited.
[ASSUMPTION] We assume that the operator manual validation policy allows hard errors at runtime for buggy but registered engines, and the test suite should continue to assert loud errors in those cases.
