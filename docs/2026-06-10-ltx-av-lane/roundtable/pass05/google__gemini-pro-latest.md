<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The testing strategy is comprehensive but lacks the `request_template=None` safety guard, requires specific M4-GPU isolation for the audio byte-identical test, and needs explicit fallout handling for the taxonomy and AST sweeps.

MUST-FIX BEFORE BUILD:
1. [TEST ENUMERATION / WIRING] `assert_usable` signature crash. The protocol defines `request_template=None`. If `assert_usable` blindly accesses `request_template.canvas` for `av_dims`, it will raise `AttributeError` when queried by legacy paths or external tools. Fix: Add a CPU unit test in `test_eng_ltx_av_additive.py` asserting `assert_usable(..., request_template=None)` passes without error, and add an `if request_template:` guard in the adapter.
2. [EXISTING-TEST FALLOUT] `test_video_retry_taxonomy.py` sweeps the real declared fallback chain. Adding two new engines will break it. Fix: Update the test to expect `ltx_av_talk -> humo -> humo_1.7B -> latentsync -> still_kenburns` and `ltx_av_music -> ltx_video -> still_kenburns`.
3. [EXISTING-TEST FALLOUT] The `b7` forbidden imports AST test. Fix: Add `nodes/_otr_video_engines/eng_ltx_av.py` to the `b7` sweep target list. The AST loop variable must be exactly `imp` (repo gotcha).
4. [BYTE-IDENTICAL GUARD] Dedicated variant must be M4-GPU-only. The CPU "prune-to-node-7" trick strips the video nodes entirely, defeating the purpose of testing if the video adapter's `prepare`/`render_clip` leaks seed mutations. Fix: Define a dedicated M4-GPU-only test function in `test_audio_byte_identical.py` that forces LTX-AV via env vars/overrides and asserts the audio hash matches the baseline.
5. [DESKTOP-VS-HEADLESS NODE GATE] Missing test for node absence. Fix: Add a CPU unit test in `test_eng_ltx_av_additive.py` that mocks `sys.modules["nodes"].NODE_CLASS_MAPPINGS` to omit the LTX node class, asserting `assert_usable` catches it and raises `EngineUnusable` with `MISSING_MODEL`.

SHOULD-FIX:
1. [GPU-GATED VS CPU TESTS] The M0 hardware sheet should not be an untracked manual step. Fix: Check in `docs/2026-06-10-ltx-av-lane/M0_RESULTS.md` and add a CPU pytest that asserts the file exists and contains "NVML" to ensure the operator actually ran the smoke test.
2. [TEST ENUMERATION] Produce the definitive test list. Fix: Implement the following structure:
   - `test_eng_ltx_av_additive.py` (mirrors `test_video_humo.py`): Proves registry presence, dark-by-default, `assert_usable` order, `av_dims` unit cases (1472x832 passes, 1450x832 raises, snap-up T=25->25, T=26->33), cold-import (V-12), fake-AV-mp4 strip (`-an`) + zero-audio ffprobe, identity stamps, pad-tail marker emission.
   - `test_ltx_av_chains_additive.py` (mirrors `test_video_fallback_chain_additive.py`): Proves 5-hop talk chain, 3-hop music chain, and trail retention.
   - `test_ltx_av_driver_wiring.py` (mirrors `test_brief_prompt_finishing.py`): Proves role compat, flag-off render-time degrade, force-map role guard, and announcer portrait alias.
   - `test_ltx_av_golden_fixtures.py`: Proves dark-lane golden fixtures (existing-engine requests bit-identical).
3. [EXISTING-TEST FALLOUT] "Forgot it" detector matrix. Fix: Ensure the suite fails exactly here if a touch is missed:
   - Forget `schemas.py` family -> module-level `assert set(FAMILIES) == set(FAMILY_REQUIRED_INPUTS)` fails on import.
   - Forget `role_compat.py` -> `test_video_role_compat.py` fails.
   - Forget `__init__.py` guarded import -> `test_eng_ltx_av_additive.py` registry presence fails.
   - Forget `render_driver.py` `ENGINE_FAMILY` -> `test_video_render_driver.py` fails.
   - Forget `render_driver.py` `_render_one` -> `test_eng_ltx_av_additive.py` `request_template` mock fails.
   - Forget `render_driver.py` synthetic-timing -> `test_ltx_av_golden_fixtures.py` fails.

OPTIONAL / NICE-TO-HAVE:
- Add a test asserting `degradation_trail` correctly appends `ltx_av_talk` before hopping to `humo` during a render-time fallback.
- In `test_audio_byte_identical.py`, add a comment explicitly explaining why the LTX-AV forced test cannot use the prune trick.

CUT THESE (over-engineering):
1. [WIRING] Group-prune wiring claims. The plan already correctly notes "claim removed" for group pruning, so nothing to cut here, but ensure no tests are written for group pruning since the lane has no provider groups.

[ASSUMPTION] `test_video_retry_taxonomy.py` and the `b7` forbidden imports AST test exist and function as described in the prompts/docstrings.
[ASSUMPTION] `ledger["images"]` is accessible to `render_driver.py` for the announcer portrait alias resolution.
[ASSUMPTION] `NODE_CLASS_MAPPINGS` is the standard ComfyUI dictionary used to check node presence in the target environment.