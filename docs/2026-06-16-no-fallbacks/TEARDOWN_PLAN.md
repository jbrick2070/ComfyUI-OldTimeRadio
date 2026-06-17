# No-fallbacks teardown -- status + go-forward (2026-06-16)

Operator directive (2026-06-16, verbatim intent): **no fallbacks at all** -- a
failed engine fails LOUD; no engine swap, no still-image floor. "It proves a
proven model path works... this is art, not a space shuttle." Plus: delete the
14B `humo` engine, keep only the two small ones (`humo_1.7B` portrait +
`humo_1.7B_169` 16:9).

## SHIPPED (green + pushed)

**Chunk 1 -- render path fails loud. Commit `547671d` on `v2.0-alpha`.**
- `render_shot` now does a SINGLE attempt and raises `RenderError` on a hard
  failure. No chain walk, no restamp, no radio floor. The signature is unchanged,
  so `run_episode`'s degrade-handling (the `for rec in decisions` loop + the AS-2
  prune) is now a harmless no-op (decisions is always `[]`).
- New `RenderError(RuntimeError)` in `render_driver.py`.
- `tests/test_video_render_driver_additive.py`: the 6 degrade-behavior tests
  rewritten to assert the loud-failure contract.
- Suite **4452 passed / 33 skipped / 0 failed**; Bug Bible green.

This already delivers the BEHAVIOR the operator asked for: any engine that can't
render fails the episode loud, with no silent degrade and no floor.

## NOT YET DONE (deferred -- entangled, needs an operator design call)

The fallback *machinery* and the 14B `humo` are now DEAD/bypassed but still
present. Removing them cleanly is a ~30-file change because the 14B and the
fallback chain are woven into the GPU **soak self-test** (which exists ONLY to
prove the now-removed degrade-to-floor behavior). That raises a design question:

> **OPEN QUESTION for the operator:** with no fallbacks, what should the A-S7.5
> soak / A-ship gate verify? It currently forces an OOM mid-episode and asserts
> the chain degrades `triposg_talk -> humo -> humo_1.7B -> latentsync ->
> still_kenburns`. With no fallbacks that invariant is gone. Recommended new
> gate: "N beats each render a real clip, frozen audio byte-identical, two runs
> deterministic, VRAM <= ceiling" -- i.e. drop the OOM-forcing + trail asserts.
> Confirm before rewriting the gate.

### Blast radius (already mapped) for the full rip-out

Production:
- `nodes/_otr_video_engines/eng_humo.py` -- remove `@register` from `HuMoEngine`
  (keep the class as the base for the two 1.7B subclasses).
- Remove `fallback_engine` attr from every engine: eng_humo (x2), eng_character_3d
  (x3 -> were "humo"), eng_ltx_av (x2), eng_latentsync, eng_still_parallax,
  eng_mesh_stage, eng_ltx_video.
- `nodes/_otr_video_engines/registry.py` -- delete the `"humo"` capability profile
  (the test asserts `CAPABILITIES == all_engine_names()`).
- `nodes/_otr_video_engines/render_driver.py` -- remove `make_fallback_of`,
  `SYNTH_FALLBACKS`, `FLOOR_NAMES`, `UNIVERSAL_FLOOR`, the `resolve_fallback_chain`
  import, `RenderFloorError`, the OOM-soak apparatus (`run_gpu_soak`,
  `assert_soak_ok`, `_episode_facts`, `assemble_report`, `_norm_decisions`,
  `EXPECTED_OOM_TRAIL`, `_CHAR3D`, `OOM_ENGINES`, `_PROFILES` "humo"), and the dead
  `for rec in decisions` + AS-2 prune block in `run_episode`. Repoint `engine_family`
  / `ENGINE_FAMILY` "humo". `render_single` default "humo" -> "humo_1.7B".
- `nodes/otr_video_render_batch.py` -- drop the `"soak"` mode (or repoint to the
  redefined gate) from `INPUT_TYPES` + `render()`.
- `__init__.py` -- `/otr/video_render_single` default "humo" -> "humo_1.7B";
  `/otr/video_render_soak` route (remove or repoint).
- `scripts/otr_video_dep_pilot.py` -- remove the `"humo"` OPT_IN entry.
- `nodes/_otr_shared/fallback.py` -- DELETE.

Scripts (GPU dev tools, not in pytest; update or retire):
- `scripts/otr_video_soak.py`, `scripts/otr_video_gpu_smoke.py`,
  `scripts/_otr_soak_capstone.py`, `scripts/_otr_soak_marathon.py`,
  `scripts/coverage_sweep*` -- all built around force-OOM -> floor.

Tests to fix/delete (humo, fallback_engine, machinery, soak):
- DELETE `tests/test_video_fallback_chain_additive.py` (tests `resolve_fallback_chain`).
- `tests/test_video_humo.py` (humo registration + the fallback-chain test),
  `tests/test_capability_profiles.py` (humo + the heavy-engine override -> use
  `wan_i2v`), `tests/test_video_soak_fixture.py` (soak apparatus),
  `tests/test_ltx_av_driver_wiring.py` (`SYNTH_FALLBACKS`),
  `tests/test_video_still_parallax.py` / `test_video_mesh_stage.py` /
  `test_video_character_3d.py` (`fallback_engine` asserts),
  `tests/test_image_platform_c1.py` (humo fixture engine_id),
  `tests/test_look_qa_round5.py`, `tests/test_ltx_open_health.py`,
  `tests/test_brief_prompt_finishing.py`, `tests/test_coverage_sweep_acceptance.py`
  (humo -> humo_1.7B), and the `make_fallback_of` / `build_soak_fixture` /
  `classify_failure(OomSignal)` tests in `test_video_render_driver_additive.py`.

### Recommended approach
Do it as ONE coder pass (it is mostly mechanical once the soak-gate question is
answered), suite-driven: make the production removals, run the full suite, fix
the enumerated tests, repeat to green, then commit+push. Revert baseline is
`547671d` (chunk 1, green).
