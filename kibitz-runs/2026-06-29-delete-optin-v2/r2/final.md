# Delete code-gating across ALL modalities -- FINAL coding plan (kibitz r2 synth)

Operator: no opt-in / validation / promotion gates ANYWHERE (video, image, voice,
LLM). Registry IS the menu; registered => selectable (may hard-fail). Manual
validation. Panel = Codex + Antigravity, both grounded; findings CONFIRMED.

## Behavior to achieve
- Picking ANY registered engine renders it (or hard-fails LOUD) -- no flag, no
  validated-subset filter, no guard.
- Real gates that STAY: file-on-disk (MISSING_MODEL), role compatibility, the
  no-fallback LOUD raise. AUDIO byte-identical master/mux output STAYS frozen +
  verified (output determinism, not a promotion gate).

## A. Unregister the dark scaffolds (NotImplementedError render path)
video: `triposr`, `triposg_talk`, `hunyuan3d_talk`, `trellis_talk`
image: `hidream_i1`, `sd35_large`
For each: drop the `@register` / package import (`_otr_*_engines/__init__.py`) +
the CAPABILITIES row + any VALIDATED entry. KEEP the source file (returns when
built). Ripple to fix IN THE SAME CHUNK:
- render_driver `SYNTH_FALLBACKS` (L58), `ENGINE_FAMILY` (L66-67), `OOM_ENGINES`
  (L96), `EXPECTED_OOM_TRAIL` (L106) -- drop the unregistered names.
- `otr_image_director.three_d_locked_slots` (L112-119) -- no longer references a
  registered talker; refactor its test policy or use a test-registered stub.
- fixtures/tests that EXPECT them registered: `test_video_render_driver(.py)` +
  `_additive`, `test_video_soak_fixture`, `test_still_aspect_and_labels`,
  `test_image_platform_c1` (L357-366 triposg lock), `test_video_triposr`.
  Replace the synthetic 3D/OOM engine with a test-registered stub (cleanest) so
  suite-green-per-chunk holds.

## B. Remove the flag GATE everywhere (inventory-driven, NOT 6 named engines)
- Base: `engine_registry_base.py` `assert_usable` -- delete the GATED_BY_FLAG
  block (L222-228). Update the misleading docstrings (L67-83, L191-204).
- Adapter `assert_usable` os.getenv(flag) checks to delete (KEEP disk checks):
  video `humo`, `wan_i2v`, `wan_ti2v`, `still_parallax`, `mesh_stage`
  (OTR_ENABLE_MESH_STAGE), `ltx_video` (OTR_ENABLE_LTX_VIDEO), `ltx_av`/
  `ltx_audio_in` (OTR_ENABLE_LTX_AV), `visualizer` (OTR_ENABLE_VISUALIZER);
  image `flux2_klein`, `z_image_turbo` (rely on base only -> just disk),
  `qwen_image`, `lumina_image` (declare flag).
- Field: KEEP `requires_flag: Optional[str]` on EngineCore/Video/Image but set it
  to `None` on every surviving adapter (do NOT delete the annotation -- the
  protocol-parity tests iterate `AudioEngine.__annotations__` and audio is frozen
  with the field; a vestigial None kills the gate without breaking parity).
  Keep `GATED_BY_FLAG` enum member (dead) for the same parity reason.

## C. Drop the validated-subset dropdown filter
- Delete `VALIDATED_ENGINES` + `validated_engine_names()` from
  `_otr_video_engines/registry.py` (L277-325) and `_otr_image_engines/registry.py`.
- `otr_video_director._video_model_combo` / `_image_model_combo` +
  `otr_image_director` -> use `all_engine_names()`. Dropdown == registry.
- Flip the consuming tests: `test_tested_only_dropdown_gate` (contract becomes
  "every registered engine is selectable"), `test_still_aspect_and_labels:132`,
  `test_video_triposr:45-46`, `test_ltx_audio_in_engine:71`,
  `test_video_cheap_render:99`.

## D. Harness -- decouple from the flag (do NOT infer from CAPABILITIES)
- `otr_video_gpu_smoke.py`: drop the `flag`/`flag_set` ready-assertion (L168-170,
  L209).
- `otr_coverage_sweep.py`: drop the `OTR_ENABLE_WAN_*` acceptance_preflight
  (L121-132).
- `otr_video_dep_pilot.py` / `otr_image_dep_pilot.py`: KEEP the probe manifest
  (module/class/forward metadata not in CAPABILITIES) but rename OPT_IN_ENGINES ->
  "probe engines" and delete the `flag` keys; fix `test_video_dep_pilot:109`
  (`adapter.requires_flag == spec["flag"]`) + `test_audio_dep_pilot`.

## E. Voice + LLM (operator extension) -- same pattern, audio output frozen
- Audio voice engines: remove the requires_flag GATE checks from their
  `assert_usable` (eng_bark/eng_kokoro/eng_indextts2/eng_chatterbox/eng_dia/
  eng_stable_audio*/eng_musicgen) + the audio registry base gate, KEEP field=None.
  HARD CONSTRAINT: `test_audio_byte_identical` MUST stay green -- the DEFAULT
  voice/music engines + the master-mix/mux path are UNCHANGED; only selectability
  of non-default engines changes. Verify byte-identical every chunk.
- LLM: remove any opt-in gate on writer/LLM selection (e.g. OTR_ENABLE_OPENROUTER
  as a *gate*) so any configured LLM is selectable; keep the API-key/file presence
  check (that is "creds present", not a promotion gate). Verify exact consumer
  before editing.

## Sequencing (suite + Bug Bible green per chunk; commit per chunk)
C2 base gate + video/image adapter gate removal (field->None) + parity holds ->
unblocks the smoke. C3 unregister dark scaffolds + ripple/fixtures. C4 drop
VALIDATED filter + tests. C5 harness + tests. C6 voice+LLM gates (byte-identical
guard). C7 docstring/comment cleanup + a guard test (no registered engine gates on
a flag; no registered engine renders NotImplementedError).

OPEN: push policy -- CLAUDE.md says push-per-green-chunk; this session's kickoff
said "do NOT push unprompted". Honoring the kickoff (commit only) until the
operator confirms.
