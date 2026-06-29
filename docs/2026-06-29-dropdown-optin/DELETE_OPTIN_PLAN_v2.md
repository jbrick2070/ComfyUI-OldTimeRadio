# PLAN v2: no code-gating at all -- registry IS the menu (operator final)

## End goal (operator 2026-06-29, supersedes v1's two "decisions")
"No hidden gated things. Once we ADD a model it's usable -- buggy maybe, may hard-
fail, that's OK -- but selectable. No behind-the-scenes models waiting to be
promoted. No hidden non-validated: I'll do validation MANUALLY; it won't be gated
in code."

Consequences (all gating/curation that lives in CODE is removed; the operator
curates by what is REGISTERED):
1. NO `requires_flag` opt-in gate (delete it -- base + adapters + field).
2. NO `VALIDATED_ENGINES` / `validated_engine_names()` dropdown filter -- the
   OTR_VideoDirector + image dropdowns list EVERY REGISTERED engine
   (`all_engine_names()`), not a tested-only subset. Validation is manual.
3. NO production "validated-only" guard (v1 DECISION 2 is REVERSED -- do NOT add
   it). force-map / raw-JSON / lsync-base may name any REGISTERED engine.
4. Dark scaffolds that cannot render (raise `NotImplementedError`) are
   UNREGISTERED (v1 DECISION 1) -- a model is registered ONLY when it actually
   runs. They return to the registry when the operator builds + adds them. No
   registered-but-dark "waiting to be promoted" engines remain.
5. A registered engine that is merely BUGGY/unfinished (renders, but rough) STAYS
   registered + selectable (operator: hard-fail is acceptable).

## What stays (real, non-promotional gates)
- Files-on-disk: `assert_usable` still raises `MISSING_MODEL` if the checkpoint is
  absent (this is "the files aren't here", not "waiting to be promoted").
- Role compatibility: `role_compat` still filters which engines fit which slot
  (an audio-face engine can't serve a text-only role) -- structural, not a promo.
- No-fallback render still RAISES LOUD on a real failure (a buggy pick hard-fails,
  exactly as the operator wants).
- AUDIO spine FROZEN + untouched (separate registry + separate enum copy).

## Confirmed mechanics (grounded)
- Gate to delete: `engine_registry_base.py` `assert_usable` L222-228 (the
  `GATED_BY_FLAG` block). KEEP the `GATED_BY_FLAG` enum MEMBER (dead) -- the
  protocol-parity tests (`test_video_platform_aseam.py`,
  `test_image_platform_c1.py`) assert the shared enum equals audio's frozen copy.
- Dropdown filter to delete: `registry.validated_engine_names()` +
  `VALIDATED_ENGINES` frozenset; `otr_video_director._video_model_combo` /
  `_image_model_combo` switch to `all_engine_names()`. The `_image_engines`
  registry has the parallel pair -- do both.
- Dark scaffolds to UNREGISTER (render path is NotImplementedError): `triposr`,
  and `triposg_talk` / `hunyuan3d_talk` / `trellis_talk` (in `eng_character_3d.py`).
  VERIFY each is truly non-rendering before removing; remove its `@register` +
  CAPABILITIES row + any VALIDATED entry; keep the source file (returns later).
- Render-ready engines whose flag-gate is removed (they just render): `humo`,
  `wan_i2v`, `wan_ti2v`, `still_parallax`, `flux2_klein`, `z_image_turbo`
  (default-ON `visualizer`/`ltx_video`/`ltx_av`/`flux_gen1` already ungated).
- Harness (keep their own manifests; just stop them gating on the flag):
  `otr_video_gpu_smoke.py` drop the `flag_set` ready-assertion (L168-170, L209);
  `otr_coverage_sweep.py` drop the `OTR_ENABLE_WAN_*` `acceptance_preflight`
  (L121-132); `otr_video_dep_pilot.py` / `otr_image_dep_pilot.py` KEEP the
  `OPT_IN_ENGINES` probe manifest (module/class/forward metadata not in
  CAPABILITIES), delete only the dead `flag` keys.
- `requires_flag` field: once no engine sets it and nothing reads it at runtime,
  remove it from `EngineCore` (video+image) + every row. (Audio EngineCore is a
  separate copy -- leave it.)

## Done in C1 already
Reverted the interim option-B (`apply_selection_enable_set`/`_restore_enable_set`
+ the `run_real_episode` try/finally) and DELETED its test. render_driver AST OK.

## Sequencing (suite green + commit per chunk; no push)
- C2: delete the base `GATED_BY_FLAG` gate + the render-ready adapters' flag
  checks; update registry-usability + adapter tests.
- C3: unregister the dark scaffolds (+ remove their VALIDATED entries); update
  their tests (now "not registered", not "gated").
- C4: drop the `VALIDATED_ENGINES`/`validated_engine_names()` dropdown filter ->
  `all_engine_names()`; update `test_tested_only_dropdown_gate.py` (contract flips
  to "every registered engine is selectable").
- C5: harness (gpu-smoke / coverage / dep-pilot) + their tests.
- C6: remove the `requires_flag` field (video+image) + GATED_BY_FLAG-as-dead-enum
  note; final full suite + Bug Bible.

## Coding-help questions for the panel
1. Enumerate EXACTLY which registered video+image engines have a
   NotImplementedError render path (must be unregistered) vs render (keep). Did I
   miss any (mesh_stage? any image engine: sd35_large/qwen_image/lumina_image/
   hidream_i1)?
2. Dropping `validated_engine_names()` -> `all_engine_names()`: does anything ELSE
   consume `VALIDATED_ENGINES` (soak acceptance, coverage, a test fixture) that
   would break or silently change meaning?
3. Unregistering an engine that another engine names as its `fallback_engine`
   (e.g. `triposr` is `mesh_stage`'s? `still_parallax`'s?) -- does the fallback
   resolver fail closed cleanly, or does it crash? Sequence the unregister safely.
4. After removing `requires_flag` from `EngineCore`, any dataclass/`__init__` or
   audio-shared import that breaks? (audio has its own copy -- confirm no shared
   import.)
5. Safest commit order so the suite stays green at EACH chunk (e.g. is C4 before
   or after C3?).
