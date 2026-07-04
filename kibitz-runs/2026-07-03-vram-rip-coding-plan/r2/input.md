# CODING PLAN: rip ALL VRAM tiers + the runtime ceiling -> clean workflow

Operator (2026-07-03): remove the hard-baked VRAM tier classification AND the runtime OOM
ceiling. The per-hardware tier JSONs (8/16/cpu/cloud, built later) are the protection;
users pick the JSON that fits and flip switches themselves. KEEP only reclaim/free
(frees models between beats) + live VRAM measurement as telemetry. Grounded on the Fable
audit (VRAM_TIER_RIP_AUDIT.md) + codex must-fixes; this resolves codex's 5 open decisions.

## RESOLVED DECISIONS
- **D1 Cross-registry:** rip `vram_class` + `vram_estimate_mb` from ALL three registries
  (video registry.py, image registry.py, audio registry.py) + the "DRAFT estimates"
  headers. Keep every OTHER capability field (roles, required_inputs, required_toolchain,
  accepts_still, render_aspect, family, etc.).
- **D2 Ceiling = FULL rip (no seatbelt):** remove `motion_common.dynamic_vram_ceiling_mb`,
  `VRAM_CEILING_MB`, `assert_vram_within_ceiling`, `VramPeakProbe`, `assert_peak_within_ceiling`,
  and EVERY call site: eng_ltx_video.py:1158,1221; eng_humo.py:392; eng_ltx_av.py:502-508;
  eng_wan_i2v.py:315-318; eng_wan_ti2v.py:482-485; render_driver.py:2421-2446,2502,2527.
  Remove the validator's `OTR_VRAM_CEILING_MB` export (otr_workflow_validator.py:317-334)
  + the host-fit tier suggestion (:290-315). KEEP `vram_used_mb`/`gpu_residency` (telemetry).
- **D3 Profiles as plain pick maps:** rip `max_model_class` + `vram_budget_mb` (both keys)
  from the 3 profiles + `_TOP_LEVEL_KEYS` (capability_profiles.py:75,78); rip `_fit_reason`
  vram lines (:294-297), `VRAM_CLASS_RANK` (:62 + __all__ :46), `REASON_CLASS_OVER_CAP`/
  `REASON_VRAM_OVER_BUDGET` (:256-257), `_DECL_KEYS["vram_class"|"vram_estimate_mb"]`
  (:242-243). KEEP requires_cuda/required_toolchain/requires_sidecar/cpu_ok + registry-
  existence validation (NOT vram -- these are the CUDA/toolchain gates that stay).
- **D4 vram_ceiling_gb widget:** full removal on OTR_LedgerFreezeCascade -- INPUT_TYPES
  (:214-228) + run kwarg (:302,404) + orchestrator kwarg/warn block (_otr_freeze_cascade.py:
  670,722-744) + node 62 JSON widgets_values (7->6, drop the index-2 `14`) + the single
  `vram_ceiling_gb` inputs[] socket on node 62. Delete the now-orphan LFC watchdog ceiling
  bits (_otr_lfc_watchdog.py VRAM_DEFAULT_CEILING_GB:55 + vram_over_ceiling:226-242 if no
  other caller).
- **D5 vram_tier_label + labels:** delete `vram_tier_label` (registry.py:390-406) + the
  call in otr_video_director `_label_for` (:102) + docstring (:93-94). Display-only, no JSON
  (saved values are bare ids).
- **D6 eng_humo.safe_render_frames:** KEEP. It is the 14B's MEASURED max-frames spec
  (bakeoff-proven, env-overridable OTR_HUMO_14B_SAFE_FRAMES), an engine operating parameter,
  NOT a VRAM tier/ceiling. Removing it makes the 14B OOM on normal use. [operator: flag if
  you want it gone too -- recommend keep.]
- **D7 _otr_model_catalog LLM vram:** EXCLUDE from this rip. It is the story-writer LLM
  model-selection guardrail (a different subsystem), not the render workflow's VRAM tiers.
  [flag; separate cleanup if wanted.]
- **D8 frame-cost predictor + _vram_log:** VERIFY-AT-BUILD -- if `motion_common`'s
  frame-cost model GATES/refuses a render (a soft ceiling) rip that gate; if it only
  LOGS/predicts, keep as telemetry. `_vram_log.py` stays as telemetry (scrub "ceiling"
  wording; no policy authority).

## RIP ORDER (chunk = commit+push, green each)
1. **Labels (D5):** trivial, display-only. Tests: test_still_aspect_and_labels.py:118-136,
   test_cloud_video_adapters.py:43, test_cloud_image_adapters.py:31.
2. **Widget (D4):** node+JSON same-change (BUG-LOCAL-097, mid-list index-2). Tests:
   test_workflow_json_guardrails.py:1047-1102. Re-validate workflow (round-trip + widget
   audit + OTR_WorkflowValidator).
3. **Registry fields + profile filtering (D1+D3):** all three registries lose the two keys;
   capability_profiles loses the vram schema/rank/reason/fit lines; 3 profiles lose the two
   keys (+ _TOP_LEVEL_KEYS same commit -- fail-closed unknown/missing key coupling). Tests:
   test_capability_profiles.py:95-114,226-268 (DELETE test_vram_budget_excludes_over_budget_
   engine), test_wan_capability_row.py:20-44, test_still_word.py:44, test_word_razzle.py:25,
   test_video_mesh_stage.py:328, test_video_ltx_av.py:50-51. KEEP green: the requires_cuda /
   cold-import / two-heavy / set(CAPABILITIES)==set(all_engine_names) invariants.
4. **Runtime ceiling (D2):** motion_common ceiling fns + all engine call sites + validator
   export/suggestion. Tests: test_capability_profiles.py:295-314 (DELETE the two seatbelt
   tests), test_video_render_driver.py:81, test_otr_workflow_validator.py:219-274 (export/
   suggestion tests), any engine assert tests. VERIFY no engine path references a removed
   symbol (AST/import scan).
5. Full suite + Bug Bible + B7 + apply_profile identity check + a live smoke that a heavy
   engine still renders (now with NO ceiling assert -- proves the reclaim path alone works).

## LANDMINES (from the panels)
- L2 fail-closed key coupling: profile JSON + _TOP_LEVEL_KEYS + _DECL_KEYS + all 3 registry
  tables MUST move in the SAME commit per chunk or the validator hard-crashes.
- L3 do NOT remove requires_cuda / required_toolchain / cpu_ok -- those keep GPU engines out
  of cpu_floor + the cu128 lanes dark. They are NOT vram tiers.
- L4 node 62 widget is mid-list -> the guardrail drift test catches any code/JSON lag.
- Keep reclaim_idle_models / free_after_use (BUG-291) -- unrelated to the ceiling.

## Invariants: NO back-compat shim; node/JSON same-change; audio spine untouched; suite +
Bug Bible + B7 green + push per chunk; the ONLY "protection" post-rip is the tier JSONs.
