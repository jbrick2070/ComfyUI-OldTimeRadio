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

## CODEX R2 HARDENING (folded -- these make the plan code-ready)
- **Full ceiling call-site list (D2 was incomplete):** ALSO remove VramPeakProbe /
  assert_peak_within_ceiling at eng_ltx_av.py:979,996; eng_wan_i2v.py:294,319;
  eng_wan_ti2v.py:450,486 (plus the sites already listed). Convert any still-wanted peak
  value to telemetry-only logging (no raise).
- **D8 RESOLVED (no verify-at-build):** `compute_real_frame_budget`
  (motion_common.py:370-411) IS a gate -- `budget_mb = min(free_vram, ceiling)*margin`.
  Change it to clamp on LIVE FREE VRAM ONLY (drop the `ceiling` term; no policy ceiling) so
  frames fit what's actually free, telemetry-driven not tier-baked. `MotionEngineBase.teardown`
  (:476-487) waits below dynamic_vram_ceiling_mb -> replace with a reclaim/stability wait
  (or drop the threshold), no ceiling symbol.
- **wrapper_bridge separate ceiling (missed):** remove `VRAM_CEILING_MB` constant + its
  `__all__` export at wrapper_bridge.py:36-37,642-644 (or prove dead).
- **LFC watchdog (D4) -- exports + telemetry shape:** keep ONLY
  `vram_at_cascade_entry_gb` via a direct allocation read; drop the ceiling stamp/warn
  (_otr_freeze_cascade.py:727-743), `VRAM_DEFAULT_CEILING_GB` + `vram_over_ceiling`
  (_otr_lfc_watchdog.py:38-46,55,226-242), and update `__all__`.
- **Smoke/soak scripts enforce ceilings (contradiction):** remove the over-ceiling aborts
  in scripts/_otr_soak_capstone.py:66,654-659 + scripts/run_ltx_av_q_bakeoff.py:103,614-616,727
  (or archive them out of the required smoke path).
- **Validator (should-fix):** delete ONLY the `profile["vram_budget_mb"]` branch + the
  OTR_VRAM_CEILING_MB env export (otr_workflow_validator.py:297-333); KEEP the no-CUDA +
  platform-mismatch stamp checks.
- **More test breakage (fold into the chunks):** test_video_motion.py:287-298,
  test_video_motion_common_additive.py:99-104 (assert_vram_within_ceiling), test_clip_fill.py:29
  (clears OTR_VRAM_CEILING_MB env). Registry-field tests rewrite to the SURVIVING fields
  (required_toolchain / requires_sidecar / cpu_ok / model_requirements / registry-consistency):
  test_cloud_image_adapters.py:30-31, test_cloud_video_adapters.py:42-43,
  test_wan_capability_row.py:21,43-44, test_video_ltx_av.py:51, test_still_word.py:44,
  test_word_razzle.py:25, test_video_mesh_stage.py:328.
- **Dark 3D engines:** eng_character_3d.py:74,257,324,393 + eng_triposr.py:52,121 carry hard
  3D VRAM ceilings but are UNREGISTERED dark scaffolds (no CAPABILITIES rows; pinned
  unregistered by tests). Document as unreachable; optional tidy, not blocking.
- **Per-chunk gate (repo rule):** run full suite + Bug Bible after EACH chunk (not only the
  final step) + push per green chunk.
- **Post-rip grep gate:** repo-wide zero-hit check for `vram_ceiling`, `VRAM_CEILING`,
  `OTR_VRAM_CEILING_MB`, `assert_peak_within_ceiling`, `assert_vram_within_ceiling`,
  `VramPeakProbe`, `dynamic_vram_ceiling_mb`, `vram_class`, `vram_estimate_mb`,
  `max_model_class`, `vram_budget_mb`, `vram_tier_label` (outside the LLM catalog + telemetry).

## FABLE END-TO-END (6-subagent fan-out) -- FINAL AMENDMENTS (make it truly build-ready)
- **A1 CHUNK-ORDER BUILD-BREAKER (was NO-GO):** OTR_WorkflowValidator reads
  `int(profile["vram_budget_mb"])` at otr_workflow_validator.py:299 AND :319 on EVERY
  stamped prompt. Removing vram_budget_mb from profiles in chunk 3 while the validator edit
  sits in chunk 4 -> every production run KeyErrors after chunk 3. FIX: MERGE the validator
  branch (:297-305) + export (:317-333) deletion INTO chunk 3 (same commit as the profile
  key removal). Moves test_otr_workflow_validator.py:223,230,270,274 into chunk 3.
- **A2 D8 arithmetic -> 2 missed tests:** dropping `min(free,ceiling)` at motion_common.py:402
  changes budget math (14775*0.85 -> (12558.75-7000)/185 = 30 -> 4n+1 UP-snap
  (wrapper_bridge.py:394) = 33, not 29). test_clip_fill.py:41-45 (assert ==29 @:45) +
  test_wan_ti2v.py:229-236 (assert ==29 @:236) hard-pin 29 -> update to 33. Sole prod caller
  eng_wan_ti2v.py:328-331 clamps, never aborts -- survives. No caller passes ceiling_mb= ->
  dropping the kwarg is signature-safe.
- **A3 script + cosmetic grep-gate touchpoints:** scripts/otr_video_gpu_smoke.py:87,107,109,
  124,170-173,232,240 (full vram_ceiling plumbing incl. the HYPHENATED `--vram-ceiling` flag
  -- invisible to a plain grep; report-only, scrub to vram_used_mb telemetry); scripts/
  smoke_check.py:30 (dead `VRAM_CEILING=14.5`, delete); wrapper_bridge.py:643 is a MIXED
  __all__ line -- keep "PIX_FMT","COLOR_PRIMARIES"; motion_common.py:507,513 __all__ (delete);
  render_driver.py:2494-2495 (`run_gpu_soak(...vram_ceiling_mb=None)` kwarg) + :1881-1882
  comment. GREP THE BUG BIBLE REPO too (comfyui-custom-node-survival-guide -- outside this tree).
- **A4 KEEP-set adjacency traps (delete asserts ONLY, not the neighbours):**
  (1) motion_common.py:486 teardown calls dynamic_vram_ceiling_mb inside the lease settle-wait
  (KEEP the wait, swap to a reclaim/stability threshold-agnostic wait -- gpu_residency.
  wait_until_below_mb stays). (2) eng_ltx_av.py:502-508 assert_usable NVML gate is KEEP --
  only the ceiling interpolation in its error MESSAGE (:508) goes. (3) reclaim_idle_models
  sits beside the asserts at eng_humo.py:387 vs :391-392 + eng_ltx_av.py:992 vs :996 -- delete
  the assert lines only. (4) vram_used_mb telemetry (eng_wan_i2v:315 / eng_wan_ti2v:482) feeds
  the same log lines that interpolate the ceiling (:318/:485) -- line-precise edits.
  (5) wrapper_bridge.VRAM_CEILING_MB:37 is a DEAD duplicate (zero consumers); reclaim/
  free_after_use read no ripped symbol. Verified: NO `from X import <ripped-name>` anywhere
  (all module-attribute access), no __init__ re-export, no string/monkeypatch access.
- **Node 62 confirmed:** widgets_values `[true,true,14,"all",6,false,""]`, vram_ceiling_gb is
  index 2, exactly 3 string hits all in the ONE widget-input entry (link:null) -> NO links[]
  surgery; delete inputs entry + widgets_values[2] together. Guardrail test
  test_workflow_json_guardrails.py:1047-1083 goes len 7->6; the :1147-1199 order guard is the
  tripwire. Archived __ORIGINAL_b001 snapshot has no vram_ceiling_gb -- untouched.
- **Grep-gate additions:** `--vram-ceiling`, `vram_within_ceiling`, `VRAM_CEILING` (bare),
  `vram_over_ceiling`, `VRAM_DEFAULT_CEILING_GB`, `lfc_vram_ceiling_gb`. Documented exemptions:
  _vram_log.py (telemetry, scrubbed), _otr_model_catalog.py + test_vram_envelope_c4.py (D7),
  eng_character_3d/eng_triposr `_VRAM_CEILING_MB_3D` (dark unregistered), docs/kibitz artifacts,
  frozen m0_object_info snapshot. FYI (not this pass): otr_image_gen_dispatcher.py:619 has a
  hard-baked wait_until_below_mb(15000) -- KEEP-class, log-only.

**VERDICT: GO** (Fable end-to-end + codex R2, both grounded) once A1-A4 are folded. A1 is
the one that would have broken production; it is now merged into chunk 3.

## Invariants: NO back-compat shim; node/JSON same-change; audio spine untouched; suite +
Bug Bible + B7 green + push per chunk; the ONLY "protection" post-rip is the tier JSONs.
