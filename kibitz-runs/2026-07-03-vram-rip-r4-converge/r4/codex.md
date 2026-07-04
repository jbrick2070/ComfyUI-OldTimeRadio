VERDICT: yes-with-fixes — close, but not build-ready until the chunk-order contradiction, telemetry regression, and missed ceiling probe script are resolved.

MUST-FIX BEFORE BUILD:
1. [RIP ORDER 3/4 + FABLE A1] The plan still says validator export/suggestion removal happens in chunk 4, while A1 correctly says it must move into chunk 3. This is a build-breaker because C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_workflow_validator.py:299 and :319 read profile["vram_budget_mb"]. Concrete fix: rewrite RIP ORDER so chunk 3 explicitly removes the validator vram_budget_mb host-fit branch and OTR_VRAM_CEILING_MB export, and chunk 4 explicitly excludes validator edits.

2. [D2 + CODEX R2 HARDENING] Removing VramPeakProbe conflicts with the goal to keep live VRAM measurement as telemetry. The class is a sampler in C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\motion_common.py:242, and current telemetry users include eng_ltx_av.py:979/:1006, eng_wan_i2v.py:294, and eng_wan_ti2v.py:450. Concrete fix: keep it as a renamed telemetry-only sampler, or keep VramPeakProbe but remove only assert_peak_within_ceiling and threshold enforcement/log wording.

3. [A3 / Post-rip grep gate] The plan misses C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\_otr_a_s2_probes\probe_d_vram_boundary_failclosed.py:25, :29, and :34, which still implement OTR_VRAM_CEILING_MB admission gating. This will fail the stated zero-hit grep and contradicts the full ceiling rip. Concrete fix: delete/archive this probe out of required paths, or rewrite it as a telemetry/free-VRAM probe with no admission ceiling.

SHOULD-FIX:
1. [D8 + CODEX R2 HARDENING] D8 still says VERIFY-AT-BUILD, while CODEX R2 says D8 is resolved. Concrete fix: delete the stale unresolved D8 wording and keep the resolved free-VRAM-only compute_real_frame_budget step tied to motion_common.py:370-402 and teardown at :486.

2. [A3] Add C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\otr_wan_smoke.py:208 and scripts\soak_operator.py:131 to the scrub list or mark them explicitly non-required diagnostics. They still present 14.5GB as a pass/fail ceiling.

3. [RIP ORDER 5 + CODEX R2 HARDENING] The initial order says full suite + Bug Bible only at the end, later text says after every chunk. Concrete fix: make “full suite + Bug Bible + push after each green chunk” the single gate.

OPTIONAL / NICE-TO-HAVE:
- Rename telemetry fields from “ceiling” language wherever kept, but do not churn unrelated historical docs.
- Keep the dark 3D ceiling constants as documented exemptions unless the grep gate explicitly includes them.

CUT THESE:
1. [D2 / CODEX R2 HARDENING] Cut duplicated line-number call-site lists once the final grep gate is authoritative. Keep only one source of truth plus exemptions.
2. [FABLE END-TO-END provenance] Cut audit provenance text from the build plan after folding fixes; it is useful history, not builder instruction.

VERIFY-AT-BUILD checklist:
- Confirm no profile removal KeyError: no live reference to profile["vram_budget_mb"] remains before config/profiles/*.json drops that key.
- Run OTR_WorkflowValidator, JSON round-trip, link integrity, and widget/input audit on workflows\otr_scifi_16gb_full.json.
- Confirm node 62 has vram_ceiling_gb input removed, widgets_values length 6, no links[] surgery needed.
- Grep repo and Bug Bible for vram_ceiling, VRAM_CEILING, OTR_VRAM_CEILING_MB, assert_peak_within_ceiling, assert_vram_within_ceiling, VramPeakProbe, dynamic_vram_ceiling_mb, vram_class, vram_estimate_mb, max_model_class, vram_budget_mb, vram_tier_label, --vram-ceiling, vram_over_ceiling, VRAM_DEFAULT_CEILING_GB, lfc_vram_ceiling_gb, with only documented exemptions.
- Verify compute_real_frame_budget uses live free VRAM only and update the 29-to-33 frame tests.
- Verify VRAM telemetry still records live used/peak values where available, but no threshold abort remains.
- Run full suite + Bug Bible + B7 + apply_profile identity check + live heavy-engine smoke with no ceiling assert.