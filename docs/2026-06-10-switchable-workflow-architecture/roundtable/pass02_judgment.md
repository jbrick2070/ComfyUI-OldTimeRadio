# Pass02 judgment log (judge: claude-fable-5)
Panel: gpt-5.5, gemini-3.1-pro, grok-4.3, deepseek-v4-pro on pass01_plan.md. Spend $0.2486 (manifest). All ok.
Pass02 found real internal inconsistencies in the pass01 synthesis. All grounded; all resolved below.

## ACCEPTED (verified against the repo)
1. **Stamp-in-properties is not executable; node-63 widget vector would break; wrapper_bridge cannot see the graph; assertion had no defined home/ordering** (GPT#1/#2/#5, Gemini#2/#3, Grok#2/#3, DeepSeek#1/#3/#8 -- 4/4 consensus). VERIFIED: validator INPUT_TYPES = exactly 3 widgets (lines 183-189); API prompts carry inputs only. RESOLVED design:
   - Stamp = three NEW optional STRING widgets on `OTR_WorkflowValidator` (`profile_id`, `master_hash`, `generated_by`), INPUT_TYPES + widget-vector tests updated in the same change.
   - Stamp written ONLY by `emit_snapshot()`; `apply_profile()` never stamps. Master stays UNSTAMPED and remains the dev/reference artifact; users run snapshots; assertion active only when stamp non-empty (DeepSeek#8/#9).
   - `wrapper_bridge` reads `OTR_VRAM_CEILING_MB` env > 14500 ONLY (DeepSeek#1). UI path: the stamped validator node exports the resolved budget into `os.environ` when it executes (Gemini#3); ordering guaranteed by wiring `validation_report` into a new optional input on node 92 (GPT#5) -- one-time master edit. Headless/launcher set the env directly. Known accepted gap: independent audio nodes may execute before an assertion failure aborts the prompt -- LOUD + cheap, noted in plan.
2. **Node-63 relative paths resolve against CWD** (GPT#3, Gemini#4, Grok#1). VERIFIED at `_otr_workflow_validator.py:69` (`p = Path(path)`; only empty-string falls back). Fix: resolve non-absolute against `_REPO_ROOT`.
3. **Byte-identical identity check contradicts stamping/path injection** (GPT#4 + CUT1, Gemini CUT1, DeepSeek#2/#5, Grok S2). RESOLVED: split `apply_profile` (semantic) vs `emit_snapshot` (path+stamp); identity gate becomes `to_api_prompt(master) == to_api_prompt(apply_profile(master, 16gb_full))` dict-equality (UI state excluded automatically); parity gate ignores node-63 path + stamp fields; BOOTSTRAP: the initial 16gb profile is EXTRACTED from the master's current values, not imposed.
4. **Schema-shape mismatch: INPUT_TYPES() tuples vs /object_info lists** (Gemini#1). The shared module gets an explicit INPUT_TYPES->object_info adapter; live /object_info cross-check stays in the soak lane.
5. **Widget coverage incomplete + hard-coded node ids brittle + widget names unverified** (GPT#7/S9, DeepSeek#4/#6/#10). VERIFIED: node 1 = OTR_LedgerScriptWriter (model slots), node 3 = OTR_SceneSequencer (carries "bark"), node 88 = OTR_ImageDirector. RESOLVED: a schema-driven COVERAGE TEST enumerates every COMBO/STRING widget whose value is a registered engine id; each must be profile-managed (keyed `(node_type, widget_name)` + unique-match assertion) or exempted with a written reason. Widget-name mapping doc (profile key -> node_type.widget_name from real INPUT_TYPES) is an S0 deliverable.
6. **FORCE_ENGINE_MAP precedence contradiction** (GPT#8). RESOLVED: force = dev bypass of the profile enable-set, NEVER of registry reality; LOUD warning when forcing outside the profile; tests for parse-error + outside-profile cases.
7. **Static "two heavies" rejection ill-defined** (GPT#6 + CUT2). VERIFIED: master legitimately saves multiple heavy choices across roles; residency is wrapper_bridge lifecycle. RESOLVED: profile validator checks per-engine fit only.
8. **Whitelist must be enumerated + queue_smoke migrated in the same sprint** (Grok#4, Gemini S1). VERIFIED queue_smoke patches: slot pickers loop + target_words/num_characters/act_count. Whitelist fixed: `target_words, num_characters, act_count, request_seed + seed-policy fields, prompt/title text fields, openrouter_slot_*/comfy_slot_*` (reusing `_is_openrouter_admissible`/`_is_comfy_admissible` -- GPT S4).
9. **Sequencing** (GPT S2, Grok S3): S0 = shape-only validation + mapping doc; capability cross-validation lands S1; parity gate moves to S3 (needs emit_snapshot); cold-import becomes a BLOCKING S1 gate (GPT S5/S8, Grok S1) since the generator relies on direct NODE_CLASS_MAPPINGS import; registry capability metadata must be dep-free declarations (GPT S6).
10. **Profiles carry overrides only** (Grok CUT2): registry `default_engine_for_role` supplies the base; role/slot fields become optional overrides.
11. **allow_sidecars / max_model_class semantics defined** (GPT S3): sidecar-requiring engines excluded when false; vram_class capped by max_model_class. Both kept.
12. **Determinism normalization defined; per-commit gate cut** (GPT S7 + Grok CUT1 reconciled): tier-SHIP gate only; audio reuses the byte-identical machinery; video compared by stream hash with container metadata stripped; ledger fields compared.

## RESOLVED CONFLICTS (judge call)
- **Applier guard vs grep test** (GPT CUT4 wants guard-only; DeepSeek#7/CUT wants grep-only): BOTH, each in its stateless form -- `patch_creative()` is a pure function validating names against the whitelist (no state, so DeepSeek's fragility objection dissolves), plus one cheap regression test asserting soak scripts contain no direct `patch_widget_by_name` calls on profile-managed names.
- **Launcher emission timing** (GPT CUT5): stays in S5 (already after core); headless sets env itself; UI ceiling comes from the stamped validator export, so nothing blocks on the launcher.
- **Grok OPT profile_id mirror widget on node 1**: REJECTED -- one stamp home; the ledger carries profile identity for support.

## REJECTED
- None outright beyond the above; no panel item proposed breaking an invariant this pass.

## Convergence call
Architecture direction unchallenged 2 passes in a row (master + generated tiers + profile layer). Pass02 must-fixes were mechanics of the pass01 synthesis; all are now resolved with concrete, verified designs in pass02_plan.md. Run pass03 on pass02_plan.md as the confirmation pass; converge if it yields no NEW must-fix.
