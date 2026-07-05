VERDICT: yes-with-fixes. Two production-wiring ambiguities and one staged-module signature mismatch still block a reliable T1 build.

MUST-FIX BEFORE BUILD:
1. [PATCH_PLAN: New modules, lines 31-33] `_otr_source_interpreter.py` says science wires `build_news_briefs` verbatim, but the staged facade calls `news_briefs_builder(article=article, **builder_kwargs)` at `transplant_work/production_new_modules/_otr_source_interpreter.py:78`; production `build_news_briefs` does not accept `article` and requires `technical_fn`, `full_text`, `headline`, `summary`, `outlet`, `pub_date`, `style`, and `seed` (`production_mirror/nodes/news_interpreter.py:731-746`). Concrete fix: either make the facade map `article` to the real keyword set before calling `build_news_briefs`, matching the current writer call shape at `production_mirror/nodes/OTR_LedgerScriptWriter.py:2873-2886`, or change the plan to require a wrapper builder and add a production-side test using the real signature.

2. [PATCH_PLAN #1, lines 44-47] “non-science banks ignore the style slug” is under-specified. Production still passes `resolved["style"]` into cast locking, outline, metadata, and visual plan (`production_mirror/nodes/OTR_LedgerScriptWriter.py:3010`, `:3178`, `:5286`, `:5415-5417`). Two builders could validly set it to empty, story_model id, pack label, or profile text and produce incompatible outputs. Concrete fix: define the exact non-science replacement value for `resolved["style"]` and its metadata stamps, then test that `style`/`style_custom` widget text is not read while downstream required style arguments remain deterministic.

3. [PATCH_PLAN #1/#9, lines 55-58 and 94-99; r4/input Applied #8, lines 31-34] Production input names and bridge socket shape are not one canonical contract. PATCH_PLAN names whitelist keys as `source_bank`, `story_model`, `story_pipeline`, `visual_style`, while lab nodes and bridge spec use `source_bank_id`, `story_model_id`, `story_pipeline_id`, `visual_style_id` (`nodes.py:198-201`, `src/upstream_story_lab/contracts.py:429-433`). PATCH_PLAN also says “policy/bridge JSON sockets” and “path/JSON socket,” while r4/input says an explicit path STRING. Concrete fix: list the exact appended writer inputs in order, including one canonical bridge input, e.g. `bridge_artifact_path` STRING forceInput; specify mapping from production widget names to artifact `*_id` fields; state that non-science banks require a non-empty artifact path and science defaults may leave it empty.

SHOULD-FIX:
1. [GO_FORWARD_PLAN: What is done, lines 20-22] Test evidence is stale/inconsistent: GO_FORWARD says “41 pytest,” while r4/input says “46 pytest green” (`kibitz-runs/.../r4/input.md:3-6`). Concrete fix: update GO_FORWARD to the current collected/pass count or remove hardcoded counts and list the exact commands that must pass.

2. [PATCH_PLAN: ATOMICITY + ORDER, lines 16-19] The `story_orchestrator._fetch_science_news` verify step exists, but make it executable: in the live production repo/venv, import `nodes.story_orchestrator`, assert `_fetch_science_news` exists, and run the writer RSS failure path once to confirm the current fail-loud message remains compatible. The mirror proves the writer imports it (`production_mirror/nodes/OTR_LedgerScriptWriter.py:1137-1158`) but `story_orchestrator.py` is not mirrored in `PRODUCTION_MIRROR_MANIFEST.md`.

OPTIONAL / NICE-TO-HAVE:
None.

CUT THESE:
1. [GO_FORWARD_PLAN: What is done] Cut hardcoded historical pytest counts from the handoff prose. They already drifted and do not add build safety beyond the actual gate commands.

VERIFY-AT-BUILD checklist:
1. Earlier UNVERIFIABLE carried into r4: `story_orchestrator._fetch_science_news` live import/failure behavior. Verify in the live production repo before writer edits land. [ASSUMPTION] This is the only unresolved earlier UNVERIFIABLE carried in the provided r4 input.
2. Re-mirror check: production HEAD must still match `d48a9d76f39db6db16c758d9b2c1c22a9af38d3f`; otherwise refresh `production_mirror/` and rerun drift tests.
3. Adapter validation against real production `NewsBriefs`, plus the fixed science facade against real `build_news_briefs` signature.
4. Writer self-test updated in the same chunk as appended widgets: current mirror asserts `n_optional == 16` at `production_mirror/nodes/OTR_LedgerScriptWriter.py:5728-5739`.
5. Whitelist parity for the chosen production widget names in `scripts/otr_api.py` and `nodes/_otr_workflow_apply.py`.
6. Workflow JSON remains last: append-only widget audit, forceInput link audit, JSON round-trip, workflow validator, and science baseline pre/post tests all green.