<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The zero-workflow-edit conclusion may be right, but the document has an internal widget-index contradiction and leaves acknowledged downstream beat/meta safety questions unresolved.

MUST-FIX BEFORE BUILD:
1. [Node 1 current widget map] Widget count/indexing is internally inconsistent. The section says “24 entries” and “append point is index 24”, but the listed entries are `[0]` through `[22]`, i.e. 23 entries. Later, [IF a widget append becomes necessary] says “new index 23/24...”, which contradicts the stated append point. Concrete fix: audit `workflows\otr_scifi_16gb_full.json` and `OTR_LedgerScriptWriter.INPUT_TYPES`; either add the missing `[23]` entry to the map or change the count to 23 and the append point to index 23. Use one unambiguous append index throughout.

2. [IF a widget append becomes necessary] The append procedure is unsafe as written because it does not require preserving the live `INPUT_TYPES` widget order relative to `widgets_values`. It only says “Add the matching optional widget,” but positional widget loading depends on the node’s declared widget order. [ASSUMPTION] Concrete fix: state that any new optional widget must be appended at the end of `OTR_LedgerScriptWriter.INPUT_TYPES` in the same order as the appended `widgets_values` slot, and that no existing widget declaration may be reordered, renamed, moved between required/optional, or deleted in that commit.

3. [Open questions for the panel #4] The plan declares “ZERO edits” before resolving whether downstream render nodes read `meta.*` fields or rely on `lines[]` ordering. This is not a harmless open question: if downstream nodes consume those fields structurally, F2/F3/F8 can become workflow-adjacent even without adding widgets. Concrete fix: before build, audit every consumer of node 1 outputs `script_text`, `script_json`, `news_used`, `estimated_minutes`, and `technical_model`; specifically verify all reads of `meta`, `lines`, `arc_phases`, `announcer_beats`, and any beat/slot identifiers. Document “no consumer dependency found” or list required compatibility constraints.

4. [Open questions for the panel #5] F8 may change `arc_phases`, `announcer_beats`, `music_inter_count`, beat count, and slot IDs, but the plan still classifies F8 auto-pick as “NO JSON change.” If downstream workflow nodes assume a fixed beat/slot shape, this can break render behavior without a widget change. Concrete fix: decide one of these before build: either constrain F8 v1 to preserve the existing beat-count/slot-id contract exactly, or update the consumer contract/tests and mark F8 as wiring-adjacent.

5. [Per-fix wiring impact / F3 ending-aware outro] F3 is described as “prompt text + reads existing `meta`,” but the plan does not identify which `meta` keys are existing, where they are produced, or what fallback happens if absent. Concrete fix: name the exact existing `meta` fields F3 will read, verify they are already present in node 1 output before F3 runs, and require a no-op/fallback path when missing.

SHOULD-FIX:
1. [Architecture fact] The claim that the “ENTIRE story pipeline runs inside ONE node” and that all other graph nodes are render-side is the foundation of the plan, but no verification step is attached. Concrete fix: add a pre-build check: inspect the workflow JSON for all links out of node 1 and all inputs into story/render nodes; verify no other node has story-generation inputs or story-control widgets affected by F1-F10.

2. [Per-fix wiring impact / F5 speech-register] The row says speech register “surfaces via existing `all_voice_cards`,” but the plan does not verify that `all_voice_cards` is already present in node 1 output or consumed only internally. Concrete fix: verify whether `all_voice_cards` is internal-only or serialized into `script_json`; if serialized, audit downstream consumers for schema assumptions.

3. [Per-fix wiring impact / F10 anti-repeat list] “Always-on with a constant window + a local JSON file” introduces a runtime dependency not covered by the workflow plan. Concrete fix: specify file path, schema, creation behavior, corruption handling, concurrency behavior, and whether the file is project-local, workflow-local, or global. If this state affects reproducibility, add a reset/disable mechanism even if not exposed as a widget.

4. [IF a widget append becomes necessary] The validation step is underspecified. “OTR_WorkflowValidator + JSON round-trip” is not enough unless the exact pass/fail criteria are known. Concrete fix: define the concrete command or script and require checks for: node 1 widget count matches live widget declarations, all saved widget values deserialize without positional drift, all links reference existing nodes/slots, and node 1 output slot count/type matches existing links.

5. [Net recommendation] The document mixes v1 implementation guidance with hypothetical future widget append procedure. This makes the build decision less clear. Concrete fix: split the plan into “v1 zero-JSON implementation” and “future widget migration appendix,” with v1 explicitly forbidding widget changes unless the plan is reopened.

OPTIONAL / NICE-TO-HAVE:
- Replace brittle JSON line references with node id/title/path references plus a generated widget audit snippet.
- Add a small compatibility test fixture using the current `workflows\otr_scifi_16gb_full.json` and a minimal node 1 output payload.
- Add a schema/version note for additive `script_json.meta` keys so render consumers can ignore unknown keys safely.

CUT THESE (over-engineering):
1. [IF a widget append becomes necessary] Cut the exposed-widget branch from the v1 build plan if F8/F10 are accepted as internal/auto. It is safe to move to an appendix because v1’s stated recommendation is zero workflow edits, and keeping hypothetical append instructions in the main path increases the chance someone edits `widgets_values` unnecessarily.

2. [IF a widget append becomes necessary] Cut “commit+push to `v2.0-alpha` with the code in the same commit” from the technical wiring spec. Branch/commit policy does not validate workflow correctness and can live in release process docs.

3. [IF a widget append becomes necessary] Cut “one coder window” from the build spec. It is process advice, not a wiring requirement; locking the append index and validating the workflow are the actual safety controls.