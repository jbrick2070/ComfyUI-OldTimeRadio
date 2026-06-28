# Kibitz judgment -- the 6 pre-existing RED tests at HEAD 886c31ce (v2.0-alpha)

Panel: Codex `gpt-5.5`@high (kibitz fan-out) + Antigravity `gemini-3.5-pro` (run MANUALLY by
operator, pasted back). Claude = code-grounded judge. The two agents DISAGREED on 2 of 4
groups; grounded against the real files below, the agreement is:

## ONE dominant root cause: ComfyUI Desktop UI-save pollution of otr_scifi_16gb_full.json
A UI-save during the route-a work serialized the canonical workflow in the polluted Desktop
form. GROUNDED in the live JSON:
- Nodes 80,81,82,83,87 have NORMAL widgets serialized as converted-INPUTS carrying a
  `"widget"` key (node 80: voice_bank/cast_voice_policy/delivery_profile/allow_voice_reuse;
  81-83: engine/stereo_policy; 87: 16 widgets). cast_lock.py INPUT_TYPES shows only
  script_json + gate_in are `forceInput`; the rest are plain widgets that belong in
  widgets_values, NOT inputs[].
- Node 87 widgets_values store DISPLAY LABELS ('humo_14B_169 (16:9)', 'visualizer (16:9)')
  not the BARE engine id. otr_video_director.py L46-48 + `_engine_id_from_pick` are explicit:
  **the SAVED value must stay the bare engine id**; the label is display-only. So the labels
  in widgets_values are the drift.

### Resolving the agent disagreement (grounded)
- `test_force_input_sockets_have_no_widget_key`: ANTIGRAVITY RIGHT (genuine JSON pollution),
  Codex's "test bug" reading REJECTED -- the 4 node-80 inputs are plain widgets wrongly
  converted; the canonical clean form has ZERO widget-keyed inputs on these nodes, so the
  test's all-inputs assertion is correct, not over-broad.
- profile/structure tests (`test_16gb_profile_extracted_from_master_values`,
  `test_production_workflow_visual_structure_pinned`, `test_*_apply_*_16gb_*`): CODEX RIGHT
  (canonical persisted form = BARE id; node 87's labels are the bug). Antigravity's "update
  the profile to the (16:9) label" REJECTED -- it would entrench the UI-save drift. The
  profile correctly stores bare `humo_14B_169`; the FIX is to normalize node 87 widgets_values
  back to bare ids, NOT to relabel the profile.

### Correct fix (workflow JSON cleanup -- operator-gated, CLAUDE.md sec 0)
Restore the canonical form of otr_scifi_16gb_full.json: (a) drop the widget-keyed converted
inputs on nodes 80-83 and 87 (keep only true sockets: script_json/ledger_json/gate_in); (b)
normalize node-87 widgets_values labels -> bare ids ('humo_14B_169 (16:9)'->'humo_14B_169',
'visualizer (16:9)'->'visualizer'). Do NOT shift positional widgets_values (BUG-LOCAL-097).
Re-validate: OTR_WorkflowValidator + JSON round-trip + link referential integrity + widget-
count vs INPUT_TYPES. INTENT CONFIRM NEEDED: route-a intended announcer/music/character video
= humo_14B_169 (per memory + the live JSON); if so the stale `ltx_audio_in` pin in
test_workflow_live_passes_validator.py is updated to the bare humo id (a TEST edit, not JSON).

## Second, independent root cause: B7 forbidden-sweep token
Commit 57170cca added scripts/build_humo_bakeoff_workflow.py using the banned word `alias`
as a parameter/var on lines added since the s29-clean-slate-gate TAG -> `test_forbidden_
sweep_runs_clean` fails. (Codex caveat CONFIRMED: the sweep diffs `*.py` since s29, so it is
working-tree sensitive -- but b7 fails at the clean stash baseline too, so the alias token is
the real cause.) Fix: rename `alias` -> `node_alias`/`alias_name` in that script. SAFE.

## Decision: NOT safe to commit story-quality on top
conftest KNOWN-FAIL-GUARD raises SystemExit(2) on any nodeid not in EXPECTED_FAILED_NODEIDS
(currently empty) -> a hard suite abort that fails the green-per-chunk gate for EVERY commit,
AND leaves the story-quality commits wrongly blamed for the workflow reds. Per CLAUDE.md
sec 3/sec 7 the normal path is fix-first. Recommend: fix the baseline (JSON cleanup + alias
rename + the one stale pin) as its OWN commit, re-green the suite, THEN resume the
story-quality C1..C6 chain on a clean baseline.

Agent calls: codex (kibitz, OK) + antigravity (manual, OK). Disagreements judged on grounded
code evidence above.
