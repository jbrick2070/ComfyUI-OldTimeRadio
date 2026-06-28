# Kibitz: pre-existing RED baseline on 6 workflow/profile/B7 tests (story-quality build blocker)

## Situation
Starting the story-quality G1 build (kibitz-runs/2026-06-28-story-quality/final.md). The
FIRST commit (C1: additive leaf helpers in nodes/_otr_line_hygiene.py + a new golden test +
fixture) is done and is PROVABLY regression-free: with C1 stashed, HEAD 886c31ce
(branch v2.0-alpha) ALREADY fails the same 6 tests. So these 6 reds pre-date the
story-quality work and block the build's "full suite green per chunk" rule.

The 6 failing nodeids (all workflow-JSON / capability-profile / forbidden-sweep -- NOT story
content):
- tests/test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean
- tests/test_capability_profiles.py::test_16gb_profile_extracted_from_master_values
- tests/test_full_workflow_v2_audio_wiring.py::test_force_input_sockets_have_no_widget_key
- tests/test_workflow_apply.py::test_apply_profile_to_workflow_headless_seam
- tests/test_workflow_apply.py::test_identity_apply_16gb_profile_is_a_noop
- tests/test_workflow_live_passes_validator.py::test_production_workflow_visual_structure_pinned

The suite's KNOWN-FAIL-GUARD reports them as NEW (not in EXPECTED_FAILED_NODEIDS). The
canonical workflow `workflows/otr_scifi_16gb_full.json` on disk matches HEAD (NOT locally
modified), so this is a committed state, not local dirt. Recent commits before HEAD were
route-a feat work touching the workflow (radio-face bookend / looping credits music:
0439b1ce, cee68422, bfce4f40) then docs-only handoff (886c31ce).

CLAUDE.md sec 0: the workflow JSON is the operator-gated source of truth; node/wiring/widget
changes are hard-gated.

## Ask Codex (read the REAL repo)
1. ROOT-CAUSE each of the 6 failures: read the test + the workflow JSON + the
   capability-profile/applier code. Is each a GENUINE regression (the route-a workflow edits
   broke a structural invariant -- e.g. a force-input socket that still carries a widget key,
   a 16gb-profile value that drifted from master, a visual-structure pin), or an ENVIRONMENTAL
   flake (e.g. test_b7 needs the local s29-clean-slate-gate git ref; a missing model file)?
   Group them by root cause.
2. For each genuine-regression group, name the EXACT node id(s) / widget(s) / link(s) in
   otr_scifi_16gb_full.json at fault and the minimal fix -- WITHOUT shifting any positional
   widgets_values (BUG-LOCAL-097).
3. Decision: can the story-quality content build (CPU/content only, NO workflow JSON change)
   safely proceed and commit ON TOP of these 6 pre-existing reds (green-relative-to-baseline),
   or must they be fixed first? Flag any risk that a story-quality commit would be wrongly
   blamed for these reds later.
