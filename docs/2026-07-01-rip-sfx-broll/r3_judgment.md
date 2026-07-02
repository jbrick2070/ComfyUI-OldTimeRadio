# r3 judgment (Claude driver) -- panel: anchor + codex + claude-code (agy TIMEOUT/credits, dropped per operator)

## Accepted (grounded against real files)
- codex MF1: node 3 sfx_audio_clips is input slot 2; script_json slot 3 carries link 2
  [2,62,1,3,3,"STRING"]. Removing the sfx inputs shifts script_json to slot 2 -> link 2
  dst_slot MUST be rewritten 3->2. CONFIRMED by JSON read. Plan's "links[] untouched" was
  WRONG for node 3 (still right for node 87: gate_in slot 0 precedes the removals).
- codex MF2: cross_validate_profile (capability_profiles.py:331-336) rejects unmapped
  override keys -> stale producers break. CONFIRMED: scripts/_otr_combo_soak.py:87
  (build_all_five_role_profile), scripts/_otr_overnight_420_soak.py:149-151 (sets
  scene_broll_visual/background_abstract_visual). Both updated in the same commit.
- codex MF3: tests/test_fixture_dur_s_audit.py:20 imports SFX_DUR_MIN_S/MAX_S. CONFIRMED.
  Deleted with the G7 rip (its sole purpose = sfx dur_s fixture audit).
- codex MF4 + CUT1: scripts/_otr_patch_pool_default.py patches the removed pooling widgets.
  CONFIRMED. DELETE the script (no tombstone shim -- cleanbreak).
- codex SF2: otr_coverage_sweep.py:87-91 SLOTS maps other_beats_visual->character_video.
  CONFIRMED. Made explicit: character lane sweeps role_overrides.character_visual.
- codex SF3: extend the post-edit workflow audit to check link dst_slot vs inputs[] order
  (slot semantics), not just referential ids. FOLDED into the validation step.
- codex CUT2: NO compat aliases (FIVE_ROLES/build_all_five_role_profile renamed outright;
  consumers fixed: _otr_combo_soak.py, test_slot_matrix_soak.py).
- claude MF1/MF2: resolve_speaker_role + stamp_default_role caller audit -- RESOLVED by
  driver grep this session: zero non-test callers of either (hits only in
  _otr_speaker_role.py + tests). Raise-on-missing scope documented in docstrings + commit.
- claude MF3: role_compat.py explicit in the change map -- delete the scene_broll +
  background_abstract ROLE_AVAILABLE_INPUTS entries; docstring "five"->"three". FOLDED.
- claude MF4: meta_brief role lookup mechanical fix -- explicit membership check + raise
  (never a bare .get() without default -> None). FOLDED verbatim.
- claude MF5: node-87 widgets_values applied as a WHOLESALE 15-entry array replacement,
  never sequential deletions. FOLDED (matches driver's intended edit).
- claude SF6: build_all_role_profile defensively pops scene_broll_visual +
  background_abstract_visual. FOLDED (one-liner).
- claude SF8: legacy adapter :527 -- RESOLVED pre-commit by driver grep: docstring-only,
  no body read of beat.sfx_cue.
- claude SF9: default_engine_for_role over the 3 roles becomes a GUARD-TEST assertion
  (pre-commit gate), not verify-at-build.
- claude SF10: guard test also asserts other_beats_image_model still present in
  OTR_VideoDirector INPUT_TYPES (kept-widget assertion).

## Rejected (with reason)
- claude OPTIONAL (tombstone scene_broll_visual/background_abstract_visual in
  FORBIDDEN_INPUT_SOCKETS): wrong mechanism -- those are PROFILE keys, not input sockets;
  cross_validate_profile already fails loud on unmapped keys (capability_profiles.py:334).
- claude ASSUMPTION on freeze raise-vs-log: RESOLVED -- per-line invariants append to
  report.errors; phase_10_gap_audit_post_and_freeze RAISES on errors ("Phase 0 collect /
  Phase 10 raise"). Loud confirmed; plus resolve/sequencer/shot_lock raises fire earlier.
- codex SF2 tail ("keep other_beats_visual as a migration-slot test if still required"):
  kept as-is -- the legacy slot itself stays under test via test_slot_matrix guard updates.

## Verify-at-build (honest residue)
- OTR_WorkflowValidator's existing link audit may or may not check dst_slot-vs-input-order;
  the driver's post-edit audit script checks it explicitly either way.
- Engine default_roles trace for the 3 surviving roles (guard test will pin it).
