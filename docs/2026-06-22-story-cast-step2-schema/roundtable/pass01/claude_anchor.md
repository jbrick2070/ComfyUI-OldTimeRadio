<!-- Claude code-grounded anchor review, R1 (arc/scope) -->
VERDICT: yes-with-fixes. Option A is the right root-cause fix; the literal STEP 2
(Option B) is a scope/invariant defect and must be cut; Option C is unsafe as-is
because it leaves the STEP-1 `role=''` interaction (fact 6) open.

MUST-FIX BEFORE BUILD:
1. [Option B] CUT the `cue_type`/`archetype` schema migration. CONFIRMED: grep of
   the real tree shows `archetype` and `cue_type` have zero occurrences; cast rows
   (`_build_cast_rows`) carry no role field; music/sfx are line-row speaker_role
   values consumed by NON_VOICED_ROLES / `_otr_speaker_role.is_music_role` /
   `is_never_humo_role`. Migrating them breaks the FROZEN schema + the plan's own
   "speaker_role is the ONLY role source" invariant. Fix = do not add the fields.
2. [Option A / fact 6] The real, must-fix defect STEP 1 leaves: the cast table
   now renders `role=''` for every member. Fix = derive the contract role in
   `_render_cast_contract_table` (announcer if name==ANNOUNCER else character) so
   the auditor compares a line's speaker_role against a real role, not ''.
3. [Option A] Audit only spoken rows. `audit_cast_contract` feeds `_render_lines_
   for_audit(lines)` ALL lines incl. cue rows. Filter to `is_spoken_role` so cue
   rows (no cast entry) cannot raise speaker_unknown/role_mismatch.

SHOULD-FIX:
1. [Option A] Guard the ANNOUNCER detection: a cast member could carry an explicit
   speaker_role already (future); prefer `row.get("speaker_role") or <derive>` so
   an explicit value always wins. Detection by name.upper()=="ANNOUNCER" matches
   the existing `_is_announcer_entry` convention in cast_lock.py (consistency).
2. Add a no-drift regression assertion that the ledger wire schema (cast row keys,
   line row keys) is unchanged by STEP 2 -- the plan asks for this.

OPTIONAL: none.

CUT THESE: the entire `cue_type`/`archetype` field set (MUST-FIX 1).

[ASSUMPTION] A cast member is always either ANNOUNCER or a character (no third
spoken cast kind). CONFIRMED against the writer: cast_names feed character rows,
ANNOUNCER is the only special-cased name; music/sfx never become cast members.
