# STEP 2 -- CONVERGED build spec (Option A). Roundtable R1, unanimous.

Panel: GPT-5.5 (gpt-5.5-20260423), Gemini-3.1-pro (gemini-3.1-pro-preview-20260219),
DeepSeek-v4-pro (deepseek-v4-pro-20260423) + Claude code-grounded anchor + judge.
R1 spend ~$0.3293. Converged R1 (narrow single-decision question, no dissent, no
new material vs the anchor -> stop per CLAUDE.md S8).

## DECISION
Adopt **Option A**. CUT the literal `cue_type`/`archetype` migration (Option B);
do NOT skip (Option C). Unanimous across all three panelists + anchor.

- Option B rejected: `cue_type`/`archetype` exist nowhere (no producer/consumer);
  cast rows have no role field; `music_*`/`sfx` are line-row `speaker_role` values
  consumed by NON_VOICED_ROLES / `_otr_speaker_role.is_music_role` /
  `is_never_humo_role`; migrating them breaks the FROZEN `{cast,lines,meta}` schema
  and the "speaker_role is the ONLY role source" invariant.
- Option C rejected: leaves the STEP-1 interaction (`_render_cast_contract_table`
  renders `role=''` for every cast member) -> spurious `role_mismatch` -> fails the
  "0 cast-contract violations" acceptance criterion.

## BUILD (prompt-boundary only; ZERO schema / JSON / wire change)

### Change 1 -- derive a real cast-member role for the auditor
`nodes/_otr_ledger_reviewer.py::_render_cast_contract_table`. Today (post STEP 1):
`role = row.get("speaker_role") or ""` -> `''` for every cast member. Replace with:
explicit speaker_role wins, else derive: `announcer` if
`name.strip().upper() == "ANNOUNCER"` else `character`. Never `music_*`/`sfx` (cue
rows are not cast members). This is the STEP 2 substitute for the migration -- it
gives the LLM auditor a real contract role to compare a line's speaker_role against.

### Change 2 -- audit spoken rows only
`nodes/_otr_ledger_reviewer.py::_render_lines_for_audit`. Today it renders ALL
lines incl. cue rows. Filter to spoken rows via the existing
`_otr_ledger_scrub.is_spoken_role` (character/announcer). Cue rows (music/sfx) have
no cast entry and cannot drift against the cast contract; excluding them prevents
spurious `speaker_unknown` / `role_mismatch`. Keep the function's existing render
format (the doctor-vs-auditor distinctness test must stay green).

### Change 3 (GPT MUST-FIX) -- make acceptance measurable
The acceptance "0 cast-contract violations" must be observable: the re-soak read
counts `role_mismatch` / `speaker_unknown` flags in the reviewer log. No code gate
needed beyond Changes 1-2; just assert in the re-soak step.

## Tests (unit, deterministic)
1. `_render_cast_contract_table`: an ANNOUNCER row (no speaker_role) renders
   `role='announcer'`; a plain character row renders `role='character'`; an explicit
   speaker_role on a row wins; never emits an engine name (STEP-1 regression guard,
   already added).
2. `_render_lines_for_audit`: a ledger with a music_open + sfx line + 2 character
   lines renders ONLY the 2 character lines; cue line_ids absent.
3. No-drift guard: `_build_cast_rows` cast-row keys + `set_lines`/`init_lines_from_outline`
   line-row keys unchanged (schema-frozen assertion).

## Invariants honored
No workflow-JSON/node/widget change. Ledger schema FROZEN (audio byte-identical).
speaker_role is the ONLY role source (Change 1 derives the canonical spoken role for
members that structurally ARE character/announcer; nothing is moved out of
speaker_role). Fail-closed. 100% local. Suite + Bug Bible green per chunk.
