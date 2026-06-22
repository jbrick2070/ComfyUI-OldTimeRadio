<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Build Option A, not B or C; but A needs an explicit acceptance/measurement fix and a prompt-contract cleanup before it is safe to call STEP 2 complete.

MUST-FIX BEFORE BUILD:
1. [STEP 2 as literally written / Option B / Grounded facts 1-4] Literal `cue_type` / `archetype` migration is architecturally invalid. It invents fields absent from the codebase, moves load-bearing line-row `speaker_role` values out of the frozen role taxonomy, and contradicts “speaker_role is the ONLY role source.” Concrete fix: replace STEP 2 text with Option A’s scope: no schema migration, no new fields, no clearing `speaker_role` on cue rows; keep `music_open`, `music_close`, `music_inter`, `sfx` as valid `lines[].speaker_role` values.

2. [Option A / _otr_ledger_reviewer._render_cast_contract_table] Option A’s role derivation must be the actual STEP 2 migration substitute, not an informal prompt tweak. Current grounded code renders `role = row.get("speaker_role") or ""`, while cast rows produced by `_build_cast_rows` carry no `speaker_role`, so the auditor sees `role=''` for every cast member. Concrete fix: render cast contract role deterministically as:
   - `announcer` when `name.strip().upper() == "ANNOUNCER"` or the row is otherwise the announcer row [ASSUMPTION: live cast rows reliably identify announcer by name or char_id],
   - `character` for all other cast rows,
   - never `music_*` / `sfx` because cue rows are not cast members.

3. [Option A / _otr_ledger_reviewer._render_lines_for_audit / Grounded fact 6] Audit scope must be spoken-only. Current `_render_lines_for_audit` renders every line, including cue rows; cue rows cannot satisfy a cast-member contract and can produce spurious `speaker_unknown` / `role_mismatch`. Concrete fix: filter `lines` through the existing spoken-role