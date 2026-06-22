# R1 judgment -- STEP 2 scope conflict

## Convergence: YES (R1, unanimous). Stop here per CLAUDE.md S8 (narrow single
## decision, no dissent, no new material beyond the grounded anchor).

## Accepted (all grounded CONFIRMED)
- Adopt Option A; cut Option B; do not skip (Option C). [GPT, Gemini, DeepSeek, anchor]
- Change 1: derive announcer/character cast role in `_render_cast_contract_table`. CONFIRMED
  vs `_build_cast_rows` (no role field) + cast_lock `_is_announcer_entry` (name==ANNOUNCER).
- Change 2: filter `_render_lines_for_audit` to `is_spoken_role`. CONFIRMED vs
  `audit_cast_contract` (feeds ALL lines) + `_otr_ledger_scrub.is_spoken_role`.
- GPT MUST-FIX (acceptance measurable): folded as Change 3 (re-soak counts flags).

## Rejected / cut
- Option B `cue_type`/`archetype` migration: CONFIRMED inapplicable + schema-breaking
  (zero occurrences in tree; frozen schema). All three panelists independently agree.
- Explicit-speaker_role-wins ordering (anchor SHOULD-FIX): kept (Change 1 uses
  `row.get("speaker_role") or <derive>`), low cost, future-proofs.

## Verify-at-build (UNVERIFIABLE downgraded)
- GPT [ASSUMPTION] "cast rows reliably identify announcer by name or char_id":
  CONFIRMED for name (cast_lock `_is_announcer_entry` uses name OR role==announcer);
  char_id is not a reliable announcer key (announcer may be c01..cNN). Use NAME +
  any explicit speaker_role. No char_id-based detection.

## Panel truncation note
All three panel outputs hit the 2000-token cap mid-MUST-FIX-3 (the spoken-row
filter). The cut tail is the same fix the anchor + the other models fully state;
no lost content affects the decision.

## Spend
R1 ~$0.3293 total. No R2-R4 (converged; grinding further would only re-confirm).
