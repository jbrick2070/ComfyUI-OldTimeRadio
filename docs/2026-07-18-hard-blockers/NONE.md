# 2026-07-18 HARD blockers -- NONE

Two-track disposition run 2026-07-18 (docs window, HEAD `d6b0706e`, branch
`v2.0-alpha`) classified every remaining open 2026-07-18 item HARD vs SOFT per
the operator directive: "Yes /kibitz on any hard blockers but if LLMs are killing
the flow with the ledger intact they should not as long as ledger is obeyed".

**Result: 0 HARDs.** Every remaining open 2026-07-18 item is SOFT (LLM quirk,
ledger invariant intact). No kibitz round required; disposition routes entirely
through `docs/2026-07-18-soft-normalizers/SPRINT_PLAN.md`.

## Items reviewed

| Item | Class | Ledger status |
|---|---|---|
| codex_v4 P2 cast-name Title-Case (`Maxwell 'Max' Hart`) | SOFT | `CastPlanV4` schema intact; `char_id`/`gender`/`character_description` untouched; failure is only on the `_is_canonical_character_name` cosmetic Title-Case check. |
| codex_v4 P5 self-vocative (line begins with speaker's own name) | SOFT | `ScriptArtifactV4` schema intact; line row shape, `speaker_role`, `boundary`, and graph closure untouched; a deterministic scrub already exists at `nodes/_otr_line_hygiene.py:69` (`scrub_self_vocative`). |
| NEWBUG `scifi_fable2_v3` rules_id | CLOSED-BY-RIP 2026-07-18 midday (not open; bank retired). |
| Rip 4 banks (`c507acff`) | Landed 2026-07-18 midday (not open). |
| COUNT-gate advisory (`ed7b37de`) | Landed 2026-07-18 evening; live-proven on fable2 120w Mistral-Nemo ("The Caretaker's Dilemma", 108 MB) (not open). |
| Local Mistral 420w/720w bake-off | Not itself a blocker; a downstream campaign that unblocks once P2 + P5 land. Tracked in the SOFT sprint bake-off checklist, not here. |

## HARD invariants verified intact (would flip to HARD if ANY broke)

- Ledger schema shape and field types (v4 pydantic models `CastPlanV4`,
  `ScriptArtifactV4`, `RadioScoreV4`).
- `ROW_KEYED` merge invariant.
- `l3-2026-05-14` schema (no drift).
- `test_audio_byte_identical` (byte-identical audio).
- Row ordering / deterministic seed reproducibility.
- Ledger production terminal (`_assemble_ledger` completes).

None broken by P2 nickname markers or P5 leading vocatives -- both are line- or
row-content quirks upstream of ledger closure that a mechanical scrub or the
`advisory_budget_defects` ladder handles without altering the invariant.

## Disposition

Proceed to SOFT sprint: `docs/2026-07-18-soft-normalizers/SPRINT_PLAN.md`.
