# Rip Plan -- retire 4 banks (Sonnet bake-off verdict, 2026-07-18)

**Decision (operator, 2026-07-18):** rip `scifi_sonnet_v3`, `media_archive_v3`, `scifi_codex_v3`,
`scifi_fable2_v3`. **KEEP `scifi_codex` (base)** -- the reliable codex fallback while `scifi_codex_v4`
stays production-fragile. Rationale + evidence: `docs/2026-07-18-sonnet-bakeoff-analysis.md`.
**Type:** coder-window structural change (roster + packs + pipeline registry + tests + full gate),
mirrors the `499386aa` roster trim. Do NOT run in a render/analysis session.

## Removal depth (READ FIRST -- not all four are equal)

- **`scifi_sonnet_v3` = FULL-LANE removal.** It is the ONLY sonnet bank (no base, no v4), so this
  retires the entire sonnet family: the bank row, pack, story_rules, the `sonnet_archive_multipass_v3`
  pipeline entry, the `_run_scifi_sonnet_lane` runner (+ any sonnet-only helpers it alone calls), and
  the dedicated `tests/test_scifi_sonnet_lane.py`. Grep `_run_scifi_sonnet_lane` and `scifi_sonnet`
  before deleting -- give any orphaned helper a clean removal (no dangling import).
- **`scifi_codex_v3` = v3-only.** Base `scifi_codex` + `scifi_codex_v4` survive. Remove the row/pack/
  rules + the `scifi_codex_circuit_v3` pipeline entry. Keep `_run_scifi_codex_lane` (v4/base use it).
- **`scifi_fable2_v3` = v3-only.** Base `scifi_fable2` survives. Remove the row/pack/rules + the
  `fable2_multipass_v3` pipeline entry. This ALSO closes the NEWBUG
  (`docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`) by deletion -- no code fix needed; delete that doc
  or mark it CLOSED-BY-RIP.
- **`media_archive_v3` = v3-only.** Base `media_archive` survives. Remove the row/pack/rules. Its
  pipeline `legacy_many_pass_v3` is SHARED with `public_domain_story_v3` + `shakespeare_v3` -- DO NOT
  remove that pipeline.

## Exact edits

1. **`nodes/story_packs/banks.json`** -- delete the 4 `banks[]` rows (`source_bank_id` ==
   scifi_sonnet_v3 / media_archive_v3 / scifi_codex_v3 / scifi_fable2_v3). Runnable roster goes 11 -> 7
   (media_archive, original_radio, scifi_fable2, scifi_codex, public_domain_story_v3, shakespeare_v3,
   scifi_codex_v4) + custom (non-runnable). Re-validate JSON.
2. **Delete pack dirs:** `nodes/story_packs/scifi_sonnet_v3/`, `nodes/story_packs/media_archive_v3/`,
   `nodes/story_packs/scifi_codex_v3/`, `nodes/story_packs/scifi_fable2_v3/`.
3. **Delete story_rules:** `nodes/story_rules/{scifi_sonnet_v3,media_archive_v3,scifi_codex_v3,scifi_fable2_v3}.json`.
4. **`nodes/OTR_LedgerScriptWriter.py`** PIPELINE_REGISTRY (~:1996-2001) -- delete the 3 entries
   `fable2_multipass_v3`, `scifi_codex_circuit_v3`, `sonnet_archive_multipass_v3`. Then remove
   `_run_scifi_sonnet_lane` and any sonnet-only helper now unreferenced (grep first). Leave
   `_run_scifi_codex_lane` / `_run_fable2_lane` / `_make_v3_runner` (still used).
5. **Tests** -- update each (do not just delete assertions; keep coverage honest):
   - `tests/test_scifi_sonnet_lane.py` -> DELETE (lane gone).
   - `tests/test_bank_variants.py` (13 refs) -> update the runnable-count + the id lists/bijection
     (12 visible / 11 runnable -> 8 visible / 7 runnable; drop the 4 ids).
   - `tests/test_fable2_registry.py`, `tests/test_bank_scalar_defaults.py`,
     `tests/test_source_snapshot.py` (5), and the v4-guard tests
     (`test_genre_guard_spoken_v4`, `test_outro_guard_v4`, `test_placeholder_guard_v4`,
     `test_scene_guard_v4`, `test_provenance_v4`) -> read each; where a ripped bank is used as a
     fixture/param/baseline, swap to a surviving bank (e.g. `scifi_codex_v4` or base `scifi_codex`) or
     drop that case. The v4-guard tests likely use `scifi_codex_v3` as a "gate-off" contrast -- pick a
     surviving contrast bank.
6. **Docs** -- README.md (4 refs) roster list; `docs/GO_FORWARD_PLAN.md` current roster + the item-3
   note; append `docs/HANDOFF_LOG.md`. Mark the NEWBUG doc CLOSED-BY-RIP.

## Gate (per CLAUDE.md -- all must pass before commit)

- Full Windows suite (`.venv` python, `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) GREEN --
  expect the total to DROP from 8144 by the removed cases; record the new number.
- Bug Bible regression GREEN (17).
- `OTR_WorkflowValidator` + JSON round-trip on `banks.json`; `workflows/otr_canonical.json` byte-unchanged.
- No dangling ref: `grep -r "scifi_sonnet_v3\|media_archive_v3\|scifi_codex_v3\|scifi_fable2_v3" nodes tests workflows` returns nothing (docs/tmp excepted).
- Commit + push to `v2.0-alpha`; verify `HEAD == origin`; AST-parse touched .py.

## Kickoff line for a fresh coder window

> resume the OTR build as a CODER window. Execute `docs/2026-07-18-rip-4-banks-plan.md` in one green
> pushed chunk: rip scifi_sonnet_v3 (full lane) + media_archive_v3 + scifi_codex_v3 + scifi_fable2_v3,
> KEEP scifi_codex base. Full gate (suite + Bible 17 + canonical byte-unchanged + no dangling ref),
> then commit+push to v2.0-alpha and update GO_FORWARD + HANDOFF.
