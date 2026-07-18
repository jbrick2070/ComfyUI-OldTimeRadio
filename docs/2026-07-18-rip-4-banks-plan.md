# Rip Plan -- retire 4 banks (Sonnet bake-off verdict, 2026-07-18)

**Decision (operator, 2026-07-18):** rip `scifi_sonnet_v3`, `media_archive_v3`, `scifi_codex_v3`,
`scifi_fable2_v3`. **KEEP `scifi_codex` (base)** -- the reliable codex fallback while `scifi_codex_v4`
stays production-fragile. Rationale + evidence: `docs/2026-07-18-sonnet-bakeoff-analysis.md`.
**Type:** coder-window structural change (roster + packs + pipeline registry + tests + full gate),
mirrors the `499386aa` roster trim. Do NOT run in a render/analysis session.
**Execute against the reusable checklist:** the Teardown protocol in `docs/SOURCE_BANK_PREFLIGHT.md`
(+ lesson 25 in `PRODUCTION_SPRINT_LESSONS.md`) -- this plan is the concrete instance of it.

### Kibitz r1 hardening (folded, all CONFIRMED against the code)
- **BOTH pipeline registries** -- delete the 3 v3 pipelines from `_RUNNER_BY_PIPELINE` AND
  `nodes/story_packs/pipelines.json` (:554/:663/:948); keep `legacy_many_pass_v3` (shared). See §4/§4b.
- **Sonnet full-lane** -- also delete the module `nodes/_otr_scifi_sonnet.py` + the `base == "scifi_sonnet"`
  route (§4/§4c).
- **NEWBUG -> PBUG, don't discard** -- `scifi_fable2_v3` carried a repeatable LIVE failure; append a
  `docs/PROD_BUG_LOG.md` entry (fix = "retired the runnable bank + pipeline/route"), THEN mark the NEWBUG
  doc CLOSED-BY-RIP. Do NOT edit `_otr_scifi_fable2.py` for the deleted lane.
- **Gate: registry invariant + no brittle count** -- add "no retired `story_pipeline_id` in
  `_otr_story_routing._ensure_loaded().pipelines`" and gate on GREEN suite + retired-id absence, not a
  predicted 8144 total (record old/new counts as evidence only).
- **v4-guard tests enumerate banks directly** (`test_placeholder_guard_v4.py:103-104`,
  `test_scene_guard_v4.py:91-92`, `test_provenance_v4.py:112-113`) -- regenerate those lists from the
  surviving runnable roster or pin the exact 7 ids.

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
4. **`nodes/OTR_LedgerScriptWriter.py`** `_RUNNER_BY_PIPELINE` (~:1996-2001) -- delete the 3 entries
   `fable2_multipass_v3`, `scifi_codex_circuit_v3`, `sonnet_archive_multipass_v3`. Then remove
   `_run_scifi_sonnet_lane` (:1851-1857) and the base route `if base == "scifi_sonnet":` (:1947).
   **`_make_v3_runner` (:1968-1994) goes DEAD once those 3 entries are removed** -- it has NO other
   caller (`legacy_many_pass_v3` runs via `_INLINE_V3_PIPELINES` :2008-2010, not `_make_v3_runner`;
   verified in-source, QA flag 3) -- so DELETE it in the same change or it is a dead lever
   (GO_FORWARD item 5 / "dead levers cost live rolls"). KEEP `_run_scifi_codex_lane`
   (`scifi_codex_circuit_v4` + base) and `_run_fable2_lane` (base `fable2_multipass`) -- both retain a
   live caller; confirm each at build (no kept runner left without a consumer).
4b. **`nodes/story_packs/pipelines.json`** (kibitz r1 MUST-FIX -- the OTHER pipeline registry; loaded by
   `nodes/_otr_story_routing.py:499-505`) -- delete the pipeline objects `fable2_multipass_v3` (:554),
   `scifi_codex_circuit_v3` (:663), `sonnet_archive_multipass_v3` (:948). KEEP `legacy_many_pass_v3`
   (surviving banks use it at `banks.json:149`, `:185`). A retired pipeline left here is a semantic
   registry failure even if no bank points at it.
4c. **`nodes/_otr_scifi_sonnet.py`** (kibitz r1 MUST-FIX) -- DELETE the whole module (~1300 LOC; the sonnet
   lane implementation, entrypoint `run_scifi_sonnet_episode` :1155). It has no surviving consumer once the
   runner + pipeline are gone. Grep `_otr_scifi_sonnet` / `scifi_sonnet` across `nodes/` after deletion and
   clean any orphaned import (e.g. an interpreter/`validate_source_payload("scifi_sonnet")` registration).
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

- **Import-smoke (QA flag 1 -- Bible 03.01/03.02):** after deleting `nodes/_otr_scifi_sonnet.py`, load
  the node registry clean ("All N nodes loaded, 0 skips") and grep `nodes/__init__.py` +
  `NODE_CLASS_MAPPINGS` for any leftover sonnet key. A string grep proves the ids are gone; it does NOT
  prove the pack still imports.
- **Ledger-ownership enumeration (QA flag 2 -- CLAUDE.md "no hole in the ledger" + PBUG-20260712-05 /
  BUG-12.49):** for EACH of the 4 banks, enumerate what it stamped into the ledger -- including COMPUTED
  keys (`f"{source_bank_id}_..."`) a literal grep misses -- and confirm zero surviving readers in the
  shared writer tail. A green suite does not prove ledger completeness (bank provenance is self-keyed so
  the residual is likely narrow -- but state the step, don't skip it).
- **Dead-runner check (QA flag 3):** every KEPT runner has a live caller post-rip; `_make_v3_runner` is
  DELETED (dead once the 3 v3 pipelines go).
- Full Windows suite (`.venv` python, `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) GREEN.
  **Gate on GREEN + retired-id absence, NOT a predicted total** -- record old/new suite counts as evidence.
- Bug Bible regression GREEN. **Record the count as evidence; do NOT pin "17"** (QA flag 4 -- same
  anti-brittle rule as the suite total; a hardcoded count false-fails or masks a real drop if a ripped
  bank had Bible coverage).
- `OTR_WorkflowValidator` + JSON round-trip on `banks.json` + `pipelines.json`; **no-BOM/UTF-8 check
  (`head -c3`) on both after edit** (QA flag 5 -- Bible 02.11/12/13); `workflows/otr_canonical.json`
  byte-unchanged (QA-verified: it carries none of the 4 ids, so no stranded COMBO -- BUG-08.06/12.23 not
  triggered).
- No dangling ref: `grep -r "scifi_sonnet_v3\|media_archive_v3\|scifi_codex_v3\|scifi_fable2_v3" nodes tests workflows` returns nothing (docs/tmp excepted).
- Commit + push to `v2.0-alpha`; verify `HEAD == origin`; AST-parse touched .py.

## Kickoff line for a fresh coder window

> resume the OTR build as a CODER window. Execute `docs/2026-07-18-rip-4-banks-plan.md` in one green
> pushed chunk: rip scifi_sonnet_v3 (full lane) + media_archive_v3 + scifi_codex_v3 + scifi_fable2_v3,
> KEEP scifi_codex base. Full gate (import-smoke 0-skips + ledger-ownership enumeration + suite/Bible
> GREEN-recorded-not-pinned + no-BOM + canonical byte-unchanged + no dangling ref), then commit+push to
> v2.0-alpha and update GO_FORWARD + HANDOFF.
