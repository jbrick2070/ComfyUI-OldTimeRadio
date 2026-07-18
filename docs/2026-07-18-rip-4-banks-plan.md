# Rip Plan -- retire 4 banks (Sonnet bake-off verdict, 2026-07-18)

**Decision (operator, 2026-07-18):** rip `scifi_sonnet_v3`, `media_archive_v3`, `scifi_codex_v3`,
`scifi_fable2_v3`. **KEEP `scifi_codex` (base)** -- the reliable codex fallback while `scifi_codex_v4`
stays production-fragile. Rationale + evidence: `docs/2026-07-18-sonnet-bakeoff-analysis.md`.
**Type:** coder-window structural change (roster + packs + pipeline registry + tests + full gate),
mirrors the `499386aa` roster trim. Do NOT run in a render/analysis session.
**Execute against the reusable checklist:** the Teardown protocol in `docs/SOURCE_BANK_PREFLIGHT.md`
(+ lesson 25 in `PRODUCTION_SPRINT_LESSONS.md`) -- this plan is the concrete instance of it.

**HARD RULE -- CLEAN RIP (operator, 2026-07-18):** the 4 banks leave ZERO footprint. No half-rip items
(every surface below removed in the one change), NO negative/absence tests (nothing asserts a ripped id
is "gone"/"unknown"/"not runnable"), and NO "retired-variant coverage" kept alive. If a test's SUBJECT
is a ripped bank, DELETE the case -- do not migrate it to a survivor to preserve the ripped thing's
coverage. Tests reference ONLY surviving banks, and the roster/bijection tests assert the surviving 7
POSITIVELY (they list what exists, they do not assert what was removed). A grep for any of the 4 bank
ids -- PLUS the 3 retired pipeline ids and `_otr_scifi_sonnet` -- across `nodes`/`tests`/`workflows`
returns nothing, test bodies included (but NOT bare `scifi_sonnet`: the `:1947` focus branch is kept).

**ATOMIC (kibitz r3 SHOULD-FIX 1).** Delete the bank rows + pack dirs + `story_rules` + BOTH pipeline
entries as ONE change BEFORE running `_otr_story_routing._ensure_loaded()`: `_sweep_and_crossref()`
(`_otr_story_routing.py:375-390`) rejects a `story_packs/` subdir not registered in `banks.json`, so a
half-applied delete (row gone, pack dir still on disk -- or vice-versa) false-fails validation.

### Kibitz r1 hardening (folded, all CONFIRMED against the code)
- **BOTH pipeline registries** -- delete the 3 v3 pipelines from `_RUNNER_BY_PIPELINE` AND
  `nodes/story_packs/pipelines.json` (:554/:663/:948); keep `legacy_many_pass_v3` (shared). See §4/§4b.
- **Sonnet full-lane** -- delete the module `nodes/_otr_scifi_sonnet.py` (§4/§4c). **Do NOT** delete the
  `base == "scifi_sonnet"` branch at `OTR_LedgerScriptWriter.py:1947` -- that is `_v3_focus_metric`
  (advisory, NOT a dispatch route), reachable via the surviving inline-v3 advisory at :6905-6910, and it
  is on the MUST-KEEP fence. Delete only `_make_v3_runner`; KEEP `run_v3_advisory` / `_v3_focus_metric` /
  `_v3_max_run` and the :1947 branch (kibitz r4 MUST-FIX 1 -- resolves the stale-vs-corrected contradiction).
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
  (`docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`) -- no code fix needed. Per §6: APPEND
  `docs/PROD_BUG_LOG.md` first, THEN mark the NEWBUG doc CLOSED-BY-RIP. Do NOT delete it -- it is the only
  causal record of a live failure (kibitz r4 MUST-FIX 2 -- resolves the delete-vs-never-delete contradiction).
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
   `_run_scifi_sonnet_lane` (:1851-1857). **(kibitz-QA CORRECTION -- do NOT touch `if base ==
   "scifi_sonnet":` at :1947: it is the `reader_alternation` advisory branch INSIDE `_v3_focus_metric`,
   NOT a dispatch route, and the MUST-KEEP fence keeps it -- surviving inline-v3 banks reach it via :6907.
   Sonnet dispatch is fully removed by the `_RUNNER_BY_PIPELINE` entry + `_run_scifi_sonnet_lane` + the
   module; there is no separate base-route to delete.)**
   **`_make_v3_runner` (:1968-1994) goes DEAD once those 3 entries are removed** -- it has NO other
   caller (`legacy_many_pass_v3` runs via `_INLINE_V3_PIPELINES` :2008-2010, not `_make_v3_runner`;
   verified in-source, QA flag 3) -- so DELETE it in the same change or it is a dead lever
   (GO_FORWARD item 5 / "dead levers cost live rolls"). KEEP `_run_scifi_codex_lane`
   (`scifi_codex_circuit_v4` + base) and `_run_fable2_lane` (base `fable2_multipass`) -- both retain a
   live caller; confirm each at build (no kept runner left without a consumer).
   **MUST-KEEP FENCE (QA -- SEV-high over-rip, Bible 05.07 NameError / 12.24 regression):** delete ONLY
   `_make_v3_runner`. Its callees `run_v3_advisory` (:1871), `_v3_focus_metric` (:1911), `_v3_max_run`
   (:1860) sit in the same contiguous `_v3_*` block (:1860-1978) and share the "v3 bake-off" name -- but
   they are NOT dead: `run_v3_advisory` has a SECOND caller at :6905-6908 (the inline lane, guarded by
   `default_story_pipeline in _INLINE_V3_PIPELINES`) that the SURVIVING `legacy_many_pass_v3` banks
   (`public_domain_story_v3`, `shakespeare_v3`) execute on EVERY render. Deleting them = runtime
   `NameError` for both banks. Do NOT sweep the `_v3_*` block; excise the one wrapper. (Aside: the
   `scifi_codex`/`sonnet`/`fable2` branches in `_v3_focus_metric` :1935-1954 go unreachable post-rip --
   LEAVE them; trimming a live helper mid-rip reopens this risk for no gain.)
4b. **`nodes/story_packs/pipelines.json`** (kibitz r1 MUST-FIX -- the OTHER pipeline registry; loaded by
   `nodes/_otr_story_routing.py:499-505`) -- delete the pipeline objects `fable2_multipass_v3` (:554),
   `scifi_codex_circuit_v3` (:663), `sonnet_archive_multipass_v3` (:948). KEEP `legacy_many_pass_v3`
   (surviving banks use it at `banks.json:149`, `:185`). A retired pipeline left here is a semantic
   registry failure even if no bank points at it.
4c. **`nodes/_otr_scifi_sonnet.py`** (kibitz r1 MUST-FIX) -- DELETE the whole module (~1300 LOC; the sonnet
   lane implementation, entrypoint `run_scifi_sonnet_episode` :1155). No RUNTIME consumer once the runner +
   pipeline are gone -- **BUT (kibitz r2 MUST-FIX) three SURVIVING tests import it directly and fail at
   COLLECTION** if the module vanishes: `test_rss_source_admission.py:11`, `test_scifi_source_repair.py:5`,
   `test_scifi_lane_schema_parity.py` (:32/:114/:316/:373/:452/:477/:626/:655/:747). Handle them in the
   SAME change (§5). Remove sonnet from BOTH the `LANE_MODULES` (:30-32) AND `LANE_SOURCES` (:745-747)
   tuples plus every sonnet-only parametrization (:114/:316/:373/:452/:477/:626/:655) -- the test has two
   parallel roster tuples, not one (kibitz r4 SHOULD-FIX 1); the dangling-ref grep must also cover
   `_otr_scifi_sonnet` + `sonnet_archive_multipass_v3` across `nodes/` + `tests/`, not just the 4 bank ids.
4d. **`nodes/OTR_LedgerScriptWriter.py:3757`** (kibitz r2 MUST-FIX -- a SECOND live `fable2_multipass_v3`
   ref) -- the fable2 target-word gate is `if _selected_pipeline_id in ("fable2_multipass",
   "fable2_multipass_v3"):`. Drop `fable2_multipass_v3` (leave the surviving `fable2_multipass`) in the same
   writer change -- it is a PIPELINE id, so it is only caught by the WIDENED gate grep (a bank-id-only grep
   misses it; kibitz-QA Flag 2).
5. **Tests** -- update each (do not just delete assertions; keep coverage honest):
   - `tests/test_scifi_sonnet_lane.py` -> DELETE (lane gone).
   - `tests/test_rss_source_admission.py` (:11), `tests/test_scifi_source_repair.py` (:5),
     `tests/test_scifi_lane_schema_parity.py` (the ~10 refs above) -> remove the `_otr_scifi_sonnet` import
     and DELETE the sonnet-only schema/parity cases (sonnet is fully removed -- there is no surviving sonnet
     to migrate to); keep the codex/fable2 coverage intact (kibitz r2 MUST-FIX 1, CLEAN RIP).
   - `tests/test_bank_variants.py` -> update the runnable-count (11->7) + id lists + DELETE the 4 ripped
     bijection rows (incl. `:146` `scifi_fable2_v3`) so it asserts the surviving 7 POSITIVELY (not the
     absence of the 4). The advisory tests using retired ids (`:210-218` `scifi_sonnet_v3`, `:229-230`
     `media_archive_v3`): if the case's SUBJECT is the ripped bank, DELETE it; if it tests the SURVIVING
     inline-v3 advisory mechanism, keep it with a surviving fixture id (`public_domain_story_v3` /
     `shakespeare_v3`) -- a live positive test, never an absence test (kibitz r2 MUST-FIX 3, CLEAN RIP).
   - `tests/test_fable2_registry.py` -> state the NEW EXACT order, not just counts: `:54` pins
     `ids[-3:]` and `:253-259` pins the full `list_bank_ids()` tuple incl. all 4 retired rows. VERIFIED
     new full order (4 retired rows removed from the live tuple, checked against `:253-259`):
     `("media_archive", "original_radio", "scifi_fable2", "scifi_codex", "public_domain_story_v3",
     "shakespeare_v3", "scifi_codex_v4", "custom_source_bank")` (kibitz r2 MUST-FIX 4 + r4 SHOULD-FIX 3).
   - `tests/test_source_snapshot.py` (`:129`/`:157`/`:275-276`, `scifi_fable2_v3`) -> **DELETE those cases**
     (CLEAN RIP). That coverage was FOR the ripped variant; do NOT migrate it to a survivor and do NOT add
     an absence test. (If a SURVIVING inline-v3 bank needs snapshot coverage, that is a separate positive
     test, not a rescue of this one.)
   - `tests/test_bank_scalar_defaults.py` and the v4-guard tests (`test_genre_guard_spoken_v4`,
     `test_outro_guard_v4`, `test_placeholder_guard_v4`, `test_scene_guard_v4`, `test_provenance_v4`) ->
     the guard tests enumerate banks directly (`test_placeholder_guard_v4.py:103-104`,
     `test_scene_guard_v4.py:91-92`, `test_provenance_v4.py:112-113`) -- regenerate those lists from the
     surviving runnable roster or pin the exact 7 ids; the v4-guard "gate-off" contrast likely uses
     `scifi_codex_v3`, so pick a surviving contrast bank.
   - `tests/test_fable2_tail_context.py:295-299` (kibitz r3 MUST-FIX 1) -> drop `"scifi_sonnet"` from the
     `source_bank` parametrize (:297); the sonnet lane is gone. (Bare-sonnet -- caught by the tests-only
     grep, not the main sweep.)
   - Also update the now-stale `_RUNNER_BY_PIPELINE` comment at `OTR_LedgerScriptWriter.py:1981-1988`
     (it still describes "the three sci-fi v3 pipelines"); cosmetic but it misleads the next wiring edit
     (kibitz r3 SHOULD-FIX 3).
6. **Docs** -- README.md (4 refs) roster list; `docs/GO_FORWARD_PLAN.md` current roster + the item-3
   note; append `docs/HANDOFF_LOG.md`. **NEWBUG handling (kibitz r3 MUST-FIX 2):** `docs/PROD_BUG_LOG.md`
   has NO `scifi_fable2_v3` entry yet -- APPEND one FIRST (the live failure + fix = "retired the runnable
   bank + its pipeline/route"), THEN mark `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md` CLOSED-BY-RIP.
   NEVER delete it -- it is the only causal record of a live failure.

## Gate (per CLAUDE.md -- all must pass before commit)

- **Import-smoke (QA flag 1 -- Bible 03.01/03.02):** after deleting `nodes/_otr_scifi_sonnet.py`, load
  the node registry clean ("All N nodes loaded, 0 skips") and grep the REPO-ROOT `__init__.py` (the real
  loader surface -- `NODE_CLASS_MAPPINGS` lives at `__init__.py:116` + `:351-363`, NOT in `nodes/__init__.py`;
  kibitz r4 MUST-FIX 3) for any leftover sonnet key. A string grep proves the ids are gone; it does NOT
  prove the pack still imports.
- **Ledger-ownership (QA flag 2 / CLAUDE.md "no hole in the ledger"; narrowed by kibitz r3 CUT 1):** for
  the 3 `_v3` banks the removed EXACT ids are the only computed-key prefixes -> already covered by the
  retired-id grep. The one non-obvious residual is SONNET: `meta["scifi_sonnet"]` (BARE) is written all
  over the deleted `_otr_scifi_sonnet.py` and is excluded from the main grep -- so confirm no SURVIVING
  reader of `meta["scifi_sonnet"]` (its readers die with the module; the kept `:1947`/advisory path reads
  a DIFFERENT `<lane>_advisory` key, not this one). So the enumeration collapses to: retired-id grep +
  one `meta["scifi_sonnet"]`-reader check.
- **Dead-runner check (QA flag 3):** every KEPT runner has a live caller post-rip; `_make_v3_runner` is
  DELETED (dead once the 3 v3 pipelines go), but its `_v3_*` neighbors are KEPT (see the MUST-KEEP fence).
- **Runtime advisory smoke (QA -- the gap every other gate misses):** a targeted unit test OR a 30-word
  live smoke of a SURVIVING inline-v3 bank (`public_domain_story_v3` or `shakespeare_v3`) that reaches the
  advisory call at :6907. The bank-id grep never contains `run_v3_advisory`; import-smoke is import-time
  while :6907 is RUNTIME -- only running a surviving inline-v3 bank proves the `_v3_*` helpers survived.
- Full Windows suite (`.venv` python, `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) GREEN.
  **Gate on GREEN + retired-id absence, NOT a predicted total** -- record old/new suite counts as evidence.
- Bug Bible regression GREEN. **Record the count as evidence; do NOT pin "17"** (QA flag 4 -- same
  anti-brittle rule as the suite total; a hardcoded count false-fails or masks a real drop if a ripped
  bank had Bible coverage).
- Registry correctness on `banks.json` + `pipelines.json`: JSON load/round-trip **+
  `_otr_story_routing._ensure_loaded()` (registry load + crossref sweep, :499-505)** -- that is what
  enforces registry JSON, NOT the workflow node (kibitz r2 SHOULD-FIX 1). Run `OTR_WorkflowValidator` on
  the canonical WORKFLOW. **No-BOM/UTF-8 `head -c3`** on both JSONs after edit (QA flag 5 -- Bible
  02.11/12/13). `workflows/otr_canonical.json` byte-unchanged (QA-verified: none of the 4 ids present,
  no stranded COMBO -- BUG-08.06/12.23 not triggered).
- No dangling ref (kibitz-QA Flag 2 -- must cover the PIPELINE ids + module, not just bank ids; §4c/§4d
  retire different strings):
  `grep -r "scifi_sonnet_v3\|media_archive_v3\|scifi_codex_v3\|scifi_fable2_v3\|fable2_multipass_v3\|scifi_codex_circuit_v3\|sonnet_archive_multipass_v3\|_otr_scifi_sonnet\|run_scifi_sonnet_episode" nodes tests workflows`
  returns nothing (docs/tmp excepted). Run it **SOURCE-ONLY** -- `--include='*.py' --include='*.json'
  --exclude-dir=__pycache__` (or delete `nodes/**/__pycache__` + `tests/**/__pycache__` first): stale
  `.pyc` files false-fail otherwise (kibitz r3 MUST-FIX 3). **Do NOT grep bare `scifi_sonnet`** in the
  main sweep -- the kept `:1947` focus branch (MUST-KEEP fence) would false-fail.
- Carve-out blind-spot closer (kibitz r3 MUST-FIX 1 / SHOULD-FIX 2): the bare-`scifi_sonnet` exclusion
  above means bare-`scifi_sonnet` TEST refs slip through, so ALSO run `grep -rn '"scifi_sonnet"' tests`
  -> returns nothing. Every bare-sonnet test ref is to the ripped lane and must be DELETED:
  `test_fable2_tail_context.py:297` (drop `"scifi_sonnet"` from the parametrize) and
  `test_bank_variants.py` `:49` (the `("scifi_sonnet_v3","scifi_sonnet")` tuple), `:58` (the real-base
  list), `:148` (the bijection row). The ONLY allowed bare `scifi_sonnet` is the `:1947` branch in
  `nodes/OTR_LedgerScriptWriter.py`.

**PowerShell gate commands (kibitz r4 MUST-FIX 4 -- the builder runs powershell.exe; `grep`/`head` do
not exist).** Translate the unix forms above to PS:
- Source-only retired-id scan: `Get-ChildItem nodes,tests,workflows -Recurse -Include *.py,*.json |
  Where-Object FullName -notmatch '__pycache__' | Select-String -Pattern 'scifi_sonnet_v3|media_archive_v3|scifi_codex_v3|scifi_fable2_v3|fable2_multipass_v3|scifi_codex_circuit_v3|sonnet_archive_multipass_v3|_otr_scifi_sonnet|run_scifi_sonnet_episode'`
  -> zero rows.
- Tests-only bare-sonnet scan: `Get-ChildItem tests -Recurse -Include *.py | Select-String -Pattern '"scifi_sonnet"'`
  -> zero rows.
- No-BOM/UTF-8 (SHOULD-FIX 2 -- check EVERY touched text file, not just the two JSONs: the edited `.md`,
  `.py` tests, `banks.json`, `pipelines.json`): for each `$p`, `[System.IO.File]::ReadAllBytes($p)[0..2]`
  must NOT equal `239 187 191` (the UTF-8 BOM). AST-parse touched `.py` via the venv python.
- Commit + push to `v2.0-alpha`; verify `HEAD == origin`; AST-parse touched .py.

## Follow-up (PINNED 2026-07-18 -- separate coder chunk, AFTER this rip)

Decouple the "v3 class" toward fully independent banks: rip the shared `_v3_*` advisory
(`run_v3_advisory` :1871 + `_v3_focus_metric` + `_v3_max_run`), `_INLINE_V3_PIPELINES` :2008, and the
:6907 inline call. `run_v3_advisory` is ADVISORY-ONLY (writes `meta["<lane>_advisory"]`, never mutates
lines/beats/cast, never raises) with no story-critical consumer -- so there is NOTHING to inline. Bonus:
it imports `base_source_bank_id` (:1892) and is likely the LAST writer-path user of the version-family
map -- rip it and the lineage machinery goes too. Touches the surviving inline-v3 banks
(`public_domain_story_v3`, `shakespeare_v3`) + the freeze-cascade/ledger-scrub persistence + its
persistence test, so it is its own gated chunk. Verify `base_source_bank_id` has no other live caller
before removing the family-map.

## Kickoff line for a fresh coder window

> resume the OTR build as a CODER window. Execute `docs/2026-07-18-rip-4-banks-plan.md` in one green
> pushed chunk: rip scifi_sonnet_v3 (full lane) + media_archive_v3 + scifi_codex_v3 + scifi_fable2_v3,
> KEEP scifi_codex base. Full gate (import-smoke 0-skips + ledger-ownership enumeration + suite/Bible
> GREEN-recorded-not-pinned + no-BOM + canonical byte-unchanged + no dangling ref), then commit+push to
> v2.0-alpha and update GO_FORWARD + HANDOFF.

## Verify-at-build (kibitz r4 consolidated pre-flight -- run top to bottom)

1. **Import-smoke:** node registry loads with 0 skips; repo-root `__init__.py` / `NODE_CLASS_MAPPINGS`
   has no sonnet reference.
2. **Registry load:** JSON round-trip for `nodes/story_packs/banks.json` + `pipelines.json`;
   `_otr_story_routing._ensure_loaded()` succeeds and contains no retired pipeline ids. (Delete rows +
   packs + rules + pipeline entries ATOMICALLY before this runs -- see the ATOMIC note.)
3. **Workflow:** `workflows/otr_canonical.json` byte-unchanged; `OTR_WorkflowValidator` passes.
4. **Ledger ownership:** no surviving reader of `meta["scifi_sonnet"]`; retired computed keys covered by
   the retired-id grep.
5. **Dead-runner:** kept runners have live callers; `_make_v3_runner` deleted; `run_v3_advisory` /
   `_v3_focus_metric` / `_v3_max_run` + the :1947 branch still exercised.
6. **Runtime advisory:** a 30-word live smoke of a SURVIVING inline-v3 bank (`public_domain_story_v3` or
   `shakespeare_v3`) reaches `run_v3_advisory` at :6907 (import-smoke does NOT prove this -- it is runtime).
7. **Source-only retired-ref scan** (PS `Select-String`, `__pycache__` excluded) over `nodes,tests,workflows`
   returns nothing for the 4 bank ids, 3 pipeline ids, `_otr_scifi_sonnet`, `run_scifi_sonnet_episode`.
8. **Tests-only bare-sonnet scan** returns no `"scifi_sonnet"` except the allowed `:1947` branch.
9. **Full Windows suite green; Bug Bible regression green;** counts RECORDED, not pinned.
10. `docs/PROD_BUG_LOG.md` appended BEFORE the NEWBUG doc is marked CLOSED-BY-RIP (never deleted).
11. Touched `.py` AST-parse; touched text files no BOM / UTF-8; commit AND push to `v2.0-alpha`; verify
    `HEAD == origin`.
