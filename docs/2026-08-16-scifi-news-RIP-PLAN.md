# RIP PLAN -- retiring the `scifi_news` bank + the codex lane

**Operator decision 2026-08-16:** *"we ditch scifi_news and the pro becomes
the standard... we leave the name pro since it won."* Evidence: the blind
per-bank narrative read (`docs/2026-08-16-blind-bank-narrative-ranking.md`)
scored `scifi_news` LAST at 2.0/10 and `scifi_news_pro` FIRST at 7.5/10, and
the deterministic structure score ranked the same two banks last and first
independently.

Governing checklist: `docs/SOURCE_BANK_PREFLIGHT.md` "Teardown protocol".
Surfaces enumerated by a four-agent read-only survey, 2026-08-16.
**Pre-rip anchor tag: `otr-2026-08-16-sixbank-lemmyA` at `da44f642`** (the
tree where all six banks were live-proven). Reversal recipe:
`SOURCE_BANK_PREFLIGHT.md` "Reversal" section -- THE RIP LANDS AS ONE ATOMIC
COMMIT so `git revert` restores it whole.

**Depth: FULL-FAMILY.** `scifi_news` is the only bank on
`scifi_news_circuit`, whose runner is `nodes/_otr_scifi_codex.py`. No
surviving bank routes to that module. (`scifi_news_pro` is a DIFFERENT lane
-- `scifi_news_pro_multipass` -> `_otr_scifi_fable2.py` -- and survives
untouched. Never let a bare `scifi_news` grep match it.)

## GATE A -- the ledger-ownership audit: PASSED, no hole

Every field the lane stamps is (a) inside the `meta["scifi_codex"]`
namespace with ZERO surviving production readers, or (b) a shared key whose
writer survives elsewhere. Computed keys were swept explicitly: the bank id
is never interpolated into a meta key -- it appears only as a VALUE. Nested
computed keys (`line_text_sha256[<line_id>]`, `accepted_lines[<line_id>]`,
`call_journal.p0_repair_trim[<n>]`) die with their parent. The only
non-test readers of the namespace are three defensive `_as_dict` reads in
`scripts/otr_ledger_view.py`. The shared writer tail, freeze cascade,
voice/video/credits nodes and `obs_publish` read none of it.

## Surfaces -- code and data

| # | path | action | note |
|---|---|---|---|
| 1 | `nodes/story_packs/banks.json:166-186` | DELETE row | line 165 ends the prior row, 187 opens the next -- no comma surgery |
| 2 | `nodes/story_packs/pipelines.json:357-454` | DELETE block | **TRAP:** it is the LAST array element -- line 356 `},` must become `}` |
| 3 | `nodes/story_packs/scifi_news/` (1 file) | DELETE dir | **ATOMIC WITH #1:** `_otr_story_routing._sweep_and_crossref` raises `RegistryValidationError` for a pack dir with no bank row, and vice versa -- either alone hard-fails import |
| 4 | `nodes/_otr_lane_specs.py:80-85` | DELETE entry | the ONLY live cross-module edge to the runner |
| 5 | `nodes/_otr_lane_specs.py:16` | EDIT docstring | bare-token prose hit; CLEAN RIP requires the final grep to be clean |
| 6 | `nodes/_otr_scifi_codex.py` (4,664 lines) | DELETE module | orphaned once #4 goes; nothing imports it |
| 7 | `nodes/_otr_scifi_source_repair.py` | DELETE module | exactly ONE importer, the codex module -- dies with it (verify at rip time) |
| 8 | `workflows/otr_story_only.json:529` | EDIT value | saved `source_bank` COMBO value would be STRANDED (`value_not_in_list`, BUG-08.06/12.23) -> set to `roll (any eligible bank)` to match canonical |
| 9 | `workflows/otr_canonical.json` | UNCHANGED | verified ZERO "scifi" occurrences; sha256 `02c51da6...` must not move |
| 10 | `nodes/OTR_LedgerScriptWriter.py` INPUT_TYPES default | EDIT | the `source_bank` widget default is `scifi_news`; re-point (canonical saves the roll sentinel, so no widget-order change) |

`nodes/story_rules/` does not exist on this tree -- that checklist line is a
no-op. `config/`, `workflows/variants/`, `workflows/external_examples/`:
zero hits.

## Surfaces -- tests

| path | action | note |
|---|---|---|
| `tests/test_scifi_codex_lane.py` | DELETE file | dedicated lane test |
| `tests/test_codex_news_coda.py`, `test_codex_per_beat_dialogue.py`, `test_p0_deterministic_repair_wired.py`, `test_p0_source_windows.py`, `test_45word_failure_regressions.py`, `test_a3_nbsp_entity_span_regression.py` | DELETE / TRIM | subject is the ripped lane; confirm each file's whole subject before deleting vs trimming |
| `tests/test_clean_transaction.py` (class `TestFinalizerProtocol`, import :379, uses :381/:435/:465/:468) | **MIGRATE** | **TRAP, protocol-named:** its SUBJECT is the SHARED clean-transaction machinery, not the ripped bank. Deleting only the import NameErrors at COLLECTION and takes the whole file down. Re-source the finalizer or build a surviving-lane stand-in |
| `tests/test_cast_lock_policy_repin.py:303` (+ assertion :360) | DELETE row | `BANK_CAMEO_POLICY` is asserted equal to the shipped registry BOTH ways -- a leftover row fails loudly |
| `tests/test_content_owned_cast_contract.py` | TRIM | the codex-only cases (written 2026-08-16); the fable2 + shared-helper cases survive |
| `tests/test_bank_variants.py` | UPDATE roster | id list + counts |
| guard tests enumerating banks (`_CURRENT_BANKS` / inline lists) | UPDATE | grep `_CURRENT_BANKS` across `tests/`; 2026-07-18 precedent missed two on a hand list |

`nodes/_otr_casting.py:1238` `_LEMMY_EXCLUDED_SOURCE_BANK_IDS` is
`{public_domain, shakespeare}` only -- **nothing to remove there.**

## The one open DESIGN decision (operator call, flagged not decided)

`_CodexTailFinalizer` is the ONLY implementer in the repo of BOTH the
writer's `TailFinalizer` protocol (`before_save`/`after_save`) AND the clean
transaction's three-method proof protocol. `Fable2TailParts` has no
`tail_finalizer`. After the rip that extension surface in
`OTR_LedgerScriptWriter` and `_otr_clean_transaction` is PRODUCERLESS.
Nothing breaks -- the transaction gates on `meta.content_authorship` and
stamps its receipt independently -- but the "no dead levers" gate wants an
explicit **keep** (as designed extension space for the next lane) or
**excise** (dead code today). Recommendation: **KEEP**, documented as
extension space, because the next content-owned lane needs exactly this
shape; excising and re-deriving it later is the more expensive mistake.

Second, smaller: `GO_FORWARD:962` asserts "the canonical `scifi_news`
episode topology STANDS and is still the contract." That topology may
re-anchor on `scifi_news_pro` rather than retire with the bank -- an
operator call, not a doc edit.

## Docs -- EDIT (state false present facts)

`README.md:92,98,110-113` (the "shipped six" / "seventh peer" counts and the
default-bank claim); `docs/WRITER_INPUT_MATRIX.md:21-24,29,39` (**after the
rip NO lane binds a decoding grammar** -- that is the file's self-declared
most load-bearing fact); `docs/GO_FORWARD_PLAN.md` (many: sweep criteria
:414-418, repair sites :419-420, chunk A/B scoping :425-458, roll pool
:533-537 **menu 8/pool 6 -> menu 7/pool 5**, row-2 exit :945, topology :962,
test-design caveats :1412, on-deck 6 :1443, gender roster :1484, P0 repair
:1568, the two codex no-shims items :797-806, the rename table :869-888);
`docs/NEXT_WINDOW_KICKOFF.md`; `docs/2026-08-16-lemmy-chunkB-BUILD-CONTRACT.md`
(the codex half evaporates -- see below); `docs/2026-08-16-lemmy-chunkB-cameo-roll-PLAN.md`;
`docs/2026-08-15-graduated-extraction-span-reader-enumeration.md` (CLOSED-BY-RIP);
`docs/2026-08-14-temperature-problem-statement.md`;
`docs/2026-08-14-placed-interrupts-problem-statement.md`;
`docs/2026-08-05-character-gender-ladder-SPEC.md:112,707`;
`docs/2026-08-07-PLAN-small-sprint-items.md:555-556`;
`docs/2026-07-24-independent-source-banks-v1-plan.md:18-20,58-60`;
`docs/2026-08-03-script-parse-repair-CODE-READY.md:27,313`;
**`docs/SOURCE_BANK_PREFLIGHT.md:428-429`** -- the governing checklist's own
worked example cites `scifi_news` as a SURVIVING sibling; it must be
corrected in the same change, and this rip becomes its first full-family
example for a runner-dispatched lane.

`docs/PROD_BUG_LOG.md`: ADD a rip entry (`fix: retired the runnable bank +
its pipeline/route`), correct PBUG-20260811-03's scope back to one lane, and
mark any OPEN entry whose subject is `scifi_news` CLOSED-BY-RIP. Never
delete a causal record.

## Docs -- LEAVE ALONE (historical record, not present fact)

`docs/HANDOFF_LOG.md` (append-only; the rip gets a NEW top entry and nothing
below is edited), the dated findings/handoffs/specs/bake-off records, and
`docs/2026-08-16-blind-bank-narrative-ranking.md` -- which is part of the
RATIONALE for this rip and stays verbatim.

## Chunk B collapses to fable2-only

`docs/2026-08-16-lemmy-chunkB-BUILD-CONTRACT.md` covers both runners; the
codex half (the schema-locked reserve, the five grammar-decoded vocabulary
sites, the P5/P5R speaker-contract route, the conditional preset pre-seed)
is MOOT. What remains is the fable2 design: no schema edits, no grammar
binding, headroom to `MAX_SPEAKING_CAST`. The hard half of the next sprint
disappeared with this decision.

## Gates before the single commit

AST-parse every touched `.py`; node registry loads clean ("All N nodes
loaded, 0 skips") -- grep the REPO-ROOT `__init__.py`, not `nodes/__init__.py`;
`OTR_WorkflowValidator` + JSON round-trip; `otr_canonical.json` sha256
unchanged; no-BOM/UTF-8 on every touched text file; full Windows suite +
Bug Bible GREEN, counts recorded as evidence never pinned; **bare-token**
scan for `scifi_news`/`scifi_codex`/`scifi_news_circuit` over source only
(exclude `__pycache__`) gating on an ENUMERATED carve-out list, never a
blind zero. Tooling note: the Grep tool's ripgrep has look-around DISABLED
(`(?!_pro)` is rejected) -- use `rg --pcre2` from Bash or PowerShell
`Select-String`.
