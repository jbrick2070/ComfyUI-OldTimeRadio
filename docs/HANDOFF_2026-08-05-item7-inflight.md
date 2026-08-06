# HANDOFF -- Item 7 IN FLIGHT, UNCOMMITTED (2026-08-05 evening)

**Branch:** `v2.0-alpha`. **HEAD:** `acb09719`. **The working tree has
UNCOMMITTED Item 7 work.** It survives a restart; nothing is lost. Read this
before touching those files.

## The one thing to do first

Run the full suite, and if green, COMMIT. The work was mid-verification when the
session ended.

```
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -ExecutionPolicy Bypass -File tmp\_run_full_suite.ps1
```

Last known state: the targeted suites pass (62 across the four touched files, and
22 more for the new audit). The previous FULL run surfaced exactly one regression
-- `test_shakespeare_interpreter.py::test_shakespeare_prompt_keeps_rights_terms_and_no_content_guardrail`
-- which is **already fixed** (see "the judgment call" below). The final confirming
run had not reported when the session ended.

## Uncommitted files

| file | change |
|---|---|
| `nodes/OTR_LedgerScriptWriter.py` | the fix: ownership key, routing, receipts, `source_bank_id` |
| `nodes/_otr_public_domain_sources.py` | `URL:` removed from the prompt, `PROMPT_VERSION` -> `v3`, dead `link` local removed |
| `nodes/_otr_shakespeare_sources.py` | `URL:` removed, `PROMPT_VERSION` -> `v2`, `Date/Rights:` KEPT |
| `nodes/_otr_ledger_freeze.py` | stale G14 comment corrected |
| `tests/test_provenance_v4.py` | `TestCurrentBanksInert` renamed (its own assertions showed two ACTIVE banks) |
| `docs/PROD_BUG_LOG.md` | new `PBUG-20260805-04` |
| `scripts/audit_spoken_citations.py` | NEW -- the durable corpus audit |
| `tests/test_spoken_citation_audit.py` | NEW -- 22 tests pinning the audit predicate |

`docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md` was already untracked before
this session. Not mine.

## What Item 7 is

The announcer SPOKE the source citation, and `_otr_captions.py:283-286` copies raw
`lines[].text` into the ASS cue, so it was **burned into the delivered video**.
Measured over 1,587 ledgers: 84 leaked lines, 100% announcer, 100% at the coda
row, 0 non-announcer; 30 episodes on/after 2026-08-04.

The deterministic replacement `meta["provenance_coda_line"]` already existed and
had **zero readers** -- the 2026-08-04 fix for this same defect was applied inside
`spoken_coda_line()`, a function nothing calls. **A fix applied to a function with
no callers is not a fix.**

## DONE in the working tree

- **B1** ownership key `"provenance" in meta`; coda stamped UNCONDITIONALLY (the
  old `if _coda:` left the key absent exactly when provenance was unknown, so a
  presence test misread an owned lane as unowned and fell back to the raw note).
- **B2** owned lane always takes the deterministic append regardless of
  `_style_grammar_on`; owned+empty goes straight to `fallback_announcer_outro("")`
  with NEITHER composer entered (which also dodges the `RuntimeError` at
  `_otr_line_composer.py:1300-1301`).
- **B3** `source_bank_id=resolved["source_bank"]` now passed to
  `compose_news_coda`. **Pre-existing bug** -- every lane had been resolving
  media_archive's `coda_system` prompt.
- **B5** `meta["spoken_coda_source"]` receipt, closed vocabulary
  (`_SPOKEN_CODA_SOURCES`) validated at write time; `news_coda_emitted` now from
  the effective fact and stamped outside the style gate.
  `source_note_deferred_to_credits` made CONDITIONAL on the credits actually
  carrying the line, in BOTH places -- it was false telemetry when
  `credits_source_line` is empty.
- **B7 (partial)** `URL:` stripped from both interpreter prompts + version bumps.
- **B8** stale comments in the writer and the G14 freeze gate; test class renamed.
- The audit + its 22 tests. Corpus baseline: **69 episodes** flagged pre-fix.

## STILL OWED

1. **B4 -- factor the coda block into ONE helper** so tests exercise the
   production reader. **TRAP:** do NOT extract `:5463-5588` verbatim.
   `news_meta` is defined at the top of that range and read by the CALLER below
   it -- extracting as written raises `NameError` on every episode. Both review
   lanes caught this independently. Keep `news_meta` in the caller.
2. **B6 -- bump `CURRENT_SCHEMA_VERSION`** (`nodes/_otr_ledger.py:58`). The audit
   must REQUIRE the receipt on post-fix ledgers while tolerating absence on the
   1,587 legacy ones; without a version boundary a dropped receipt is
   indistinguishable from history. `LEGACY_SCHEMA_VERSIONS` in the audit script
   is already written to expect this. Update the lineage comment,
   `_otr_ledger_freeze.py:92` fallback, fixtures, and keep the l3 compat tests.
3. **Writer-level routing tests** (depend on B4): both fidelity banks x
   {non-empty, empty} provenance; an owned+non-empty case with
   `_style_grammar_on == False`; assert the coda is PRESENT (not merely that the
   sentinel is absent) in `lines[].text` and in a `Dialogue:` cue via
   `build_ass_from_ledger`. **Control is `media_archive`, NOT `scifi_news`** --
   scifi_news dispatches to `scifi_news_circuit` and returns before this block.
4. **Bug Bible coverage** -- mandatory per `CLAUDE.md:127-132`, not a "decision".
5. **Live legs** -- `public_domain` AND `shakespeare` via
   `scripts/otr_headless_canonical.ps1` **without `-NoReset`**. `RESULT SUCCESS` +
   `obs_publish OK` do NOT prove captions happened: `OTR_CaptionBurn` passes the
   uncaptioned video through on failure. Also require `OTR_CaptionBurn OK`, a
   non-empty `<episode>_captions.ass` with at least one `Dialogue:` event, and the
   post-fix audit at exit 0.

## The judgment call worth not re-litigating

Codex r2 recommended stripping BOTH `URL:` and `Date/Rights:` from the Shakespeare
prompt. **I kept `Date/Rights:` and removed only the URL.** That prompt explicitly
asks the model for a *"Folger/noncommercial source note"*, so the licence is INPUT
TO A REQUESTED OUTPUT, and `test_shakespeare_interpreter.py` pins it deliberately
("RIGHTS terms stay -- they are a licensing fact about the source"). The URL is
different: no instruction references it at all. Removing the rights string broke
that test in the full suite -- the suite caught what the panel missed.

## The full arc is on record

`kibitz-runs/2026-08-05-item7-citation/` -- r1..r4, each with `driver_anchor.md`,
`codex.md`, `antigravity.md`, `judgment.md`, `final.md`. **8 external reviews**
(Codex `gpt-5.6-sol` high + Antigravity x4). `r4/final.md` is the LOCKED plan.
`scope_receipt.md` records that r2's agy lane was quota-held and backfilled
through the UI, and that agy's r4 lane WROTE CODE instead of reviewing -- those
edits were reverted; the tree contains none of them.

Every round changed the build: r1 the blast radius (captions, not prompts), r2 the
ownership key, r3 the routing contract + the `NameError`, r4 a pre-existing
wrong-prompt bug and the schema bump.

## Parked elsewhere

A separate session's worktree
`.claude/worktrees/awesome-brahmagupta-a509b4` (branch
`claude/awesome-brahmagupta-a509b4`, at `acb09719`) holds UNCOMMITTED work
deleting the dead `news_coda_spoken_reduction` receipt chain and
`finalize_news_coda_surface`. It stood down so it would not collide with B4, which
restructures the same block. **Merge or re-run it AFTER Item 7 lands**, re-grounded
against the new helper boundary. Its reasoning is at
`kibitz-runs/2026-08-05-news-coda-receipt/`.

Housekeeping, unrelated: `git worktree list` shows 15 prunable July kibitz
worktrees. `git worktree prune` clears them.
