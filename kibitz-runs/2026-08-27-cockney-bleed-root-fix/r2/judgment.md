# R2 Judgment — Coding Plan / Implementability

Driver/panelist/judge: Codex
External reviewers: Cursor (Grok) and Claude Code (Claude)
Reviewer calls this round: 2

## Accepted and integrated

1. **Runtime container contract — CONFIRMED.** A mapping iterates string keys and would evade element-only validation. The plan now imports runtime `Sequence` from `collections.abc`, rejects scalar strings/bytes and every non-Sequence container, validates all elements before matching, and pins the exact predicate skeleton.
2. **Import/build failure — CONFIRMED.** `Sequence` is absent from the current helper imports. The plan now adds it explicitly and removes obsolete roster-shape typing imports.
3. **Dead test import — CONFIRMED.** Removing `roster_has_lemmy` without removing its top-level test import would fail collection before BUG-12.86 tests. The atomic edit is now explicit.
4. **Policy separator — CONFIRMED.** The live constant begins with `"\n\n"`. The plan now preserves that separator while replacing the constant's content.
5. **Constant-name ambiguity — CONFIRMED.** “Preserve” now means keep the name and replace its old unscoped text; tests count the canonical constant rather than maintaining raw duplicate policy text.
6. **Exact line retry hook — CONFIRMED.** `_recording_creative` is passed through `compose_line(creative_fn=...)` and captures assembled messages. The plan now names that parameter and the system-message index.
7. **Exact exchange hooks — CONFIRMED.** `_CountingGen` records messages but the existing `_raw_for` is tied to MARLOW/REESE; `_fake_gen_valid` produces dynamic slot output but records nothing. The plan now requires a local Lemmy fixture/raw response and a small recorder wrapper for prepass.
8. **Atomic signature/caller migration — CONFIRMED.** Both current callers are positional. The keyword-only signature and both tuple-valued call sites must land together.
9. **Bank roll false qualification — CONFIRMED.** Canonical source-bank slot 22 is `roll (any eligible bank)`. The plan now pins `media_archive` and forced Lemmy via real whitelisted `-Set` values.
10. **One-act reachability risk — CONFIRMED.** The live proof now uses the canonical three-act shape, increasing the opportunity for an accepted mixed group and avoiding repeated short false passes.
11. **Full-run grouping reconstruction — CONFIRMED.** Regrouping only accepted beat IDs can bridge a failed/singleton hole. The plan now groups the complete ordered run first, then retains only wholly accepted groups.
12. **Workflow no-diff check — CONFIRMED.** `git diff -- workflows/otr_canonical.json` is now an explicit frozen-diff gate.
13. **System-only test assertions — CONFIRMED.** Full-cast user context legitimately contains Lemmy's labeled Cockney signature. Tests now inspect only role=`system` for this defect.

## Rejected or refined

1. **Pin Gemma 4 E2B in the live command — REJECTED.** The current canonical graph names Gemma 4 12B, while the evidence ledgers used E2B. The defect is deterministic prompt scope, not a model regression. The qualification keeps the current canonical model and records the resolved IDs; E2B may be used only after resolving its current exact combo label.
2. **Use the runner's `--source-bank` flag directly — REFINED.** The sanctioned outer PowerShell wrapper does not expose a `-SourceBank` parameter. Its real interface is `[string[]] -Set`, which forwards whitelisted patches. The final command uses that actual wrapper contract.
3. **Inline the private predicate — REJECTED.** A named, exact predicate makes the category validation independently readable and prevents the matching branch from being interleaved with append behavior.
4. **Explicitly reject only `dict` — REJECTED in favor of the stronger real category.** Runtime `Sequence` rejects all mappings, sets, generators, and object containers consistently, avoiding another role/family-style partial classification.
5. **Claude's audit-field uncertainty — RESOLVED/CONFIRMED by the judge.** `OTR_LedgerScriptWriter.py:4932-4935` stamps `exchange_prepass_audit`, and four live ledgers were opened. The field exists but contains accepted beat IDs, not group speaker sets.
6. **Treat generic `Cockney` absence in the whole prompt as the integration assertion — REJECTED.** The labeled user CAST block can legitimately contain that word. The canonical policy constant's absence from the system message is the correct lock.

## Verify at build

- Execute the exact scalar/mapping/non-string sequence tests against the chosen runtime guard.
- Confirm PowerShell passes both quoted `-Set` array entries to the wrapper's `[string[]]$Set`.
- Confirm the final live ledger has a reconstructable accepted LEMMY+other exchange group.
- Record the canonical model IDs actually resolved by the live run.
