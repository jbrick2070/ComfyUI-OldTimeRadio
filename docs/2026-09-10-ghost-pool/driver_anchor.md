# Row 2.2: finish bounded Ghost admission-prompt allocation

Base: `7eb90476730150f26649b14a736a5addbf7080cd`, `v2.0-alpha`.
Driver: GPT-6 Astra, rung 5 coder/sole-judge role (the historical table names
Opus). Review routing: CLAUDE.md 2026-09-07, one local CLI finished-diff
review. Section 0 explicitly rules NO ARC for this closed specification.

## Grounded scope before code

`ghost_prompt_signature` already implements the finalized v2 admission key;
`_ghost_validate_batch` and replay already use it. Commits `a8e4d5a6` and
`27123086` shipped this part. The old plan's leaf-key implementation directions
are stale. The v3 render composer intentionally ignores the authored leaf;
do not redirect it or promise pixel variety from admission signatures.

`deterministic_leaf` still raises after all 18 same-mode clauses are spent.
`deterministic_batch` stores only a history of signatures, and ShotLock stamps
one fallback reason across the whole batch. These are the remaining build
items. Valid replay objects remain immutable, including their original reasons.

Implement the already-ruled progression: keep the existing seed/beat-keyed
probe order while a finalized prompt is unused; a leaf under another motif is
already a new prompt; after exhaustion choose the least-recent eligible
signature, excluding both adjacent slots. Replay needs its original ordinal
so future frozen rows reserve signatures without becoming fictional past uses.
Carry allocation dispositions outside the frozen ghost object and append each
to that beat's existing fallback_reason. No new stored field or hash version.

Structural errors remain loud. A broken constant pool that supplies no
nonadjacent candidate cannot be made total by hiding corruption; the shipped
18-clause pools must be tested to supply alternatives. No generator, motif
rewrite, act-scoped uniqueness, schema extension, new env read or process call.

## Verification

- More than 18 same-mode beats, deterministic LRU and per-beat receipts.
- Mixed replay/fresh batches with both replay neighbors; replay bytes retained.
- Writer acceptance/rejection and deterministic fallback use the same key.
- All stored objects retain the exact field set and pass validation.
- Short existing Ghost allocations and non-Ghost ShotLock output compared
  before/after; no canonical graph, recipe, profile or render-composer change.
- Focused tests, whole suite with explicit environment limits, Bug Bible.
- Independent CLI review grounded against the frozen diff before push.
- Live five-act canonical forced-Ghost publication remains the row's final
  acceptance gate; test-only success does not close that gate.

## Environment facts

This is the rented Mac M4. The haunted lane's registry row excludes MPS;
the pod stays stopped and the 5080 production loop is untouched. The baseline
suite launched a real Bark audition from a test that assumes Windows evidence
files exist. That test process and suite were stopped. An offline-only restart
still reached about 11.4 GB RSS around the cameo resolver tests and was also
stopped. Baseline and post-change runs therefore use empty model/cache roots,
disable model-network access, and deselect the three audition tests that expect
those unavailable real evidence files. Report this limit explicitly. These
are test-command settings, not changes to production configuration or models.
Bug Bible baseline: 10 failed, 29 passed, 11 skipped, 3 xfailed.
