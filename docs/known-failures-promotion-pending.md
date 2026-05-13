# S19.3 — Survival-Guide Promotion (PENDING MANUAL HANDOFF)

The known-failures hook pattern (S15.1 `pytest_sessionfinish` +
`EXPECTED_FAILED_NODEIDS`, extended in S19.1 to track setup / call /
teardown phases) is a strong candidate for promotion into the sibling
survival-guide repo at
`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`.

This file is the **pending-handoff marker**. The survival-guide
commit is intentionally NOT made from this repo's automation -- per
project rules the sibling-repo push is a manual handoff. Jeffrey
opens the survival-guide repo, lands the pattern doc + any
regression test, and cross-links back from OTR's
`docs/known-failures.md` Promotion section.

**Gate state (as of 2026-05-13):**
- S15.1 hook landed 2026-05-12 (commit `f813b37`).
- S19.1 phase-tracking extension landed 2026-05-13 (commit `32f62eb`).
- Plan requires 2-3 clean sprints of OTR-scoped use before promotion;
  only 1 sprint has fully passed since S15.1.

**Unblock condition:** 2-3 sprints with zero false-positive nodeid
surfaces in OTR usage AND no schema changes to
`EXPECTED_FAILED_NODEIDS`'s shape.

**When ready:**
1. In `comfyui-custom-node-survival-guide/`, create
   `patterns/known-failures-hook.md` documenting:
   - the hook structure (`pytest_sessionfinish` +
     `EXPECTED_FAILED_NODEIDS` frozenset)
   - the 80%-of-expected subset-coverage threshold
   - the setup/call/teardown phase tracking from S19.1
   - the `PROMOTABLE` banner mechanic
2. Cross-link from OTR's `docs/known-failures.md` Promotion section
   so any future reader of either side finds the canonical reference.
3. Delete this marker file in the same commit that ships the
   cross-link from OTR side.
