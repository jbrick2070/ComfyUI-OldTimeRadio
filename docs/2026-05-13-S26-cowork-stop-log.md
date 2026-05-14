# Cowork autonomous run — stop log

## STOPPED — 2026-05-13 (S26 Sprint 4 / B2)

- **Trigger:** Design judgment required (directive §5 row 4)
- **Current state:**
  - Branch HEAD: latest s26-cleanbreak commit on local (Sprint 3 closed)
  - Sprints 1 / 2 / 3 fully executed; this stop is scoped to Sprint 4 only
  - Queue remaining at stop: just the B2 sweep itself (the run continues to §7 swim-return deliverables; see directive §5 "Stop and write a stop log entry when any of these hit. Do NOT push through; the run is paused, not failed.")
- **What needs Jeffrey:**
  - Design call on the pre-Phase-2A budget-bare-format path in `nodes/_otr_outline.py`: should the no-budget code paths (~22 sites at L17, L127, L138, L175-176, L272, L276, L288-289, L304, L471, L480, L483, L512, L643, L717, L725, L1025-1026, L1243-1247, L1258, L1349) be deleted (every outline path mandates a budget), or kept (bare-format remains a supported case)?
- **What Cowork already tried:**
  - Static grep audit (`git grep -nE "back-compat|back_compat|backcompat|legacy|pre-Phase|Phase 2A" nodes/_otr_outline.py`) returned 22 hits.
  - Reviewed each hit category:
    1. **Conditional behavior gated on `req.budget is None`** (L138, L725, L1025-1026): each path no-ops a Phase-2A validator when budget is None. Deleting requires every caller to supply a budget.
    2. **`cast_descriptions` empty-default fallback** (L471, L483, L1247): bare cast-line render when `cast_descriptions` is empty -- used by tests and "early-stage" call sites per the existing comments.
    3. **Optional widget fields** (L272-304): the "None preserves pre-Phase-2A back-compat for tests and early-stage" comment is explicitly behavioral, not just docstring narrative.
    4. **Self-tests** (L1243-1258, L1349): two self-test cases pin the bare-format / no-budget contract. Deleting them requires confirming no real caller still relies on bare-format.
  - Confirmed via `git grep "req.budget"` and `git grep "OutlineRequest"` that production callers DO supply budgets (Phase 2A landed and shipped), but `tests/test_phase2a_episode_budget.py` and a handful of fixture-level callers still exercise the no-budget path deliberately. Whether those tests are pinning legacy tolerance (delete) or representing a still-supported simpler-mode (keep) is the design call.
  - Per directive §4 Sprint 4: "If the budget-flow trace surfaces any ambiguity that requires Jeffrey's design judgment, STOP and write to the stop log. Do not guess." → stop.
- **Recommended named follow-up sprint:** "B2 `_otr_outline.py` budget-required cleanbreak" -- needs Jeffrey to lock the design call first (mandate budget vs. preserve bare-format) then mechanical execution per the Section B per-item loop.
- **Run resumption:** Cowork continues to §7 swim-return deliverables; B2 stays deferred until Jeffrey opens it.
