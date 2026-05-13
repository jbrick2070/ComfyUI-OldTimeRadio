# Voice-Path-Cleanbreak — Deferred Items

Tracking page for sprint tasks that have shipped enough mitigation
to lock the batch but remain open. One entry per deferral with the
reason, the mitigation already in place, and the unblock condition.

## C8 — CastContract quarantine (DEFERRED, 2026-05-13)

**Status:** deferred. The plan's premise was wrong.

**Reason:** the C8 plan said "Update any internal imports (likely none)" when scoping the move from `nodes/_otr_cast_contract.py` to `nodes/experimental/_otr_cast_contract.py`. The dependency audit at execution time found:

- `nodes/_otr_cast_repair.py:40,312` imports from `_otr_cast_contract` (CharacterEntry, _extract_dialogue_tags, others)
- `nodes/_otr_cast_repair.py` is consumed by `nodes/_otr_ledger_reviewer.py::apply_deterministic_cast_repairs` (live production code path)
- `nodes/_otr_ledger.py:897` + `nodes/_otr_line_composer.py:740` carry forensic references

Cast contract IS wired into production via the `cast_repair → ledger_reviewer` chain. Quarantining to `experimental/` without first untangling those imports would either break `apply_deterministic_cast_repairs` (which IS called at writer-time) or ship a docstring lie ("not wired into production" when it IS).

**Mitigation already in place:** none specific to this. The repair path holds invariants via its own tests (`tests/test_cast_repair.py`, `tests/test_phase3_ledger_reviewer.py`).

**Unblock condition:** one of:
1. Delete the `cast_repair → cast_contract` dependency (move the small helpers cast_repair needs into a separate `_otr_cast_helpers.py`, then quarantine cast_contract clean).
2. Quarantine the full chain `cast_contract + cast_repair + apply_deterministic_cast_repairs` together (large scope; needs a real design call).
3. Accept that cast contract is production-wired and drop the quarantine plan. Update the C8 plan-spec docstring to reflect this.

This is a real architectural call, not a mechanical move. Plan as its own sprint.

---

## S14.2 — Validator auto-invoke (DEFERRED, 2026-05-13)

**Status:** indefinitely deferred.
**Reason:** ComfyUI has no central Python-side workflow loader to wrap. The frontend parses JSON in JavaScript and dispatches per-node; there is no single chokepoint for `validate_workflow_contract()`.

**Mitigation already in place:**
- `tests/test_workflow_live_passes_validator.py` (S16.6) validates the production workflow JSON in CI.
- `tests/test_legacy_audit_clean.py` (S15.5.1) catches legacy Director-era surfaces repo-wide.

**If runtime validation is ever needed:** revisit as a ComfyUI frontend extension or an `OTR_WorkflowValidator` opt-in first-node. Both are larger than original S14.2 scope; plan as their own sprint.
