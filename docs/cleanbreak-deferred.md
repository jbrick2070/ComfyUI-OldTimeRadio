# Voice-Path-Cleanbreak — Deferred Items

Tracking page for sprint tasks that have shipped enough mitigation
to lock the batch but remain open. One entry per deferral with the
reason, the mitigation already in place, and the unblock condition.

## S14.2 — Validator auto-invoke (DEFERRED, 2026-05-13)

**Status:** indefinitely deferred.
**Reason:** ComfyUI has no central Python-side workflow loader to wrap. The frontend parses JSON in JavaScript and dispatches per-node; there is no single chokepoint for `validate_workflow_contract()`.

**Mitigation already in place:**
- `tests/test_workflow_live_passes_validator.py` (S16.6) validates the production workflow JSON in CI.
- `tests/test_legacy_audit_clean.py` (S15.5.1) catches legacy Director-era surfaces repo-wide.

**If runtime validation is ever needed:** revisit as a ComfyUI frontend extension or an `OTR_WorkflowValidator` opt-in first-node. Both are larger than original S14.2 scope; plan as their own sprint.
