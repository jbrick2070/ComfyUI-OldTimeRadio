# S26 Cleanbreak — Audit & Per-Item Results

Run start: 2026-05-13. Branch `s26-cleanbreak` cut from `s25-musicgen-parity` (HEAD `3393b39` — includes planning carry-along commit `3393b39` on top of `5369da4` cleanbreak audit addendum).

Baseline:
- pytest: 6 failed (known-fail set), 2165 passed, 8 skipped — see `baseline-pytest.txt`
- legacy footprint: 14 lines across 4 patterns — see `baseline-legacy-footprint.txt`
- known-fail nodeids: 6 — see `baseline-known-fail-nodeids.txt`

---

## Phase 1 — Section A deletes (results appended per item)

## A1 — Legacy ledger.sfx[] writeback loop deleted
- Commit: (pending)
- File: nodes/batch_audiogen_generator.py — Path 1 block + writeback loop + C2 ghost-path gate removed; dual-stat log surface collapsed to lines-only; comment header rewritten to v2-only.
- Test file removed: tests/test_audiogen_legacy_gate.py (6 tests pinning behavior on the deleted path)
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py -q`
- Result: 17 passed
- Unexpected failures: none
- Notes: AST parse clean. The `sfx_rows = led_disk.get("sfx") or []` lookup, `if sfx_rows: warnings.warn(...)` DeprecationWarning, and parallel-index `for i, item in enumerate(render_queue)` loop are all gone; only the v2 lines[] stamping path remains.

