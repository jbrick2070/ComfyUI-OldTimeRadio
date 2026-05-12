# Known Test Failures — Quarantine List

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Last reconciled:** 2026-05-12 (voice-path-cleanbreak Sprint 1)
**Baseline pytest:** 2033 passed / 7 skipped / **6 failed**

This file lists every failing test the regression suite is allowed to
produce. Any failure NOT on this list is a real regression that blocks
merge.

When a quarantined test is fixed, mark it RESOLVED and remove from the
active list within the same commit.

---

## Active known failures (6 total)

### KNOWN-FAIL-001: `test_production_ledger::test_save_merges_schema_l3_fields_from_disk`

- **Error:** `KeyError: 'phase_ms'`
- **Symptom:** `production_ledger.save_ledger_safe`'s in-place merge of schema-l3 fields drops `meta.phase_ms` when the on-disk ledger predates the field's introduction.
- **Tracked:** voice-path-cleanbreak Sprint 1 audit (not voice-path-caused)
- **Owner:** TBD
- **Target sprint:** unscheduled (separate writer-internals cleanup)
- **First seen passing baseline:** pre-LFC sprint
- **First seen failing:** pre-voice-path-cleanbreak (carried through from prior sprints)
- **Reproduce:** `python -m pytest tests/test_production_ledger.py::TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk -v`

### KNOWN-FAIL-002: `test_save_to_episode_workspace::test_save_to_per_episode_dir_when_singleton_active`

- **Error:** `assert 0 == 1` (expected 1 PNG saved, got 0)
- **Symptom:** Save path produces `BAD_IMAGE_SAVE: ... 268 bytes (under 4096-byte gate); input tensor shape=(8, 8, 3) dtype=float32 min=nan max=nan`. Test fixture builds a tensor with NaN values; the save node correctly rejects it with the gate but the test expected a file to appear.
- **Tracked:** voice-path-cleanbreak Sprint 1 audit (not voice-path-caused)
- **Owner:** TBD
- **Target sprint:** unscheduled (test fixture needs valid tensor data, not NaN)
- **Reproduce:** `python -m pytest tests/test_save_to_episode_workspace.py::test_save_to_per_episode_dir_when_singleton_active -v`

### KNOWN-FAIL-003: `test_save_to_episode_workspace::test_portraits_role_routes_to_portraits_dir`

- **Error:** `assert 0 == 1` (same NaN-tensor pattern as KNOWN-FAIL-002)
- **Symptom:** identical to KNOWN-FAIL-002, different routing branch (portraits subdir).
- **Tracked / Owner / Target:** ditto KNOWN-FAIL-002.
- **Reproduce:** `python -m pytest tests/test_save_to_episode_workspace.py::test_portraits_role_routes_to_portraits_dir -v`

### KNOWN-FAIL-004: `test_save_to_episode_workspace::test_falls_back_to_legacy_dir_when_no_singleton`

- **Error:** `assert 0 == 1` (same NaN-tensor pattern)
- **Symptom:** identical to KNOWN-FAIL-002, exercising the no-singleton fallback path.
- **Tracked / Owner / Target:** ditto KNOWN-FAIL-002.
- **Reproduce:** `python -m pytest tests/test_save_to_episode_workspace.py::test_falls_back_to_legacy_dir_when_no_singleton -v`

### KNOWN-FAIL-005: `test_save_to_episode_workspace::test_per_episode_counter_starts_at_1`

- **Error:** `assert [] == ['full_env_00001_.png', 'full_env_00002_.png']` (same NaN-tensor pattern)
- **Symptom:** identical to KNOWN-FAIL-002, exercising the counter-increment path.
- **Tracked / Owner / Target:** ditto KNOWN-FAIL-002.
- **Reproduce:** `python -m pytest tests/test_save_to_episode_workspace.py::test_per_episode_counter_starts_at_1 -v`

### KNOWN-FAIL-006: `test_video_composite::test_default_canvas_is_native_832x480_at_25fps`

- **Error:** `assert 1472 == 832` (canvas width assertion drift)
- **Symptom:** Test expected the default canvas to be 832×480 (LTX 2B v0.9 native), but the current default emits 1472. Likely the canvas-size default was bumped during the HuMo / LTX 2.3 video-stack sprint without updating this test.
- **Tracked:** voice-path-cleanbreak Sprint 1 audit (not voice-path-caused)
- **Owner:** TBD
- **Target sprint:** unscheduled (video-stack consistency cleanup)
- **Reproduce:** `python -m pytest tests/test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps -v`

---

## Enforcement

The voice-path-cleanbreak sprint regression run produces exactly 6
failures. Adding a failure that doesn't have a `KNOWN-FAIL-NNN` ID
on this list is a regression that must be addressed (either fixed or
explicitly added with reason).

Sprint 2+ ship-gates enforce: if the regression run produces N failures
and N != len(known-failures list), the sprint is blocked until the
delta is reconciled.

Practical recipe:

```bash
python -m pytest tests/ --ignore=tests/integration -q 2>&1 | grep -E "^\d+ failed"
# Expected: "6 failed, 2033 passed, 7 skipped"
# If "6 failed" is anything else, audit before shipping.
```

---

## Resolution log (none yet)

When a KNOWN-FAIL-NNN is fixed:
1. Add a `[RESOLVED commit-hash YYYY-MM-DD]` line under the entry.
2. Move the entry to a "Resolved" section at the bottom of this file.
3. Decrement the active-count expectation at the top.

Keep resolution entries forever (deletion erases the history this file
is meant to track).
