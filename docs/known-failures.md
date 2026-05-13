# Known Test Failures — Quarantine List

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Last reconciled:** 2026-05-12 (voice-path-cleanbreak Sprint 15.2)
**Baseline pytest:** **6 failed**, 2096+ passed, 6 skipped
**Enforcement:** `tests/conftest.py::EXPECTED_FAILED_NODEIDS` (S15.1
hook) tracks the failure SET, not just the count. Adding a failure
not in that set fires `[KNOWN-FAIL-GUARD] NEW failures (REGRESSION)`
and exits with code 2; a known failure that starts passing fires
`[KNOWN-FAIL-GUARD] PROMOTABLE`.

This file lists every nodeid the regression suite is allowed to
produce. The list MUST stay in lockstep with
`tests/conftest.py::EXPECTED_FAILED_NODEIDS` -- when one moves, the
other moves in the same commit.

When a quarantined test is fixed:
1. Remove its entry from `EXPECTED_FAILED_NODEIDS` in `conftest.py`.
2. Move the entry below to the "Resolved" section with the fixing
   commit hash + date.
3. Decrement the "Baseline pytest" failure count above.

---

## Schema (S15.2 nodeid-tracking)

Each entry below uses the schema:

```
KNOWN-FAIL-NNN
  Nodeid:        <pytest collection ID -- copy-paste reproducible>
  First seen:    <commit-sha or sprint name when first added>
  Expected mode: <error class + brief signature>
  Owner:         <name or TBD>
  Removal cond:  <what has to land before the test can be promoted>
  Reproduce:     <one-line pytest command>
```

The `Nodeid` field is the load-bearing one -- the conftest hook
matches by exact string against pytest's collected `item.nodeid`.

---

## Active known failures (6 total)

### KNOWN-FAIL-001

```
Nodeid:        tests/test_production_ledger.py::TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk
First seen:    pre-voice-path-cleanbreak (carried from prior sprints)
Expected mode: KeyError: 'phase_ms'
Owner:         TBD
Removal cond:  production_ledger.save_ledger_safe stops dropping
               meta.phase_ms when the on-disk ledger predates the
               field's introduction. Tracked: writer-internals
               cleanup (unscheduled).
Reproduce:     python -m pytest tests/test_production_ledger.py::TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk -v
```

### KNOWN-FAIL-002

```
Nodeid:        tests/test_save_to_episode_workspace.py::test_save_to_per_episode_dir_when_singleton_active
First seen:    voice-path-cleanbreak Sprint 1 audit (not voice-path-caused)
Expected mode: AssertionError: assert 0 == 1 (expected 1 PNG saved, got 0)
               -- save node correctly rejects NaN-laden tensor with
               BAD_IMAGE_SAVE gate; test fixture is the broken party.
Owner:         TBD
Removal cond:  Test fixture switches from NaN-init tensor to valid
               image data. The save node's gate is correct; the test
               is wrong. Tracked: test-fixture cleanup (unscheduled).
Reproduce:     python -m pytest tests/test_save_to_episode_workspace.py::test_save_to_per_episode_dir_when_singleton_active -v
```

### KNOWN-FAIL-003

```
Nodeid:        tests/test_save_to_episode_workspace.py::test_portraits_role_routes_to_portraits_dir
First seen:    voice-path-cleanbreak Sprint 1 audit
Expected mode: AssertionError: assert 0 == 1 (same NaN-tensor pattern as 002)
Owner:         TBD
Removal cond:  Same fix as KNOWN-FAIL-002 (NaN tensor in fixture).
Reproduce:     python -m pytest tests/test_save_to_episode_workspace.py::test_portraits_role_routes_to_portraits_dir -v
```

### KNOWN-FAIL-004

```
Nodeid:        tests/test_save_to_episode_workspace.py::test_falls_back_to_legacy_dir_when_no_singleton
First seen:    voice-path-cleanbreak Sprint 1 audit
Expected mode: AssertionError: assert 0 == 1 (same NaN-tensor pattern)
Owner:         TBD
Removal cond:  Same fix as KNOWN-FAIL-002.
Reproduce:     python -m pytest tests/test_save_to_episode_workspace.py::test_falls_back_to_legacy_dir_when_no_singleton -v
```

### KNOWN-FAIL-005

```
Nodeid:        tests/test_save_to_episode_workspace.py::test_per_episode_counter_starts_at_1
First seen:    voice-path-cleanbreak Sprint 1 audit
Expected mode: AssertionError: assert [] == ['full_env_00001_.png', 'full_env_00002_.png']
               (same NaN-tensor pattern; counter-increment path)
Owner:         TBD
Removal cond:  Same fix as KNOWN-FAIL-002.
Reproduce:     python -m pytest tests/test_save_to_episode_workspace.py::test_per_episode_counter_starts_at_1 -v
```

### KNOWN-FAIL-006

```
Nodeid:        tests/test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps
First seen:    voice-path-cleanbreak Sprint 1 audit (HuMo / LTX 2.3
               video-stack default-canvas drift)
Expected mode: AssertionError: assert 1472 == 832 (canvas width drift)
Owner:         TBD
Removal cond:  Either (a) the default canvas reverts to 832x480, or
               (b) the test updates to expect 1472. The video-stack
               sprint consensus determines which.
Reproduce:     python -m pytest tests/test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps -v
```

---

## Enforcement (S15.1 conftest hook)

`tests/conftest.py::EXPECTED_FAILED_NODEIDS` is the source of truth
for the conftest hook. The list above and that set MUST stay in
lockstep -- when adding or removing a known-fail, both files move
in the same commit.

The `pytest_sessionfinish` hook in conftest.py runs after the suite
finishes and:

1. Builds the actual-failed-nodeid set from
   `item.rep_call.failed`.
2. Diffs against `EXPECTED_FAILED_NODEIDS`.
3. If any nodeid in `actual_failed - expected` (NEW failure ->
   regression): prints `[KNOWN-FAIL-GUARD] NEW failures` and exits
   with code 2.
4. If any nodeid in `expected - actual_failed` (PROMOTABLE -> a
   known-fail is now passing): prints `[KNOWN-FAIL-GUARD]
   PROMOTABLE` so the contributor can promote it.

Subset-run guard: the diff only fires when at least 80% of the
expected nodeids were actually collected. This means a focused
``pytest tests/test_xyz.py`` won't fire PROMOTABLE on every other
known-fail it didn't run.

---

## Resolution log (none yet)

When a KNOWN-FAIL-NNN is fixed, move it here with:

```
KNOWN-FAIL-NNN  [RESOLVED <commit-sha> YYYY-MM-DD]
  ... (original entry preserved)
  Resolution:    <what landed; commit subject is fine>
```

Keep resolution entries forever -- deletion erases the history this
file exists to track.

---

## Promotion to survival guide (S15.3, deferred)

The S15.1 hook + nodeid-tracking pattern stays OTR-scoped for at
least 2-3 sprints of active use before promotion to the
`comfyui-custom-node-survival-guide` repo. The deferral is
deliberate: the schema is new, false-positive modes haven't surfaced
yet, and the survival guide should publish patterns that have a
track record. S15.3 is its own commit when the threshold lands.
