# Known Test Failures — Quarantine List

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Last reconciled:** 2026-05-21 (full-walk 27-failure triage: 25 fixed, 2 quarantined)
**Baseline pytest:** **2 failed** (KNOWN-FAIL-007/008, quarantined below), 2460 passed/skipped of 2462 collected
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

## Active known failures (2)

KNOWN-FAIL-001 through 006 were promoted on 2026-05-13 (see the
Resolution log below). KNOWN-FAIL-007 + 008 were added 2026-05-21
during the full-walk 27-failure triage: 25 of the 27 failures were
real fixes; these last 2 are a product/licensing decision and are
quarantined pending that decision.

```
KNOWN-FAIL-007
  Nodeid:        tests/test_default_workflow_validator.py::test_default_workflow_validator_passes_on_shipped_default
  First seen:    2026-05-21 full-walk triage; failure predates 6a525f8
                 (present at clean HEAD 02fde67).
  Expected mode: AssertionError -- check_default_workflow_creative_binding
                 reports the shipped workflow's writer node (id=1)
                 binds 'google/gemma-4-E4B-it' (license_audit_status=
                 'pending'); the Sprint D / D3 creative-binding gate
                 requires 'mit_equivalent' for any default-shipped
                 binding.
  Owner:         Jeffrey
  Removal cond:  Resolve the writer-model licensing question, then
                 promote. Either (a) complete the google/gemma-4-E4B-it
                 license audit and update that row's
                 license_audit_status in nodes/_otr_model_catalog.py,
                 or (b) rebind the default workflow's
                 OTR_LedgerScriptWriter (node id=1) creative + technical
                 model slots to a mit_equivalent catalog row (the
                 catalog DEFAULT_LLM is mistralai/Mistral-Nemo-
                 Instruct-2407, which this test's docstring already
                 expects) and re-verify. NOTE: pyproject.toml still
                 describes the project as "Gemma 4 scripts" -- this is
                 a genuine product-direction call, not a drift fix.
  Reproduce:     pytest "tests/test_default_workflow_validator.py::test_default_workflow_validator_passes_on_shipped_default"
```

```
KNOWN-FAIL-008
  Nodeid:        tests/test_model_catalog_schema.py::test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot
  First seen:    2026-05-21 full-walk triage; failure predates 6a525f8
                 (present at clean HEAD 02fde67).
  Expected mode: AssertionError -- same root cause as KNOWN-FAIL-007:
                 the shipped default workflow binds
                 'google/gemma-4-E4B-it' (license_audit_status=
                 'pending') to the creative slot; the gate requires
                 'mit_equivalent'. This test asserts the gate from the
                 model-catalog-schema entry point rather than the
                 workflow-validator one.
  Owner:         Jeffrey
  Removal cond:  Same as KNOWN-FAIL-007. Both tests assert the same
                 creative-binding gate and will go green together once
                 the licensing question is resolved -- promote both in
                 the same commit.
  Reproduce:     pytest "tests/test_model_catalog_schema.py::test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot"
```

Adding a new known failure requires updating both this file and
`tests/conftest.py::EXPECTED_FAILED_NODEIDS` in the same commit, and the
fix's removal-condition must be specific enough that the next reviewer
can independently judge whether the condition has been met.

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

## Resolution log

```
KNOWN-FAIL-001  [RESOLVED ba8a02e 2026-05-13]
  Nodeid:        tests/test_production_ledger.py::TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk
  First seen:    pre-voice-path-cleanbreak
  Expected mode: KeyError: 'phase_ms'
  Resolution:    Real production bug, not a test issue. BUG-LOCAL-018
                 (7c84ee8) added meta.paths to the in-memory ledger
                 initializer; after that, in_mem["meta"] was always
                 non-empty at save() time, making the bulk-replace rule
                 in Ledger._merge_with_disk silently drop disk-side
                 meta.phase_ms.bark / git_commit / etc. Fixed by
                 splitting meta out of TOP_PRESERVE and giving it its
                 own per-key recursive-merge clause. ba8a02e.
```

```
KNOWN-FAIL-002..005  [RESOLVED a70aeb8 2026-05-13]
  Nodeids:       tests/test_save_to_episode_workspace.py::test_save_to_per_episode_dir_when_singleton_active
                 tests/test_save_to_episode_workspace.py::test_portraits_role_routes_to_portraits_dir
                 tests/test_save_to_episode_workspace.py::test_falls_back_to_legacy_dir_when_no_singleton
                 tests/test_save_to_episode_workspace.py::test_per_episode_counter_starts_at_1
  First seen:    voice-path-cleanbreak Sprint 1 audit
  Expected mode: AssertionError 0 == 1 / [] == [...] from BAD_IMAGE_SAVE
                 4096-byte gate tripping on the 8x8 fixture
  Resolution:    Shared fixture _fake_image_tensor() bumped from 8x8
                 (~268 byte PNG, under the gate) to 128x128 (~30-40 KB,
                 well over the gate). The diagnostic min/max=nan in
                 the BAD_IMAGE_SAVE message was the diagnostic's
                 fallback for non-torch tensors, not actual NaN data
                 in the fixture -- the prior KNOWN-FAIL entries
                 misattributed the failure to NaN content. The fixture
                 was always valid random float data, just sized for
                 the pre-gate world. a70aeb8.
```

```
KNOWN-FAIL-006  [RESOLVED 8181950 2026-05-13]
  Nodeid (old):  tests/test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps
  Nodeid (new):  tests/test_video_composite.py::test_default_canvas_is_layered_1472x832_at_25fps
  First seen:    voice-path-cleanbreak Sprint 1 audit
  Expected mode: AssertionError: assert 1472 == 832
  Resolution:    Test migration (b) per the original removal-condition.
                 BUG-LOCAL-030 (5a71f83) bumped canvas to layered
                 1472x832 so HuMo 1280x720 + LTX 1216x704 + FLUX env
                 backdrop fit natively without pillarbox at composite;
                 humo_target_height moved 480 -> 832 in lockstep.
                 Test was renamed and assertions updated to the
                 current contract. 8181950.
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
