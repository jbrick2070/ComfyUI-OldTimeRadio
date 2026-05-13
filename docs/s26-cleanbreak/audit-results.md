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

## A2 — MusicGen `_find_cached` legacy timestamped branch deleted
- Commit: (pending)
- File: nodes/musicgen_theme.py — `_find_cached` collapsed to single-tier canonical-filename lookup (`<prefix>.wav`). Deleted: `legacy_prefix`, `matches`, `_legacy_sort_key`, iterdir loop, multi-match warning, sort + tail-select.
- Targeted test command: `pytest tests/test_musicgen_parity.py tests/test_musicgen_strict_failure.py -q`
- Result: 10 passed
- Unexpected failures: none
- Notes: The docstring also rewritten to remove the "Two-level lookup" framing and the Phase D consult note about glob metacharacters (the iterdir loop they referenced is gone).

## A2-sibling — AudioGen `_find_cached` legacy timestamped branch deleted
- Commit: (pending)
- File: nodes/batch_audiogen_generator.py — `_find_cached` collapsed to single-tier canonical-filename lookup (matches the A2 pattern).
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py -q`
- Result: 17 passed
- Unexpected failures: none
- Notes: Identical edit pattern to A2 — same iterdir loop, same `_legacy_sort_key`, same multi-match warning, same docstring framing. Now the AudioGen + MusicGen cache lookups share the same minimal "canonical exists? else None" surface.

## A3 — production_ledger.py `"sfx": []` schema scaffold deleted
- Commit: (pending)
- File: nodes/production_ledger.py — `"sfx": []` line removed from Ledger.__init__ schema initializer.
- Pre-delete audit (both quote styles):
    - `ledger["sfx"]` / `ledger['sfx']` -> 0 hits (no KeyError consumers)
    - `.get("sfx" ...)` -> 2 hits in nodes/scene_sequencer.py (L950, L1319), both `.get("sfx") or []` — default-empty semantics intact. (These are B6 surfaces and will be handled separately in Phase 3.)
- Test migration in-commit: tests/test_production_ledger.py::test_new_ledger_creates_structure dropped "sfx" from its expected-key tuple and added `assert "sfx" not in led.data` to pin the new contract.
- Targeted test command: `pytest tests/test_production_ledger.py tests/test_otr_ledger_consumers.py tests/test_procsfx_ledger.py -q --tb=no`
- Result: 1 failed, 82 passed
- Unexpected failures: none. The single failure (`TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk`) is in the baseline known-fail set — pre-existing, not introduced by A3.
- Notes: AST parse clean. `git grep -nE "['\"]sfx['\"]: \[\]" nodes/ tests/` -> 0 hits post-commit.

### A3 extension — required-list validator + test-fixture scrub (in-commit)
The initial A3 commit removed only the schema scaffold line. The post-commit zero-hit grep surfaced 17 test fixtures still constructing ledger dicts with `"sfx": [],`. Mechanically scrubbing them surfaced a deeper coupling:
- `nodes/_otr_ledger_freeze.py::_REQUIRED_TOP_LEVEL_LISTS` still required a top-level `sfx` list.
- `tests/test_lfc_phase_0_10_gap_audit.py::TestNullRejection` parametrized over that required-list including `sfx`.

Per directive ("downstream breakage is a feature; fix the caller, not the legacy code"), A3 extended to:
- Remove `"sfx"` from `_REQUIRED_TOP_LEVEL_LISTS`.
- Update the freeze module docstring to drop the now-removed top-level from the schema mapping (keep `ALLOWED_SPEAKER_ROLES` intact — line.speaker_role == "sfx" is the v2 contract).
- Remove `"sfx"` from the 3 parametrize lists in test_lfc_phase_0_10_gap_audit.py.
- Re-run the full LFC + ledger + silent-test-episode suite: 400 passed, 1 failed (pre-existing baseline known-fail).

Blast radius: 19 files (2 production + 17 tests). Below the §5 circuit-breaker bound. Architectural surface unchanged (no module boundary or class signature moves). Within scope of A3 -- the validator was the contractual mirror of the deleted schema scaffold.

## A4a — script_json node-class default "[]" -> "{}" (AudioGen + ProcSFX)
- Commit: (pending)
- Files:
  - nodes/batch_audiogen_generator.py:210 — `"default": "[]"` -> `"default": "{}"`
  - nodes/batch_procedural_sfx.py:115 — same
- Rationale: matches MusicGen (already `"{}"`) and the v2 ledger contract — `load_ledger` parses a JSON dict, not a list. The previous `"[]"` default failed silently when wired with no upstream (interpreted as an empty list, not the expected dict).
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py tests/test_procsfx_isolation.py tests/test_procsfx_ledger.py tests/test_procsfx_writeback_convention.py -q`
- Result: 32 passed
- Unexpected failures: none
- Notes: AST clean. A4b workflow-fixture audit (next item) propagates this default to saved widget vectors in the workflow JSONs.


