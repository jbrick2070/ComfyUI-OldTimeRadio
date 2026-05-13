# S25 QA Post-Mortem

**Sprint:** S25 -- MusicGen parity + soft-rollout flip + legacy gating
**Branch:** `s25-musicgen-parity`
**Stack head:** `8ecde26` (Phase 10 commit; final push hash matches this once FINAL phase lands)
**Base:** `98489da` (v2.0-alpha HEAD at sprint open)
**Date:** 2026-05-13

---

## Acceptance state

- [x] All P0 findings (P0-1..P0-6 from the playbook) closed
- [x] All P1 findings (P1-1..P1-3 from the playbook) closed
- [x] 3 decisions (CD-1..CD-3) resolved inline; audits attached
- [x] BUG_LOG: entries 211..220 added with general-lesson framing on each
- [x] ROADMAP: CURRENT WORK refreshed; Roadmap-only items + CD outcomes appended
- [x] No deferrals to S26 except the explicit CD-3 deletion line item

---

## Test deltas

- **New test files:** `tests/test_musicgen_parity.py` (6 tests), `tests/test_audiogen_legacy_gate.py` (6 tests), `tests/test_style_palette_drift.py` (5 tests).
- **Extended test files:** `tests/test_workflow_audio_widget_vectors.py` (+1 test: BUG-LOCAL-210 production-drift regression pin), `tests/test_procsfx_writeback_convention.py` (2 strict-default pins updated for the AG-9 flip).
- **Net new test functions:** +18 (2147 -> 2165 passed).
- **Regression:** 2165 passed / 8 skipped / 6 known-fail (baseline 2147/8/6 held).

### New tests by phase

| Phase | New tests | Files |
|---|--:|---|
| Phase 1 (style palette drift) | 5 | `test_style_palette_drift.py` |
| Phase 6 (acceptance) | 13 | `test_musicgen_parity.py` (6) + `test_audiogen_legacy_gate.py` (6) + `test_workflow_audio_widget_vectors.py` (+1) |

---

## LOC by file

```
 BUG_LOG.md                                  | 105 ++++-
 ROADMAP.md                                  | 122 ++++--
 docs/cleanbreak-deferred.md                 |  28 ++
 nodes/_otr_ledger_consumers.py              |  33 ++
 nodes/_otr_ledger_freeze.py                 |  18 +
 nodes/_otr_style_palette.py                 | 111 +++++
 nodes/batch_audiogen_generator.py           |  79 +++-
 nodes/batch_procedural_sfx.py               |  42 +-
 nodes/musicgen_theme.py                     | 632 ++++++++++++++++---------
 tests/test_audiogen_legacy_gate.py          | 144 +++++++
 tests/test_musicgen_parity.py               |  88 ++++
 tests/test_procsfx_writeback_convention.py  |  23 +-
 tests/test_style_palette_drift.py           |  87 ++++
 tests/test_workflow_audio_widget_vectors.py |  68 +++
 14 files changed, 1252 insertions(+), 328 deletions(-)
```

MusicGen carries the biggest line count change (the +/- on the same file reflects the inline `_STYLE_PALETTE` dict deletion + the parity-uplift block replacement; the actual net new MusicGen LOC is ~180 lines of fixed-up render logic + 110 lines of new test coverage).

---

## Decisions made inline (no round-robin needed)

### CD-1 (C8 CastContract quarantine) -- Option 3 selected

**Narrow grep audit (per playbook spec):**
```
$ grep -h 'from .*_otr_cast_contract import' nodes/*.py | sort -u
nodes/_otr_cast_repair.py:40:from nodes._otr_cast_contract import (
nodes/_otr_cast_repair.py:312:    from nodes._otr_cast_contract import _extract_dialogue_tags
```

Narrow rule would have pointed at Option 1 (extract helpers).

**Broader reference graph** (CastContract / detect_aliases / _extract_dialogue_tags consumers): `_otr_cast_repair.py`, `OTR_LedgerScriptWriter.py` (forensic reference), `_otr_outline.py`, `_otr_ledger.py`. Plus 4 test files.

**Why Option 3 over the narrow mechanical Option 1:** the standing no-back-compat directive forbids re-export shims, and the broader graph shows cast_contract is touched by 4 production modules + 4 test files. Option 1's "small in-sprint move" framing was based on the narrow grep alone; the deeper audit shows the scope is multi-module. Option 3 ("drop the quarantine plan, accept production-wired") honestly reflects what the codebase shows: cast_contract IS the production module for the cast pipeline; quarantining it was the wrong frame.

Audit + decision attached to `docs/cleanbreak-deferred.md` C8.

### CD-2 (IMP-46 retired LFC names) -- CLOSED EMPTY

Audit: deleted files are 3 test files + 1 wiring-smoke script. Zero retired production LFC class names. Current LFC node classes (`OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc`, `OTR_LedgerFreezeCascade`) are all live registrations -- they were never renamed in flight. **Rejection stands.** No additions land in `tests/test_legacy_audit_clean.py`.

### CD-3 (legacy `ledger.sfx[]` producers) -- SCHEDULE DELETION FOR S26

Audit returned only `production_ledger.py:357: "sfx": [],` -- the empty-list schema scaffold (consumer-side initialization, not a producer). No production code writes a non-empty `ledger.sfx[]`. **Scheduled for deletion in S26.X.** The C2 ghost-path gate landed on the legacy loop this sprint (S25/AG-2) is conservative belt-and-suspenders -- with zero producers, the gate fires zero times in production, but it keeps the contract honest until the deletion lands.

Full audit data in `ROADMAP.md` "CD-2 / CD-3 audit outcomes" section.

---

## Patterns reinforced

1. **Sibling-audit on Bible-pattern landings.** BUG-LOCAL-209 ("`-> None` on truthiness-consumed returns") was the S24 general lesson, but the audit didn't run beyond AudioGen. BUG-LOCAL-211 is the same defect in MusicGen. Going forward, every Bible-pattern entry must include a mandatory `git grep` audit across `nodes/` in the same commit -- not the next sprint.

2. **Parallel-path safety drift.** S24/C2 fixed the v2 `ledger.lines[]` writeback path but not the legacy `ledger.sfx[]` path that handles the same field. BUG-LOCAL-217. Going forward, safety fixes are paired with a `git grep <field>` audit and applied to every match in the same commit, not the next sprint.

3. **Soft-rollout deadlock.** Two features shipped with safety-net flags off and flip-criteria that referenced each other in an unreachable cycle. BUG-LOCAL-219. Going forward, any "soft rollout" defaults to soft for exactly one sprint with an inline flip-criterion AND a named owner; if the flip doesn't happen by sprint+1 the criterion is revisited as a P1 finding.

4. **Hoist on first drift.** Style-slug palette was maintained as two parallel lists in two files. BUG-LOCAL-216. Going forward, any data contract maintained as parallel lists in two files is hoisted to a shared module on first drift detection, with a pinned drift test.

5. **Ephemeral surfaces ship with their cleanup hook.** S24/C2's `_fallback/` redirect was a correct fix but shipped without GC. BUG-LOCAL-220. Going forward, when a fix introduces an "ephemeral" surface (cache dir, scratch file, temp dir), the cleanup hook lands in the same commit -- "we'll get to that later" cleanup hooks accumulate forever.

6. **Defender debris.** A silent runtime repair against a misconfiguration must be deleted when the misconfiguration is fixed at the root. BUG-LOCAL-218. Going forward, every "fix root cause" commit greps for downstream defenders against the original bug's symptom and prunes them in lockstep.

---

## Carry-forward to S26

- **Validator implementation (T1.2 from the master tracker):** S14.2 OTR_WorkflowValidator first-node, ~150 LOC; blocked on nothing; ready for S26 sprint open.
- **Legacy `ledger.sfx[]` deletion (CD-3 outcome):** scheduled for S26.X. Legacy parallel-index loop + `sfx_rows` lookup + DeprecationWarning + dual-stat log surface all delete in lockstep.
- **Per-consumer audit-walker strict-mode flip:** after 2 clean pipeline runs post-S25, flip each consumer's `audit_post_freeze_writeback(..., strict=False)` call to `strict=True`. Operator-driven; not a code change in S26.
- **Roadmap-only items:** 5 spillover items captured in `ROADMAP.md` "Roadmap-only items" section -- naming-conventions broadening, `_load_cached_wav` annotation, strict-mode flip pointer, C11 generalization, `script_json` default standardization. Fold into adjacent sprints when convenient.

---

## Files touched

```
BUG_LOG.md
ROADMAP.md
docs/cleanbreak-deferred.md
docs/2026-05-13-S25-qa-postmortem.md   (this doc)
nodes/_otr_ledger_consumers.py
nodes/_otr_ledger_freeze.py
nodes/_otr_style_palette.py            (new file)
nodes/batch_audiogen_generator.py
nodes/batch_procedural_sfx.py
nodes/musicgen_theme.py
tests/test_audiogen_legacy_gate.py     (new file)
tests/test_musicgen_parity.py          (new file)
tests/test_procsfx_writeback_convention.py
tests/test_style_palette_drift.py      (new file)
tests/test_workflow_audio_widget_vectors.py
```

---

## Verification artifacts (run all; all pass)

```bash
# Full regression
pytest -q --no-header -p no:cacheprovider -W ignore::DeprecationWarning
# Expected: 6 failed (EXPECTED_FAILED_NODEIDS / known-failures), 2165 passed, 8 skipped

# Bug Bible regression
pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q
# Expected: 23 passed, 1 skipped, 2 xfailed

# Phase 1 GATE
python -c "from nodes._otr_style_palette import KNOWN_STYLE_SLUGS, STYLE_PALETTE; print(len(KNOWN_STYLE_SLUGS))"
# Expected: 10
pytest tests/test_style_palette_drift.py -q
# Expected: 5 passed

# Phase 2 GATE -- P0-1 audit
grep -En 'def _save_wav.*-> *None' nodes/   # expected: 0 hits

# Phase 2 GATE -- NODE_CLASS_MAPPINGS prefix
python -c "from nodes.musicgen_theme import NODE_CLASS_MAPPINGS as M; assert 'OTR_MusicGenTheme' in M; assert 'MusicGenTheme' not in M; print(dict(M))"

# Phase 3 GATE
grep -n 'model_id.*in.*\["3"' nodes/batch_audiogen_generator.py
# Expected: forensic comment only (no active code lines)

# Phase 4 GATE
grep -n 'may be None\|stays None on the ledger' nodes/batch_procedural_sfx.py   # expected: 0 hits
grep -n 'strict_writeback.*True' nodes/batch_procedural_sfx.py                  # expected: 1+ hits

# Phase 5 GATE
grep -rn 'audit_post_freeze_writeback' nodes/ --include='*.py' | grep -v '_otr_ledger_consumers.py'
# Expected: 3 active call sites (AudioGen, MusicGen, ProcSFX)
python -c "from nodes._otr_ledger_consumers import ALLOWED_MUSIC_RENDER_STATUS; assert len(ALLOWED_MUSIC_RENDER_STATUS) == 6"

# Phase 6 GATE (acceptance tests)
pytest tests/test_musicgen_parity.py tests/test_audiogen_legacy_gate.py tests/test_workflow_audio_widget_vectors.py tests/test_style_palette_drift.py -q
# Expected: 24 passed
```
