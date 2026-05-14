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

---

# ADDENDUM -- Cleanbreak audit for next-sprint planning (2026-05-13 post-S25 review)

**Audience:** Jeffrey + round-robin sprint consultant. **Goal:** make a complete, deliberate decision on what to delete next so the no-back-compat directive is fully honored before any new feature work (e.g. Two-Model Selector) begins.

**Scope of this addendum:** explicitly EXCLUDES the in-flight Two-Model Selector / Sprint #1 (B) scoping work in the working tree. That's after-cleanbreak by Jeffrey's direction.

**What prompted this addendum:** the S25 sprint repeatedly chose "gate the legacy path + schedule deletion for S26" over "delete the legacy path now." That posture is itself legacy-tolerance debris under the no-back-compat directive. The audit below catalogs every such surface so the consultant can decide which to delete in the next sprint vs which need a dedicated sprint vs which are explicitly out of scope.

## A. Items S25 LEFT as legacy-tolerant (highest priority -- these are debris the sprint itself created or perpetuated)

### A1. Legacy `ledger.sfx[]` writeback loop in `batch_audiogen_generator.py:701-765`

**Current state after S25:**
- The entire `Path 1: legacy ledger.sfx[]` loop survives.
- Phase 3 added a C2 ghost-path gate (`(save_ok or had_cache_hit) AND os.path.isfile`).
- Phase 3 added a `DeprecationWarning` on non-empty `sfx_rows`.
- Phase 3 stamps `sfx_render_status` on each legacy row.
- CD-3 audit (Phase 7) confirmed **zero current producers** populate `ledger["sfx"]`.

**Cleanbreak action:** delete the entire `Path 1` block:
- `sfx_rows = led_disk.get("sfx") or []` lookup
- the `if sfx_rows: warnings.warn(...)` DeprecationWarning
- the `for i, item in enumerate(render_queue):` parallel-index loop and all its writeback stamping
- the `updated_sfx_array` counter and the dual-stat log surface (`sfx_array=N/M, lines=N/M`) collapses to lines-only

**Why it wasn't done in S25:** the playbook had it as CD-3 audit -> "schedule deletion for S26." With CD-3 returning empty (zero producers), the correct cleanbreak action was deletion in this sprint, not gating + scheduling. The C2 gate I added is itself the kind of legacy-tolerance debris the directive forbids.

**Risk:** none if the audit is right. The full codebase grep returned zero producers. A real producer would now be surfaced by the DeprecationWarning before this lands, so the gate has done its alarm-plumbing job.

### A2. MusicGen `_find_cached` legacy timestamped-filename fallback (`musicgen_theme.py:230-261`)

**Current state after S25:** Lines 230-261 are a fallback for `<prefix>_<ts>.wav` cache files written by a "pre-Phase-D" implementation that no current code path produces. The full `legacy_prefix` / `matches` / `_legacy_sort_key` machinery runs on every cache miss.

**Cleanbreak action:** delete the whole legacy branch. Drop to a single-tier lookup -- canonical filename only. On a fresh install or fresh per-episode dir, the legacy lookup matches nothing anyway.

**Risk:** any leftover pre-Phase-D wavs in a long-running install's cache dir will rebuild on first run instead of being reused. Acceptable per the directive.

**Sibling for the consultant to consider:** `batch_audiogen_generator.py:144` carries the same pattern.

### A3. `production_ledger.py:357 "sfx": []` schema scaffold

**Current state after S25:** the empty-list schema field survives as a legacy-shape carryover. The L3 schema otherwise puts SFX rows on `ledger.lines[]` with `role="sfx"`.

**Cleanbreak action:** delete the `"sfx": []` line from the schema initializer. Verify (via grep before commit) that no consumer reads `ledger["sfx"]` after the legacy writeback path (A1) goes.

**Risk:** consumers using `ledger.get("sfx") or []` keep working (default-empty). Consumers using `ledger["sfx"]` would KeyError -- this is the verification step.

### A4. AudioGen + ProcSFX `script_json` default `"[]"` (legacy parser-list shape)

**Current state after S25:**
- `batch_audiogen_generator.py:251` -- `"script_json": ("STRING", {..., "default": "[]"})`
- `batch_procedural_sfx.py:115` -- same
- `musicgen_theme.py` -- `"{}"` (v2 ledger shape)

**Cleanbreak action:** change both `"[]"` defaults to `"{}"`. Matches MusicGen and the v2 ledger contract (`load_ledger` parses a JSON dict, not a list).

**Risk:** zero runtime risk -- the value is the empty-state default; both shapes parse to "no work" in `load_ledger`. Pure consistency cleanup.

## B. Items from the broader codebase audit (deferred mid-workflow legacy surfaces)

The S25 audit grep (`grep -rn 'back-compat|back_compat|backcompat' nodes/`) returned ~35 hits across 17 files. Triaged below by category. The consultant decides scope per sprint.

### B1. Multi-shim ledger I/O surfaces in `_otr_ledger.py` (4 sites)

Lines 27, 63, 166, 906 carry "back-compat with l2 ledgers" / "older ledgers that lack this" comments. The current ledger schema is L3 (`schema_version: "l3-2026-05-08"`).

**Cleanbreak question:** are L2 ledgers still produced anywhere? If no, the L2 fallbacks delete. Producer grep is the audit.

### B2. Multi-shim outline back-compat in `_otr_outline.py` (10+ sites)

Lines 138, 210, 289, 304, 376, 471, 483, 512, 725, 1026 + test fixtures at 1243, 1247, 1258. All "pre-Phase-2A budget" or "bare cast list" back-compat. The outline pipeline has moved through Phase 2A+; question is whether ANY current writer path takes the bare fallback.

**Cleanbreak question:** does the writer + outline still accept the bare format, or is it always the rich one? This needs a code-trace from `OTR_LedgerScriptWriter.generate()` through the outline call sites.

### B3. Production-ledger back-compat shims in `set_cast` (`production_ledger.py:167, 582, 597`)

Three "Back-compat input shim" comments. These accept old key shapes "on the way IN" to `set_cast`. With v2 ledger contract locked, any caller still using old keys is itself a legacy surface.

**Cleanbreak question:** what callers do `set_cast`? Are any using the old keys?

### B4. Line-composer back-compat (`_otr_line_composer.py:468, 856, 1215, 1492`)

`back-compat with callers that don't yet build a ...` and similar phrasing across four sites. Each is a defensive fallback for a caller pattern that may or may not still exist.

**Cleanbreak question:** are there still callers without the rich call signature? Producer-side audit.

### B5. Freeze-cascade back-compat (`_otr_ledger_freeze.py:275, 478, 665`)

- L275: outline beats fallback
- L478: speaker_role substitutes (was a back-compat shim)
- L665: dur_s absent / None (older ledger tolerance)

**Cleanbreak question:** can we tighten the freeze contract so these are required not optional?

### B6. Other back-compat surfaces (lower priority)

- `OTR_LedgerScriptWriter.py:776, 1951` -- seed_text shim, no-style-picked fallback
- `batch_humo_render.py:889, 2928` -- flat-dir patterns + direct stem match
- `otr_video_plan.py:645` -- `shot_id` alias kept for some callers
- `story_orchestrator.py:483, 3814` -- alias back-compat + skip=True legacy guard
- `scene_sequencer.py:939, 958` -- back-compat consumer notes
- `video_engine.py:664` -- voice_assignments-only cast fallback
- `video_composite.py:2183` -- audio_source back-compat alias
- `_otr_paths.py:204, 338` -- back-compat search root + function-name keepalive
- `post_audio_video_pipeline.py:124` -- flat layout (note: this node is RETIRED per __init__.py:176, so the back-compat is doubly dead)

Each is a 1-3 line surgical delete pending verification of zero callers.

## C. Items NOT in scope for the next sprint

### C1. Two-Model Selector / Sprint #1 (B)

In the working tree as uncommitted edits in `ROADMAP.md` + the untracked `docs/2026-05-13-two-model-selector-scoping.md`. **Excluded by Jeffrey's direction:** after-cleanbreak; do not plan as next sprint.

### C2. C8 / CastContract quarantine

CD-1 decision was Option 3 (drop quarantine, accept production-wired). Not a cleanbreak target; the module IS the production cast pipeline.

### C3. The big `_otr_outline.py` back-compat sweep (B2)

10+ sites; needs a dedicated sprint with budget-flow code-tracing, not a "tag along" in the next bug-fix sprint.

### C4. Survivor: forensic comments in BUG_LOG.md, ROADMAP.md, ADRs

These reference deleted classes by name (e.g. "OTR_Gemma4Director", "OTR_LLMDirector") in historical context. They are documentation of past state, not legacy-tolerance code. Leave alone.

## Recommended next-sprint package (consultant decides)

**Option D-MIN (focused cleanbreak completion -- low risk, high signal):**
- A1: delete legacy ledger.sfx[] loop entirely
- A2: delete MusicGen `_find_cached` legacy timestamped branch
- A2-sibling: delete AudioGen `_find_cached` legacy timestamped branch
- A3: delete `production_ledger.py:357 "sfx": []` (after verifying no consumer reads the key)
- A4: standardize AudioGen + ProcSFX `script_json` defaults to `"{}"`

Estimated scope: ~60 LOC net deletion, 2-4 tests updated, regression must hold at 2165+/8/6.

**Option D-WIDE (full cleanbreak sweep including ledger I/O + freeze tightening):**
- Everything in D-MIN
- B1: `_otr_ledger.py` L2 back-compat sweep (4 sites)
- B5: `_otr_ledger_freeze.py` shim tightening (3 sites)
- B3: `set_cast` shim removal in `production_ledger.py` (3 sites; pending caller audit)

Estimated scope: ~150 LOC net deletion, more tests touched, larger blast radius.

**Option D-MASSIVE (everything except C1/C3):**
- D-WIDE + B4 (line composer) + B6 (other back-compat sites)
- _otr_outline.py sweep (B2) deferred separately to its own dedicated sprint

Estimated scope: too large for a single bug-fix sprint. Split.

## Acceptance criteria for "cleanbreak complete for the legacy ledger"

The standing directive in ROADMAP.md lists these acceptance criteria (originally for commit 12.3):

1. `grep -rn "OTR_LedgerScriptReviewer" nodes/ __init__.py` returns ZERO hits outside forensic comments
2. `grep -rn "Gemma4" nodes/ __init__.py` returns zero hits
3. `grep -rn "reviewer_verdict" nodes/` returns zero hits
4. The workflow JSON loads in ComfyUI Desktop with NO missing-node warnings or back-compat aliases firing
5. Bug Bible regression holds 23/1/2xf

**All five hold at S25 close.** The legacy ledger CLASS / FIELD names are gone.

What remains is legacy-tolerance CODE that handles hypothetical inputs from extinct producers. That's the cleanbreak completion item — and it's exactly what this addendum is asking the consultant to plan.

## Round-robin question for the consultant

> The S25 sprint added a C2 gate + DeprecationWarning to the legacy `ledger.sfx[]` writeback loop, then scheduled deletion for S26. Per the no-back-compat directive, the correct action was deletion in S25. The sprint repeated this "gate + schedule" pattern across multiple surfaces. Going forward, when an audit confirms zero producers, what's the consultant's framing for "delete now" vs "gate now, delete next sprint"? The pattern shipped because the playbook framed it that way; the directive forbids it.

