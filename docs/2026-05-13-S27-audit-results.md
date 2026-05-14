# S27 Cleanbreak Tail — audit results

**Branch:** `s27-cleanbreak-tail` cut from `s26-cleanbreak` HEAD `19cf286`
**Posture:** DELETE NOW. THE NEW LEDGER IS THE ONLY LEDGER. NO BACK-COMPAT FOR OLD JSON, NO BACK-COMPAT FOR OLD ON-DISK LEDGERS.

The cut point `19cf286` is post-Phase-B downstream sweep (5 missed-regression
fix commits landed against `s26-cleanbreak` between the original 5bf9d3a
S26 close and this S27 cut). Baseline-pytest is the clean 2159 passed / 8
skipped / 0 failed state. Known-fail set is empty.

## Pre-tail baseline (Phase 0)

```
baseline-pytest.txt              copy of post-triage-baseline.txt
baseline-known-fail-nodeids.txt  empty (no quarantined failures)
baseline-footprint.txt           3 surfaces confirmed present:
                                   - OTR_PostAudioVideoPipeline registered
                                     at __init__.py:176 + file
                                     nodes/post_audio_video_pipeline.py:2
                                   - production_ledger.py:810 def set_sfx
                                   - production_ledger.py:865 def apply_sfx_timings
                                   - production_ledger.py:1058 "sfx": "cue_id"
                                     in _merge_with_disk::ROW_KEYED
```

All three Phase 1 deletion targets are present — work queue proceeds.

## Per-item result log

### Item 1 — Delete `OTR_PostAudioVideoPipeline` entirely

| Surface | Action | Verification |
|---|---|---|
| `nodes/post_audio_video_pipeline.py` | DELETED (420 lines) | `test ! -f` confirmed |
| `tests/test_post_audio_video_pipeline.py` | DELETED (14 tests) | -14 from pass count, math checks |
| `__init__.py:176` registration entry + back-compat justification comment | DELETED, replaced with forensic deletion comment | grep `OTR_PostAudioVideoPipeline` in `__init__.py` returns only the forensic comment |
| `nodes/_workflow_validation.py::DELETED_NODE_TYPES` | EXTENDED with `OTR_PostAudioVideoPipeline` entry | Workflows that still reference the type now fail-loud via `WorkflowDeletedNodeError` rather than silent load |
| `README.md` node table | REMOVED node-11 row | User-facing docs no longer advertise the retired node |
| `workflows/*.json` | NO scrub needed | Pre-delete `git grep -l "OTR_PostAudioVideoPipeline" workflows/` was already zero hits (S26 cleanup removed it from the canonical workflow) |
| `scripts/_apply_*_pipeline*` (3 one-shot migration scripts) | NOT TOUCHED (out of directive scope) | String-only references inside JSON node shape; no import-time fragility. Noted for an S28 scripts/ audit. |

**Verification grep result:** `git grep -n 'OTR_PostAudioVideoPipeline\|PostAudioVideoPipeline' nodes/ __init__.py` returns 2 hits — both intentional:

  - `__init__.py:169` is the forensic comment recording the S27 deletion (the "comments in BUG_LOG/ROADMAP are fine" tolerance read broadly applies to inline source forensic comments too).
  - `nodes/_workflow_validation.py:73` is the load-bearing `DELETED_NODE_TYPES` registry entry; deleting it would defeat the purpose (it's the safety net for old workflow JSONs).

**Targeted regression:** `pytest tests/ -q -k 'not test_audiogen_legacy_gate' -W ignore::DeprecationWarning` → 2145 passed, 8 skipped. Diff from baseline (-14) accounts exactly for `tests/test_post_audio_video_pipeline.py` (14 tests). Zero unexpected fails. No `[KNOWN-FAIL-GUARD]` lines.

**Commit:** `412781f` cleanbreak(s27-1): delete OTR_PostAudioVideoPipeline entirely

### Item 2 — Delete `set_sfx`, `apply_sfx_timings`, ROW_KEYED `"sfx"` entry

| Surface | Action | Verification |
|---|---|---|
| `nodes/production_ledger.py::set_sfx` (~L810, 14 lines) | DELETED, replaced with one forensic comment | grep returns only the comment |
| `nodes/production_ledger.py::apply_sfx_timings` (~L865, 9 lines) | DELETED | grep returns only the comment |
| `nodes/production_ledger.py::_merge_with_disk::ROW_KEYED["sfx"]` (~L1042) | DELETED — ROW_KEYED shrank from 4 entries to 3 (lines, clips, music) | grep `"sfx"\s*:\s*"cue_id"` returns 0 hits |
| `tests/test_production_ledger.py::TestTimingBackfill::test_apply_sfx_and_music_timings` | SPLIT — sfx half deleted, music half kept and renamed to `test_apply_music_timings` (the contract under test is still alive for music) | one test method instead of one mixed test |
| `tests/test_production_ledger.py::TestDualLedgerFix::test_save_preserves_disk_rows_when_in_mem_array_empty` | MIGRATED — example array switched from sfx to music (was using sfx purely as a convenient sample; the contract is about ROW_KEYED merge behavior in general, which still holds for music/lines/clips) | test passes, contract unchanged |

**Verification grep result:**

```
git grep -n 'set_sfx|apply_sfx_timings' nodes/ tests/
  -> nodes/production_ledger.py:810      forensic comment only
     tests/test_production_ledger.py:480 forensic comment only
     tests/test_production_ledger.py:481 forensic comment only

git grep -nE '"sfx"\s*:\s*"cue_id"' nodes/
  -> nodes/production_ledger.py:1038    forensic comment only
```

All non-comment occurrences are gone. Forensic comments preserve the
deletion trail per directive policy.

**Targeted regression:** `pytest tests/test_production_ledger.py tests/test_audiogen_ledger.py -q` → 42 passed (was 38 + 4). Zero failures, zero `[KNOWN-FAIL-GUARD]` lines.

**Commit:** `4da8669` cleanbreak(s27-2): delete production_ledger sfx surfaces

### Phase 3 — Strip dead sfx-mirror walk from `scene_sequencer.py` (QA-6)

| Surface | Action | Lines deleted |
|---|---|---|
| `_ledger_sfx = _led.get("sfx") or []` + the `for _sfx_idx ...` walk + the `_sfx_to_mirror_into_lines` mirror block | DELETED, replaced with one forensic comment | ~110 lines (the BUG-LOCAL-107 sfx writeback + ROADMAP P0 step 4b mirror-into-lines block) |
| `legacy_sfx_array_positioned=%d/%d` field on the SceneSequencer log line | DELETED | 2 |
| `_shifted_sfx` counter + EpisodeAssembler sfx-shift loop | DELETED (lines now carry speaker_role="sfx" and are shifted by the single lines[] loop above) | ~12 |
| `+ %d sfx` field on the EpisodeAssembler shift log line | DELETED | 2 |

After A3 deleted the sfx[] schema scaffold, `_led.get("sfx") or []` was always `[]`. The `if _sfx_idx >= len(_ledger_sfx): break` guard tripped on iteration 0 every run -- the walk had been a permanent no-op since S26.

**Verification grep result:**

```
git grep -n '_ledger_sfx|_sfx_idx|_shifted_sfx|_sfx_matched|legacy_sfx_array_positioned|_sfx_to_mirror_into_lines' nodes/
  -> 1 hit, only the forensic deletion comment

git grep -n -F '_led.get("sfx"' nodes/
  -> 1 hit, only the forensic deletion comment
```

**Targeted regression:** `pytest tests/ -q -k 'scene_sequencer or sequencer or episode_assembler'` → 19 passed, 2134 deselected. `tests/test_audio_byte_identical.py` (the "audio is king" gate) included and passes.

**Commit:** included with Phases 2 + 4 below.

### Phase 2 — Sibling-pattern enumeration closures

#### QA-2 — `_load_ledger` shim deletion (inline)

Both shims had zero production callers; deletion inline.

| Site | Pre-S27 | Action | Post-S27 callers |
|---|---|---|---|
| `nodes/video_composite.py:382` `_load_ledger` | 7-line shim wrapping `_load_ledger_with_path` | DELETED | 3 test callers MIGRATED to `_load_ledger_with_path(x)[0]` |
| `nodes/batch_humo_render.py:2805` `_load_ledger` | 10-line shim wrapping `_load_ledger_with_path` | DELETED | zero callers (no test migration needed) |
| `nodes/batch_ltx_render.py:2022` `_load_ledger` | Returns `tuple[dict, Path \| None]` -- this is the WITH-PATH function under a different name, NOT a shim | NOT TOUCHED | Naming inconsistency only; out of QA-2 scope |

**Verification:** `git grep -nE "def _load_ledger\\b" nodes/` returns only `batch_ltx_render.py:2022` (the tuple-returning function -- intentional).

**Targeted regression:** `pytest tests/test_video_composite.py tests/test_batch_humo_render.py tests/test_batch_ltx_render.py -q` → 92 passed.

#### QA-3 — `shot_id` envelope alias deletion (inline)

Both alias sites deleted in lockstep. Two production consumers and one test caller were migrated to read the canonical `frame_id` key (the alias copied frame_id verbatim, so values are unchanged).

| Site | Action |
|---|---|
| `nodes/otr_shot_duration_calculator.py:287` -- `"shot_id": frame_id` envelope-token alias | DELETED |
| `nodes/otr_video_plan.py:645` -- sibling alias on the same envelope shape | DELETED |
| `nodes/otr_video_plan.py:891` -- summary-line consumer using `tok['shot_id']` | MIGRATED to `tok['frame_id']` |
| `nodes/otr_video_plan.py:926` -- ledger.shots[] writeback consumer using `tok.get("shot_id")` | MIGRATED to `tok.get("frame_id")` (preserves prior value semantics; fallback `f"sh{idx+1:02d}"` retained) |
| `tests/test_otr_video_plan.py:383` -- `assert tok["shot_id"]` | MIGRATED to `assert tok["frame_id"]` |
| `tests/test_otr_video_plan.py:488-489` -- `[t["shot_id"] for t in plan["tokens"]]` + uniqueness check | MIGRATED + renamed `test_build_shot_plan_shot_ids_unique` → `test_build_shot_plan_frame_ids_unique` |

Remaining `shot_id` references in these two files are NOT aliases:
  - Reads from `plan["shots"]` registry entries (those entries genuinely have a `shot_id` key as their canonical name)
  - `starts_shot["shot_id"]` / `ends_shot["shot_id"]` reads on a DIFFERENT dict shape
  - `shot_id` keys built for `char_portrait`, `scene_env`, `start/end` token types (those have their own canonical shot_id values, not the deleted alias)

**Verification:** `git grep -nE '"shot_id"\s*:\s*frame_id' nodes/` → zero hits.

**Targeted regression:** `pytest tests/test_otr_video_plan.py tests/test_otr_shot_duration_calculator.py tests/test_render_flux_batch.py tests/test_render_humo_batch_plan.py -q` → 110 passed.

**Lesson learned:** my first audit pass claimed "zero consumers." The full grep + a hands-on `node.plan()` invocation found two production consumers (L891, L926) plus three test consumers. The "zero consumers" check must include `tok.get("shot_id")` patterns and not just `tokens["shot_id"]` patterns -- captured implicitly in the BUG-LOCAL-222 audit-completeness lesson.

#### QA-4 — `otr_legacy_audio_dir()` caller enumeration (deferred with full inventory)

13 caller sites enumerated in `docs/2026-05-13-S26-audit-results.md` under "B6/post_audio_video_pipeline -- legacy flat-layout fallback removed" (extension subsection). The deferral entry is in the S26 audit doc per the directive's instruction -- this is closing a known gap in the prior deferral, not a new S27 surface.

Pattern in every caller: `otr_legacy_audio_dir()` is the SECONDARY entry in an auto-pick fallback list, after `otr_episodes_root()` or `otr_audio_dir()`. Future "B6 path back-compat -- small (otr_legacy_audio_dir migration)" sprint will sweep them in one pass.

Also extended `tools/validate_workflow_links.py` with a `FORBIDDEN_PATTERNS` constant including `\botr_legacy_audio_dir\b` -- future audits can import the catalogue rather than maintaining the regex set in shell scripts.

### Phase 4 — Strict-deprecation audit reclassification (QA-5)

S26 left BUG-LOCAL-221 deferred because the strict-mode FAILURES traceback could not be captured. S27 fixed the underlying instrumentation gap and classified the warnings.

| Finding | Detail |
|---|---|
| Real cause of S26's traceback swallow | `tests/conftest.py::pytest_sessionfinish` does `raise SystemExit(2)` on any [KNOWN-FAIL-GUARD] NEW failure; SystemExit aborts before pytest prints the FAILURES section. S26's "cmd.exe shell terminated" theory was wrong -- cmd.exe was honest; the conftest hook was eating the traceback. |
| Durable fix | Built `docs/2026-05-13-S27-_strict_probe.py`: a standalone harness that monkey-patches the conftest hook to a no-op before pytest collects. The FAILURES traceback now survives. |
| Warning 1 (third-party) | `pytest_asyncio.plugin:247` -- PytestDeprecationWarning about `asyncio_default_fixture_loop_scope` unset. FIXED via pyproject.toml `[tool.pytest.ini_options] asyncio_default_fixture_loop_scope = "function"` (upstream-recommended value). |
| Warning 2 (third-party) | `torchao.dtypes.uintx.__init__:1` -- deprecated import path inside transformers' `AutoProcessor` chain. OTR doesn't import torchao directly; no OTR-side fix. Documented as `third_party_deprecation`. |
| BUG-LOCAL-221 status | CLOSED. Full traceback evidence at `docs/2026-05-13-S27-deprecation-audit-reclass.txt`. |

**Commit:** Phases 2 + 3 + 4 ship together (pending).

