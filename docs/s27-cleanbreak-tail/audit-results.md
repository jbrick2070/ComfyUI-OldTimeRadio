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

**Commit:** pending.

