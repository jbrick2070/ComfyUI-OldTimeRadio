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

(Items append here as commits land.)
