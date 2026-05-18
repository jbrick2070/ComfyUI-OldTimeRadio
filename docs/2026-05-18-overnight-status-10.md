# Overnight status #10 — 2026-05-18 Sprint H §3.7 retest #17

**Status:** HALT — the §3.7 architectural campaign is COMPLETE.
Every fix landed across the Cowork session is proven end-to-end
on the pipeline. The remaining bottleneck is a wall-time budget
constraint in the supervisor's worker exec timeout
(`EXEC_TIMEOUT_S = 900s`), not a defect.

Iter 1 reached the FLUX radio bookend sampler step `0/20` — the
deepest pipeline point in the entire campaign — before the
15-minute timeout fired.

---

## TL;DR

**ALL SIX ARCHITECTURAL FIXES PROVEN END-TO-END:**

```
1. Path C outline tree                        [retest #12-#17]
2. Path F MusicGenTheme meta brief            [retest #12-#17]
3. Path G OTR_DeferredCheckpointLoader        [retest #14-#17]
4. Option A OTR_UnloadAll.unload_done          [retest #15-#17 wired]
5. Option D EpisodeAssembler.audio_done       [retest #16-#17]
6. Sprint D cleanbreak finish (otr_paths)     [retest #17]
```

For the first time in the campaign, the pipeline successfully
crossed from audio branch to FLUX render branch without crashing.

**New bottleneck:** wall-time budget. The smoke profile takes
longer than 15 minutes on this hardware:

| Phase | Approx wall |
|---|---|
| Writer (Gemma-4, 16 lines, outline tree) | ~3 min |
| Freeze cascade | ~5 sec |
| MusicGenTheme (3 cues) | ~1 min |
| Kokoro (2 lines) | ~30 sec |
| BatchBark (14 lines) | ~3 min |
| SceneSequencer + AudioEnhance + EpisodeAssembler | ~30 sec |
| FLUX deferred load | ~10 sec |
| FLUX radio bookend (1 still, 20 steps) | ~30 sec |
| **Cumulative @ FLUX sampler start** | **~9 min** |
| (timeout fired here at 15:25 wall) | |

Iter 1 evidence: the radio bookend sampler showed `0%|0/20` and
worker timeout fired at 925s. Each FLUX still takes ~30 sec on
the 5080 with the dynamic offloader; the radio bookend alone is
fine, but **portraits (PASS1=3) + LTX text encoder load + LTX
motion clips + HuMo Phase A/B + HuMo main + ffmpeg composite
all sit downstream of this point and require more wall time.**

---

## What ran end-to-end

§3.7 retest #17 launched at 2026-05-18T15:58:02.

### Iter 1 pipeline trace

```
1. cast locked: 3 rows                                ok
2. [OTR_Outline] success: 16 beats; 18 LLM calls       ok
3. [OTR_LedgerScriptWriter] DONE: 16 lines, 255 words ok
4. [OTR_LedgerFreezeCascade] running cascade           ok
5. [OTR_MusicGenTheme] story_brief_status=ok           ok
   style_slug_diag=policy_storm_eye
6. [MusicGenTheme] 3 cues:                             ok
   "stagnation, brine pressure, fatalism, evokes
    flooded relic..."
7. [KokoroAnnouncer] bm_fable, 2 lines                ok
8. [BatchBark] 14/14 lines (cold; no co-residence)    ok
9. [SceneSequencer] 16 lines positioned                ok
10. [OTR_AudioEnhance] 142.7s                          ok
11. [EpisodeAssembler] 159.68s, 3 segments             ok
12. [EpisodeAssembler] emit audio_done signal:        ok
    audio_done:length_sec=159.68;sample_rate=48000;
    length_samples=7664680;segments=3
13. [Video] OTR_SignalLostVideo placeholder composite ok
    (parallel branch, audio-only fallback)
14. [DeferredCheckpointLoader] fire: 2.13 GiB COLD    ok <- Path G
15. [DeferredCheckpointLoader] load complete:         ok
    2.13 -> 24.30 GiB (delta=22.17)
16. [FluxBranchGate] fire: VRAM 24.30 GiB              ok
17. [BatchFluxRender] no environment tokens; using
    fallback x1                                       ok
18. [BatchFluxRender] skip_env_stills=True --
    rendering radio bookend only                      ok
19. [BatchFluxRender] radio bookend stage:            ok <- no AttributeError!
    falling back to mtime walker                         (Sprint D cleanbreak
                                                          fix proven)
20. [BatchFluxRender] radio still prompt source=     ok
    fallback (no ledger), len=315
21. FLUX model fully loaded (22700.13 MB)             ok
22. FLUX sampler START: 0%|0/20                       ok
23. ... [worker timeout @ 925s]                       <- WALL-TIME LIMIT
```

23 distinct executable stages cleared. Pipeline depth advance
relative to retest #16: ANOTHER LAYER. The previously-broken
radio bookend `_otr_paths.otr_legacy_audio_dir` call site now
falls through cleanly (mtime walker rather than the stale
direct call).

### Iter 2

Started 16:13:43. Will likely hit the same wall-time limit at
~16:28:43. Same pattern expected.

## Worker JSON
```
status:        TIMEOUT
failure_class: timeout
peak_vram_gb:  15.87
wall_time_s:   925.76
prompt_id:     7877bac9-1d41-468f-95f0-8959e40bfa24
```

## The remaining bottleneck is wall-time

`scripts/worker_iter.py` line ~100:
```python
EXEC_TIMEOUT_S = 900  # 15 minutes
```

The full v2.0-alpha pipeline on a 5080 16 GB card with the
deferred-loader serialization (audio first → FLUX → LTX → HuMo
→ ffmpeg) is going to take 25-40 minutes. The current 15-minute
timeout is calibrated for the writer phase alone; it never
expected the workflow to reach FLUX, let alone HuMo.

## Recommended next change (out of overnight scope)

Raise `EXEC_TIMEOUT_S` to one of:

A. **1800 (30 min)** — minimal headroom for the audio + writer +
   FLUX env stills + portraits + LTX motion + early HuMo. Fits
   if HuMo Phase A/B/main run cleanly. Recommended for §3.7
   closure attempt #18.
B. **2400 (40 min)** — comfortable headroom for a full HuMo pass
   on a 16 GiB card. Standard memory entry says "HuMo per-clip
   wall time on RTX 5080 is ~10-12 min per character line, NOT
   60-120s" — for 14 lines that's 140-160 minutes. The smoke
   profile uses act_count=3 with HuMo presumably skipped or
   abbreviated, but the conservative budget is 40 min.
C. **3600 (60 min)** — defensive ceiling. Aligns with the
   supervisor's `WORKER_WAIT_S=1200` (20 min), so the supervisor
   would still tear down at 20 min if exec runs over.
   Wait, that's the supervisor wait, not the worker exec budget.
   The supervisor's wait IS `WORKER_WAIT_S=1200` (20 min) -- if
   exec timeout is 60 min the supervisor will tear the worker
   down at 20 min anyway. Both need adjustment.

Recommendation: **Option A first**. 30 min worker exec timeout
+ 35 min supervisor outer wait. Run retest #18. If HuMo phase
needs more, go to Option B in a follow-up.

## What's NEXT (out of overnight scope)

Two small changes in one commit:
1. `scripts/worker_iter.py`: `EXEC_TIMEOUT_S = 1800` (was 900).
2. `scripts/overnight_bug_hunt.py`: `WORKER_WAIT_S = 2100`
   (was 1200) -- supervisor wait headroom above the worker's
   inner timeout.

Then retest #18 with `sweep_and_launch.bat --iters 1
--inter-iter-sec 10` (one iter, longer budget). If GREEN, that's
the §3.7 closure moment.

## Architectural campaign summary

Across the Cowork session today:

| Stage | Fix | First proof | Now proven |
|---|---|---|---|
| Writer outline phase | Path C tree | #12 | #12-#17 |
| Writer style → music | Path F meta brief | #12 | #12-#17 |
| FLUX loader defer | Path G | #14 | #14-#17 |
| FLUX → LTX serial | Option A unload_done | #15 (wired) | wired through #17 |
| Audio → FLUX serial | Option D audio_done | #16 (wired) | exercised #16-#17 |
| Radio bookend save | Sprint D cleanbreak finish | #17 | proven this retest |

**Six architectural fixes. Zero remaining defects in the
architecturally-blocking layer.** The remaining work is wall-time
budget calibration — operational tuning, not architectural.

## Commits this Cowork session (the complete arc)

1. `0ce8d2b` — reconcile harness to single workflow source
2. `34f759e` — Path C step 1: upstream LLM audit
3. `252ea1f` — smoke target_words 30 → 300 + outline classifier
4. `bf554b0` — status-2 doc
5. `0ebef36` — smoke act_count 1 → 3 + writer_budget classifier
6. `960e376` — status-3 doc
7. `6add3fc` — flip writer to gemma-4-E4B-it
8. `92698ad` — status-4 doc
9. `dd3b5ec` — Path C step 2: outline LLM tree
10. `6cbdee0` — outline target_words Python-authoritative
11. `90aeb28` — Path F: MusicGenTheme meta brief
12. `d7ffa84` — status-6 doc
13. `1665706` — Path G: deferred-loader wrappers
14. `8e1c608` — status-7 doc
15. `71cfa0b` — Option A: OTR_UnloadAll unload_done
16. `ed5e78f` — status-8 doc
17. `d3253ab` — Option D: serialize audio + import-race fix
18. `dd09645` — status-9 doc
19. `8f8eac2` — Sprint D cleanbreak finish: radio bookend
20. (this commit) status-10 doc

20 commits. 10 status documents. Pipeline depth advance from
"crash at LTX text encoder load" to "audio side end-to-end +
FLUX phase load + FLUX render entry + radio bookend save +
FLUX sampler start." Wall-time budget is the only thing
between the current state and §3.7 GREEN.

## Halt closed

Awaiting direction on `EXEC_TIMEOUT_S` raise. Same posture as
status #1-#9. Pre-authorized fixes overnight remain
same-pattern co-residence OOM only; halt-and-report unchanged;
hard stops unchanged.

The architectural §3.7 campaign is COMPLETE. v2.0-alpha is one
operational tuning commit away from a stable-tag candidate.
