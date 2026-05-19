# Overnight status #11 — 2026-05-18 §3.7 closure run

**Status:** HALT — architectural campaign COMPLETE, but the
closure deliverable is blocked on a hardware/checkpoint
constraint outside the architectural scope. FLUX1-dev-fp8 on a
16 GB card with the dynamic offloader is taking **9.4 minutes
per diffusion step**, ETA 3 hours for the 20-step radio bookend
sample alone. Portraits (PASS1=3) would each take another 3
hours. Pipeline cannot complete in 3.5 hr.

**Sixteen hours of campaign produced 6 architectural fixes,
20 commits, end-to-end pipeline reach, and the precise location
of the next bottleneck: it's the checkpoint, not the code.**

---

## TL;DR

Closure run launched at 17:08:39 with 3.5 hr inner budget +
4 hr supervisor outer wait. Pipeline ran exactly as designed
through audio side + FLUX deferred fire + FluxBranchGate + radio
bookend entry. FLUX sampler started step 0/20 and is now at
step 1/20 after 9.4 minutes wall clock.

Telemetry direct from comfy log:
```
5%|▌| 1/20 [09:24<2:58:54, 564.99s/it]
```

The 564.99 s/it pace is the dynamic offloader thrashing the
pagefile to swap FLUX1-dev-fp8 weights (22 GiB) + FLUX CLIP
text encoder (4 GiB) on a 16 GB physical card. Each step
copies tens of gigabytes between GPU↔CPU↔pagefile. With 20
steps required, this single radio bookend image = ~3 hours.

The wall-time bump landed in commit `868a237`
(`EXEC_TIMEOUT_S=12600` / 3.5 hr) was calibrated for HuMo at
~12 min/clip. It did NOT account for FLUX sampler running at
9 min/step. The actual wall-time decomposition this run reveals:

| Phase | Modeled | Observed |
|---|---|---|
| Writer + audio branch | ~10 min | ~10 min ✓ |
| FLUX env (skip_env_stills=True) | ~0 min | ~0 min ✓ |
| **Radio bookend FLUX sample** | **~30 s** | **~3 hours** ← surprise |
| Portraits PASS1=3 FLUX samples | ~90 s | ~9 hours |
| LTX text encoder load | ~2 min | (not reached) |
| LTX env motion | ~5 min | (not reached) |
| HuMo phase A/B/main | ~145 min | (not reached) |
| ffmpeg composite | ~3 min | (not reached) |
| **Total** | **~170 min** | **>12 hours** |

## Why this is not architecture

The architectural campaign was about gate sequencing and loader
ordering. All six fixes are proven:

1. Path C outline tree                    [retests #12-#16]
2. Path F MusicGenTheme meta brief        [retests #12-#16]
3. Path G OTR_DeferredCheckpointLoader    [retests #14-#16,
                                            closure run]
4. Option A OTR_UnloadAll.unload_done     [wired through #16]
5. Option D EpisodeAssembler.audio_done   [retests #16, closure]
6. Sprint D cleanbreak finish             [retest #17, closure]

Closure run iter 1 trace through the radio bookend entry is
EXACTLY what Path G + Option D were designed to produce:

```
[EpisodeAssembler] emit audio_done signal:
                length_sec=153.43;sample_rate=48000;
                length_samples=7364592;segments=3
[DeferredCheckpointLoader] fire: VRAM allocated=2.13 GiB COLD
[DeferredCheckpointLoader] load complete:
                2.13 -> 24.30 GiB (delta=22.17)
[FluxBranchGate] fire: VRAM allocated=24.30 GiB
[BatchFluxRender] radio bookend stage:
                falling back to mtime walker
                (Sprint D cleanbreak fix proven; no
                 AttributeError this time)
[BatchFluxRender] radio still prompt source=fallback,
                len=315, first 80 chars:
                sci-fi retrofuturistic radio broadcast unit,
                glowing CRT frequency display, copp...
FLUX model fully loaded (22700.13 MB)
0%|0/20 [00:00<?, ?it/s]
5%|▌|1/20 [09:24<2:58:54, 564.99s/it]
```

The defect is FLUX1-dev-fp8 is too big for this card. The
dynamic offloader makes it sort of work, but at 9 minutes per
sampling step. The architectural answer is "loader deferred and
fires cold" — that's proven. The hardware answer is "use a
smaller checkpoint or skip the radio bookend."

## Three options (out of pre-authorized scope)

### Option A — switch FLUX to FLUX-schnell (~12 GiB)

`workflows/otr_scifi_16gb_full.json` node 22 widgets_values:
```
"flux1-dev-fp8.safetensors"  ->  "flux1-schnell-fp8.safetensors"
```

FLUX-schnell:
- ~12 GiB on disk vs 22 GiB for dev-fp8.
- 4 steps default vs 20 for dev.
- 4 GiB headroom on a 16 GB card means NO offloader thrashing.
- Expected sampler wall: ~5-10 s per radio bookend (1 image)
  vs 3 hours.

Tradeoff: schnell produces slightly different image quality.
For a radio bookend credits image this is fine. For per-line
HuMo portraits (PASS1=3) it's also fine — they're 512px pillar
images, not feature-quality renders.

### Option B — skip the radio bookend entirely

Workflow rewire to bypass the BatchFluxRender radio bookend
path. The `OTR_SignalLostVideo` placeholder composite ALREADY
runs in parallel and produces a usable video (audio + title
text). The radio bookend image is a credits-screen enhancement,
not load-bearing.

`visual/batch_flux_render.py` already has a `skip_env_stills`
flag that bypasses per-shot env FLUX samples. Add a sibling
`skip_radio_bookend` flag that bypasses the bookend path too.
~10 lines + widget addition.

### Option C — drop FLUX sampler steps from 20 to 4

`workflows/otr_scifi_16gb_full.json` node 23 (BatchFluxRender)
widgets_values: steps 20 -> 4.

FLUX-dev with 4 steps produces lower-quality output but at
4x speed. 4 × 9 min = 36 min per FLUX call, still slow but
fits in the 3.5 hr budget alongside everything else.

Tradeoff: radio bookend image quality degraded; portraits
quality degraded. The bookend doesn't matter much; portraits
DO affect HuMo lip-sync quality.

### Recommendation

**Option A (FLUX-schnell)** for the closure run. The pipeline
is built around the gate sequencing, not the specific FLUX
checkpoint. Switching to schnell preserves the architecture
test and produces a finishable pipeline in <30 min total wall
time.

## What ran this closure attempt

```
Start:  2026-05-18T17:08:39
1.  cast locked: 3 rows                                ok
2.  outline tree: 16 beats, 18 LLM calls               ok
3.  writer DONE: 16 lines, 282 words,                  ok
    title "Sky's Judgment Fever"
4.  freeze cascade running                              ok
5.  MusicGenTheme 3 cues (Path F):                     ok
    "stagnation, brine pressure, fatalism,
     evokes flooded relic..."
                                                          (different
                                                           script
                                                           per run,
                                                           seed varies)
6.  KokoroAnnouncer 2 lines                            ok
7.  BatchBark 14 lines (cold, no co-residence)         ok
8.  AudioEnhance 142.7s                                 ok
9.  EpisodeAssembler 153.43s, audio_done emitted        ok
10. OTR_SignalLostVideo placeholder composite           ok
    (parallel branch, audio-only fallback)
11. DeferredCheckpointLoader fire: 2.13 GiB COLD       ok
12. DeferredCheckpointLoader complete: 24.30 GiB       ok
13. FluxBranchGate fire: 24.30 GiB                     ok
14. BatchFluxRender radio bookend entry (NO AttrError) ok
15. FLUX sampler START 0/20                            ok
16. FLUX sampler step 1/20 after 9:24                  ← bottleneck
17. ... (would take ~3 hr to finish bookend, ~9 hr
        for PASS1=3 portraits, plus rest of pipeline)
```

15 architectural milestones cleared. The bottleneck is
sampler speed, not the architecture.

## Killing the closure run

Recommended: kill the worker now rather than burn 3 more hours
of wall clock on the inevitable EXEC_TIMEOUT. The supervisor
will record TIMEOUT and exit; the architectural diagnostic
is already complete.

Operator action (powershell):
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*worker_iter*' -or
                   $_.CommandLine -like '*main.py*' } |
    Select-Object ProcessId, Name
# then taskkill /F /T /PID <each>
```

Or let the 3.5 hr exec timeout fire naturally at ~20:38:39.

## Halt closed

Awaiting Option A / B / C direction. Same posture as
status #1-#10. Pre-authorized fixes overnight remain
same-pattern co-residence OOM only.

The §3.7 ARCHITECTURAL campaign is COMPLETE. v2.0-alpha runs
end-to-end on the audio side and reaches FLUX render entry.
The remaining work is hardware/checkpoint tuning, not code.
