# Overnight status #12 -- 2026-05-18 §3.7 closure run, corrected

**Status:** HALT recommendation stands -- but status-11's "FLUX sampler
stuck at step 0/20" framing was wrong. The sampler IS running. Jeffrey
heard it working (GPU + disk activity), and the live telemetry confirms.
This doc retracts the "stuck" claim and re-publishes the actual sampler
curve and budget math.

---

## What status-11 got wrong

I claimed FLUX sampler had not advanced past step 0/20. That was a
misread of the comfy log -- the tqdm bar was buffered (tqdm uses `\r`
for in-progress updates, only flushing the line to the log file when a
step completes). The system was unmistakably grinding:

- GPU Core load: 100%
- GPU Memory used: 16240 / 16303 MB (effectively pinned)
- D3D Shared Memory: 10445 MB (the offloader paging from system RAM)
- GPU Package power: 63.0 W
- GPU Core clock: 2977 MHz
- ComfyUI process RSS: 19.5 GB

That is exactly what dynamic-offloader thrashing looks like in flight,
not a hang. Apologies for the bad call.

---

## Actual FLUX sampler curve

| Step | Sampler wall-clock | Step duration | Running avg s/it |
|---|---|---|---|
| 1/20 | 09:24 | 564 s | 565 s/it |
| 2/20 | 14:15 | 291 s | 404 s/it |
| 3/20 | 27:01 | 766 s | 569 s/it |

The pace is **not** monotonically decreasing. Step 2 was fast (weights
happened to be resident in VRAM); step 3 was slower than step 1 (had
to swap a different layer set from pagefile). The offloader is not
"warming up" in any useful sense -- each step pays a per-layer swap
tax that varies with the diffusion timestep's access pattern.

## Budget math (revised, still over budget)

At the 569 s/it running average:
- Remaining 17 steps × 569 s = 9673 s = **161 min** to finish the radio
  bookend alone.
- PASS1=3 portraits × 20 steps × 569 s = 8.06 hours.
- Total FLUX phase ETA: ~10.7 hours.
- Plus LTX text encoder load, LTX env motion, HuMo phase A/B/main,
  ffmpeg composite -- unreached, but each is a real chunk of wall.

Even with the raised `EXEC_TIMEOUT_S=12600` (3.5 hr) the FLUX phase
alone busts the budget by 3x.

## Status-11's verdict (corrected, but stands)

The architectural campaign IS complete. All 6 fixes are proven through
15 pipeline milestones. The closure run reached the FLUX sampler and
the sampler IS running. The remaining problem is that **FLUX1-dev-fp8
at 22 GiB on a 16 GB card forces the offloader to thrash per step**,
and the per-step pace is too slow for a finishable pipeline in 3.5 hr.

The three options from status-11 still apply, with the corrected framing:

### A. Switch FLUX checkpoint to FLUX-schnell (recommended)
- `flux1-schnell-fp8.safetensors` (~12 GiB on disk).
- Fits in 16 GB VRAM with headroom: no offloader thrashing.
- Default steps 4 vs 20 for dev.
- Expected wall: ~5-10 s per radio bookend image vs 161 min.
- Preserves architecture test; quality acceptable for bookend + 512 px
  portraits.

### B. Skip the radio bookend
- `OTR_SignalLostVideo` placeholder already runs in parallel and
  produces a finishable video.
- Add `skip_radio_bookend` flag to `BatchFluxRender` (sibling of the
  existing `skip_env_stills` flag). ~10 lines + widget.
- Bypasses the bookend entirely; portraits still go through but
  PASS1=3 × 20 steps still busts budget unless paired with Option A
  or C.

### C. Drop FLUX sampler steps 20 -> 4
- Workflow node 23 `BatchFluxRender` `widgets_values` steps 20 -> 4.
- Quality degraded but 4x faster wall.
- Doesn't fix the per-step thrash, just runs fewer of them.

### Recommendation
**Option A (FLUX-schnell)** remains the right call. It eliminates the
offloader thrash entirely instead of working around it.

## Process posture

- ComfyUI PID 14716 still running, GPU at 100%.
- Supervisor PID 51276, worker_iter PID 37292 still active.
- Sampler at step 3/20, ~27 min sampler-clock elapsed.
- Inner exec timeout (3.5 hr / 12600 s) will fire around 20:38 PDT if
  not killed first.
- Run can be killed cleanly via the powershell block in status-11.

## What this commit does NOT do
- Does NOT modify the workflow.
- Does NOT touch any node / loader / gate.
- Does NOT switch FLUX checkpoint.
- Does NOT add `skip_radio_bookend` flag.
- Does NOT change sampler steps.
- Only retracts the "stuck at 0/20" framing and re-publishes accurate
  data.

## Halt closed

Awaiting Option A / B / C direction. Same posture as status-11.

The §3.7 ARCHITECTURAL campaign is COMPLETE. The pipeline runs
end-to-end on the audio side and reaches FLUX sampler entry, where
weights-too-big-for-card thrashes the offloader. Architecture works.
Hardware/checkpoint pairing is the remaining tuning.
