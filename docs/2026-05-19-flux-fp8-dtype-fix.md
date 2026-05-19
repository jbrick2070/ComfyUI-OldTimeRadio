# FLUX1-dev-fp8 dtype-upcast fix -- 2026-05-19

**Status:** FIX APPLIED across all four launcher sites (2026-05-18). Verification
of the post-fix VRAM/sampler curves pending Jeffrey's next sweep run.

**One-line summary:** `--force-fp16` global launch arg upcast the natively-fp8
`flux1-dev-fp8.safetensors` weights to fp16, doubling the footprint from ~11 GiB
to ~22 GiB on a 16 GB card. The dynamic offloader compensated by paging weights
to RAM/pagefile per sampler step at ~9-15 minutes per step. Removing the flag
restores native fp8 load (~11 GiB resident) and the expected ~10-15 sec/step
sampler pace.

---

## Symptom (closure run 2026-05-18)

`docs/2026-05-18-overnight-status-12.md` and `-11.md` document the closure
run telemetry. The architectural campaign reached FLUX entry exactly as
designed (all 6 §3.7 fixes proven through 15 milestones, audio sealed at
153.43 s, FLUX deferred fire + branch gate + cleanbreak all green), then
stalled at the sampler. Direct from `logs/comfy_session_iter_001.log`:

```
model weight dtype torch.float16, manual cast: None
[DeferredCheckpointLoader] fire: VRAM allocated=2.13 GiB COLD
[DeferredCheckpointLoader] load complete:
                2.13 -> 24.30 GiB (delta=22.17)
...
FLUX model fully loaded (22700.13 MB)
  5%|1/20 [09:24<2:58:54, 564.99s/it]
```

LibreHardwareMonitor during the sample:

- GPU Core load: 100%
- GPU Memory used: 16240 / 16303 MB (pinned)
- D3D Shared Memory: 10445 MB (offloader paging from system RAM)
- ComfyUI process RSS: 19.5 GB

The 564.99 s/it pace is what dynamic-offloader thrashing looks like on a
checkpoint that won't fit resident: each sampler step pays a per-layer swap
tax that varies with the diffusion timestep's access pattern.

## Root cause

`flux1-dev-fp8.safetensors` is a Comfy-Org fp8 quantized checkpoint that loads
at ~11 GiB native. The `--force-fp16` ComfyUI launch arg overrides the
checkpoint's native dtype and casts every weight to torch.float16 at load
time, doubling the footprint to ~22 GiB. On a 16 GB card the resident-VRAM
budget cannot hold the full checkpoint plus the FLUX CLIP text encoder
(~4 GiB) plus the sampler working set, so the dynamic offloader pages weights
to RAM/pagefile per sampler step.

This is silent: the comfy log line `model weight dtype torch.float16, manual
cast: None` looks normal because fp16 IS a valid weight dtype. The defect is
that the checkpoint should NOT have been cast to fp16 -- it was already in a
smaller fp8 native format.

The flag was inherited from a pre-fp8-era recipe ("Tested Blackwell settings
carried over verbatim"). FP8 checkpoint support is newer; the flag should
have been retired when `flux1-dev-fp8.safetensors` was adopted, but the
adoption never re-audited the launcher.

## Fix

Removed `--force-fp16` from every launcher site in the repo + manual launcher.
Four sites total carried the flag; the initial diagnosis only named one.

| # | Path | Site | Status |
|---|------|------|--------|
| 1 | `C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat` | Manual ComfyUI launch | FIXED |
| 2 | `scripts/worker_iter.py` line 549 | Overnight bug-hunt smoke worker (closure-run launcher) | FIXED |
| 3 | `scripts/start_comfy_h0_baseline.bat` line 20 + REM block | Sprint H baseline launcher (cited as source-of-truth template) | FIXED |
| 4 | `scripts/_start_comfyui.ps1` line 62 + inline comment line 51 | Cowork helper script | FIXED |

The closure-run launcher was site #2 (`worker_iter.py`), NOT site #1 (the
manual .bat) as originally framed in status-12. The supervisor
(`overnight_bug_hunt.py`) spawns `worker_iter.py` which inline-Popens
ComfyUI with the args copied from site #3's recipe -- both still carried
`--force-fp16` until this fix.

Sites 3 and 4 also got explanatory comment blocks pointing back to this
doc so the next operator who picks up the recipe doesn't reintroduce the
flag.

### Confirmed clean: no fallback dtype-upcast sources

Pre-flight audit before the edit (per status-12 fallback list):

- `extra_model_paths.yaml` (install-dir copy + Roaming canonical
  `extra_models_config.yaml`): paths-only, no fp16/dtype/force/precision
  keys
- ComfyUI Desktop `config.json` (Roaming): no fp16/dtype/force/precision
  keys
- Desktop shortcut: none present at `$env:USERPROFILE\Desktop\*.lnk`

The four launcher sites were the complete inventory.

## Verify (pending next sweep run)

Expected on the next `scripts/sweep_and_launch.bat --iters 1
--inter-iter-sec 0` run, in `logs/comfy_session_iter_001.log`:

```
[DeferredCheckpointLoader] load complete:
                2.13 -> ~13 GiB (delta=~11 GiB)
...
FLUX model fully loaded (~11000 MB)
  5%|1/20 [00:15<...,  ~15.00s/it]
```

Specifically:

1. `[DeferredCheckpointLoader] load complete: ... delta=~11 GiB` (was
   delta=22.17 GiB)
2. `FLUX model fully loaded (~11000 MB)` (was 22700.13 MB)
3. Sampler s/it at ~10-15 sec/step (was 564.99 s/it / 9.4 min)
4. Radio bookend PNG in
   `output/otr/episodes/<pending-or-named>/audio/radio_bookend_episode.png`
   within minutes (was projected 161 minutes for the bookend alone)
5. Total FLUX phase (PASS1=3 portraits + bookend) within 5-15 minutes
   (was projected 10.7 hours)
6. Full smoke iter wall <90 minutes (was projected >12 hours and busted
   the 3.5 hr exec timeout 3x)
7. VRAM peak <14.5 GiB (CLAUDE.md ceiling); previous run pegged 16240
   MB / 16303 MB

If sampler pace is still slow after the fix (>30 s/step), the dtype upcast
is happening somewhere else -- candidate sites to investigate in order:
ComfyUI Desktop registry / settings.json, model loader node widget
overrides, environment variables (`COMFYUI_*`, `PYTORCH_*`), or a stale
process reading a cached config. The pre-flight audit covered the
documented fallback list, so any remaining upcast would be in a
less-documented surface.

## Why this took 16 hours to find

The architectural campaign was focused upstream of the FLUX sampler: gate
sequencing, loader ordering, audio-side serialization, cleanbreak voice
path, Path C/F/G + Option A/D + Sprint D. Each of the 6 fixes was real
and each cleared a real milestone. The FLUX sampler running slow was
read as "weights-too-big-for-card thrashing the offloader, fix the
checkpoint" (status-11 Option A: switch to FLUX-schnell) rather than
"weights are being upcast at load, the fp8 checkpoint is fine, fix the
launch arg."

Generalize-able lesson: when a quantized checkpoint (fp8, fp4, NF4, INT8,
GGUF, etc.) loads at 2x its on-disk size, the suspect is a global
precision flag, not the checkpoint. The comfy log line `model weight
dtype torch.float16` is the tell -- a healthy fp8 checkpoint logs `dtype
torch.float8_e4m3fn` (or similar), and an upcast checkpoint logs the
forced precision instead. The flag is older than the fp8 ecosystem and
should be retired in any fp8-aware launcher.

Bible candidate: yes -- the lesson generalizes beyond FLUX to any
quantized-checkpoint workflow on a VRAM-constrained card.

## Tags

`launch-args`, `dtype-upcast`, `flux-fp8`, `blackwell`, `vram-thrash`,
`offloader`, `bible-candidate`

## See also

- `docs/2026-05-18-overnight-status-11.md` -- closure run halt narrative
- `docs/2026-05-18-overnight-status-12.md` -- closure run telemetry
  correction
- `BUG_LOG.md::BUG-LOCAL-230`
- `CLAUDE.md` Prime Directive #2 (VRAM ceiling 14.5 GB)
