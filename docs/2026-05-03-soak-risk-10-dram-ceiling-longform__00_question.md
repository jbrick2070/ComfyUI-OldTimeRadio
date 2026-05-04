# Question -- 2026-05-03

# Round-robin question — OTR Phase A composite DRAM ceiling at long-form (>5 min) episodes

## Stack
- Windows 11, RTX 5080 Laptop (16 GB VRAM, sm_120 Blackwell), Python 3.12, torch 2.10.0, CUDA 13.0.
- Composite chain (post BUG-LOCAL-030 Phase A + B, commit `0ae58a0`):
  1. `BatchHumoRender` writes per-line `videos/lNNN.mp4` clips at native 480x832 portrait, 25 fps, ~3.88 s each.
  2. `BatchLTXRender` writes per-line LTX broadcast-unit clips at native 832x480 landscape, ~10 s each.
  3. `OTR_VideoComposite._render_master_mix_per_clip_mux_mode` (in `nodes/video_composite.py`) loops over the per-line clip list and for each clip:
     - Calls `_layered_per_clip_silent` which runs an ffmpeg pass: `scale=-2:832:force_original_aspect_ratio=decrease, pad=1472:832:trunc((ow-iw)/4)*2:trunc((oh-ih)/4)*2:color=black, fps=25 -an` -> writes a normalized 1472x832 silent intermediate to disk.
     - Then a per-clip-mux ffmpeg pass attaches the corresponding audio slice from the master mix WAV using `-c:a copy`.
  4. After the loop, ffmpeg concat demuxer stitches all per-line muxed clips into the per-scene composite.
  5. `OTR_RTXUpscale` upscales 1472x832 -> 1920x1080.
  6. `OTR_PostUpscaleProcgenBlend` blends the 1920x1080 procgen mp4 over the upscaled video via `-filter_complex [0:v][procgen]blend=all_mode=lighten:all_opacity=0.5[v]`, audio passes through with `-c:a copy`.

All intermediates live on disk under `output/otr/episodes/<ep>/{videos,composited,obs}/`. ffmpeg subprocesses are spawned one at a time, never in parallel.

## Real-world tested length
Smoke / short-act episodes only. ~30 s to ~90 s of audio, ~10-20 per-line clips. Phase A + B has NOT been soaked at episode lengths above ~3 min total audio. The original v1.5 baseline (Bark + master mix only, no per-clip layered composite) ran 5-7 min episodes fine.

## The risk vector
A long-form episode (>5 min audio) generates >100 per-line clips. The composite chain processes them serially via ffmpeg subprocess.run, but:

- The ffmpeg concat demuxer file list can be very long (>100 file entries).
- Each `_layered_per_clip_silent` pass writes a normalized intermediate to disk. So we hold (per-clip-render mp4 + normalized intermediate mp4 + per-clip-mux mp4) per-line on disk simultaneously until cleanup.
- Disk pressure: ~100 clips x ~3 intermediates x ~5-15 MB each = ~1.5-4.5 GB of transient mp4 files in `output/otr/episodes/<ep>/videos/` during the composite pass. Spacesaver runs AFTER.
- Concat-demuxer ffmpeg pass loads metadata for all clips into RAM before muxing. With 100+ clips + their audio streams, peak ffmpeg DRAM during concat is unknown.
- Then RTXUpscale eats the full concat output (1472x832 mp4, ~5+ min, large file) — VRAM-bound for the AI upscaler model, but the input mp4 is decoded into system RAM as frames during ffmpeg-side prep.
- Then PostUpscaleProcgenBlend `-filter_complex blend` reads BOTH 1920x1080 mp4s simultaneously. With a 5+ min episode, both are large files; ffmpeg's `blend` filter buffers frames from both inputs.

## The questions

1. **Realistic peak system RAM for a 5-7 min OTR episode through the full Phase A + B chain** — given ~100 per-line layered composites + concat-demuxer pass + RTXUpscale-side ffmpeg decode + PostUpscaleProcgenBlend filter_complex blend, what's the realistic peak DRAM footprint? Jeffrey's machine has 32 GB system RAM; is there a clear point in the chain where we'd OOM, swap to disk, or thrash?

2. **Is the per-clip three-mp4-on-disk transient pattern sustainable at length?** Specifically: HuMo render mp4 + layered composite normalized intermediate + per-clip-mux output mp4 all coexist for each line until the cleanup step. Should we be deleting the layered intermediate the moment the per-clip-mux is done writing, or is leaving them until the post-loop cleanup acceptable?

3. **Concat demuxer at >100 entries** — any known ffmpeg failure modes? File handle limits, line-length limits, ordering/timestamp drift?

4. **PostUpscaleProcgenBlend filter_complex blend on two 5+ min 1920x1080 mp4s** — will ffmpeg's blend filter stream-process them or buffer? If it buffers, we could spike DRAM to 4-8 GB during the blend pass. Should we be using a different filter strategy (overlay with alpha?) for long-form?

5. **What's the right canary metric** to add to the next soak so we can see DRAM pressure live, not after the OOM? Jeffrey runs LibreHardwareMonitor 24/7 (poll http://localhost:8085/data.json) — is there a specific DRAM utilization threshold above which we should abort the composite and fall back to a simpler chain?

## Constraints
- Local-only, no cloud.
- C7 (audio byte-identity) MUST be preserved end-to-end. Any optimization that re-encodes audio is OUT.
- 16 GB VRAM ceiling already understood; this question is about DRAM (system RAM) only.
- We can NOT change the ffmpeg version (system-installed).

Please be concrete: numbers, named ffmpeg flags, named metrics. Vague advice ("watch RAM usage") is not actionable. If you don't know a number, say so — better to flag uncertainty than to invent a figure.
