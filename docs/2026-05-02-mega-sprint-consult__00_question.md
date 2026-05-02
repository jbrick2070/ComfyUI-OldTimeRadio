# Question -- 2026-05-02

# OTR v2.0-alpha Sprint 3 mega-sprint -- pre-smoke wiring review

## Context

OTR is a ComfyUI plugin generating 1940s-style radio drama episodes. The Sprint 3 mega-sprint wires three new components into `workflows/otr_scifi_16gb_full.json` on branch `v2.0-alpha`:

1. **OTR_BatchLTXRender** -- in-graph LTX-2 renderer for non-character ledger lines (announcer / music_open / music_close / music_inter / sfx). Uses radio_bookend.png as both start and end keyframe via `LTXVAddGuide` for seamless looping. Writes silent libx264 yuv420p mp4 to `output/otr/videos/<ep>/<line_id>.mp4` (same dir HuMo writes character clips to). Stamps `ledger.clips[].source_kind="ltx"`.
2. **OTR_RTXUpscale** -- final-stage NVIDIA RTX VSR ULTRA upscaler. Path-in / path-out wrapper. Decodes video frames in chunks via ffmpeg pipe, runs `nvvfx.VideoSuperRes` per frame, writes silent libx264 yuv420p mp4, then muxes original mp4's audio with `-c:a copy` (zero audio re-encode -- C7 byte-identical preserved). Bypassable via `bypass=True` widget for raw 832x480 deliverables.
3. **LowVRAMCheckpointLoader** -- ComfyUI-LTXVideo's CheckpointLoaderSimple subclass. Adds a `dependencies` input that forces sequential loading (HuMo unloads before LTX claims VRAM).

## Locked Architecture Truth (settled 2026-05-02; do not relitigate)

- Resolution policy: native 832x480 end-to-end; LTX writes 832x480; HuMo pillarbox 832x480 letterbox.
- LTX seamless-loop: `LTXVAddGuide` with frame_idx=0 strength=0.75, frame_idx=-1 strength=0.6.
- Frame-count rule: LTX `8n+1`, capped at `LTX_MAX_FRAMES=177` to match HuMo's verified 16 GB ceiling.
- Tiled VAE decode: `tile_size=512, overlap=64, temporal_size=4096, temporal_overlap=8` (Goofer-proven on RTX 5080 Blackwell).
- Strict teardown after LTX loop: unload_all_models + gc + empty_cache + cuda.synchronize.
- `_NEVER_HUMO_ROLES = {announcer, music_open, music_close, music_inter, sfx}` (single source of truth).
- VRAM ceiling: 14.5 GB audio, 15.5 GB video.

## Wiring done in this commit (workflow JSON link topology)

- LowVRAMCheckpointLoader (node 54): widget `ckpt_name="ltx-video-2b-v0.9.safetensors"` (the bundled LTX 2B v0.9 file at `C:\ComfyUI-Models\checkpoints\ltx-video-2b-v0.9.safetensors`). `dependencies` input <- BatchHumoRender.clips_dir (link 86, STRING into wildcard input).
- OTR_BatchLTXRender (node 55): MODEL/CLIP/VAE inputs <- LowVRAMCheckpointLoader's three outputs (links 87/88/89). `ledger_json` <- OTR_SignalLostVideo.0 (link 90, same STRING source as HuMo gets; consumers duck-type via stem fallback). `humo_clips_dir` (new optional STRING) <- BatchHumoRender.clips_dir (link 91).
- VideoComposite (node 52) `clips_dir` rewired from BatchHumoRender (old link 81) to BatchLTXRender.clips_dir (new link 92). Both renderers write to the same `otr_videos_dir(episode_id)`, so VideoComposite picks up the union by line_id.
- OTR_RTXUpscale (node 56): `source_mp4_path` <- VideoComposite.final_mp4_path (link 93). Widgets: bypass=False, target_width=1920, target_height=1080, quality="ULTRA", chunk_frames=64, out_suffix="_1080p". Output: `<ep>_1080p.mp4`.

## Deviation from Architecture Truth (intentional, documented)

The Architecture Truth specified `UNETLoader + CLIPLoader (T5) + VAELoader` for LTX 2B fp16, NOT `CheckpointLoaderSimple`. Reason given: split-load lets ComfyUI offload T5/VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.

REALITY: LTX 2B v0.9 ships ONLY as a bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB, all components in one file). No separate LTX UNet / LTX VAE files exist on Lightricks' Hugging Face repo for the 2B v0.9 line; you'd have to manually split a `state_dict` to use UNETLoader/VAELoader.

RESOLUTION: Use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader`. It IS a `CheckpointLoaderSimple` subclass, but it adds a `dependencies` input that ComfyUI uses to force the load to run AFTER the upstream node finishes (here: BatchHumoRender). Combined with the existing strict teardown in batch_humo_render.py (`unload_all_models + gc + empty_cache + cuda.synchronize` in finally), this satisfies the C2 sequencing intent: HuMo fully unloads before LTX loader fires. The "no carve-out for CheckpointLoaderSimple" rule was about preventing OOM via parallel-load on the hot HuMo cache; sequencing eliminates that risk.

## Specific risk question

**What can break this on a real 16 GB Blackwell run?**

Concrete worries to address (yes / no / what would you watch for):

1. **VRAM**: HuMo (16.5 GB staged then unloaded) -> LTX 2B (8.7 GB bundled load) sequence. With strict teardown does HuMo actually clear ~16 GB before LTX claims VRAM, or do we hit a 14.5 GB cap from a leaked tensor / cached compilation / lingering KV?
2. **Audio path / C7**: VideoComposite produces the 832x480 mp4 with `master_mix_per_clip_mux` (audio `-c:a copy` from procgen). RTX upscale chunks frames via ffmpeg `-an` decode pipe, encodes silent libx264 yuv420p, then muxes original mp4 audio with `-c:a copy`. Is there any path where the upscale stage's audio mux can drift from byte-identical (e.g. timestamp re-anchoring, faststart re-write, container repacking, missing AAC bitstream filter)?
3. **Ledger / clips_dir union**: HuMo writes `<line_id>.mp4`, LTX writes `<line_id>.mp4` to the same dir. If HuMo writes a character line and LTX (somehow) also writes the same line_id (shouldn't happen because the role filter excludes character), the second write clobbers the first. Is the `is_never_humo_role()` filter sufficient defense, or should LTX have a "skip if file already exists" check too?
4. **DAG sequencing edge cases**: ComfyUI execution order for the chain `EpisodeAssembler -> SignalLostVideo -> BatchHumoRender -> LowVRAMCheckpointLoader -> BatchLTXRender -> VideoComposite -> RTXUpscale`. With the wildcard `dependencies` input on LowVRAMCheckpointLoader receiving BatchHumoRender's STRING output, will ComfyUI honor the dependency edge, or is there a known bug where wildcard inputs don't enforce DAG ordering?
5. **RTX VSR + nvvfx**: My OTR_RTXUpscale chunks frames in 64-frame batches, runs `nvvfx.VideoSuperRes` per-frame inside the chunk, encodes via ffmpeg subprocess. Have you seen nvvfx leak GPU memory across long runs (5 min episode = 7,500 frames @ 25fps = ~117 chunks)? Should I tear down and re-create the `VideoSuperRes` context per chunk, or once per run as I'm doing?
6. **LTX prompt risk**: My `_PROMPT_BY_ROLE` includes "no people in frame" + a heavy negative prompt suppressing faces. Will LTX 2B v0.9's distilled sigma schedule (8 steps, sigmas `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`) honor the negative prompt strongly enough, or do we need a higher CFG (currently 1.0 because distilled)?

**Format requested**: Bullet list per question. Be specific. If a failure mode is real, name the exact log line / symptom you'd grep for. If a question is "won't happen" with high confidence, say so plainly. Do not relitigate the locked architecture above; only flag wiring / runtime risks introduced by THIS commit.
