# Peer review: OTR v2.0-alpha pipeline (2026-04-26)

I'm Jeffrey Brick, on a single RTX 5080 Laptop 16 GB Blackwell. I built an
audio-drama-with-video generation pipeline (OTR / "Signal Lost") and want a
sanity-check peer review. Be direct. I'm skeptical there's much to find.

## Platform pins (locked, don't suggest violating)

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120
- Python 3.12.11, torch 2.10.0+cu130
- SageAttention + SDPA active. **Flash Attention 2/3 NOT AVAILABLE** — no
  Blackwell-Windows wheel exists; don't suggest FA chasing.
- 100% local, no cloud, no API keys, no paid services
- VRAM ceiling: ~14.5 GB real-world target
- HuMo 14B fp8 + lightx2v 4-step distill LoRA + ModelSamplingSD3 stack
  staged at ~16.5 GB via async-offload (DynamicVRAM). Can't coexist with
  FLUX in VRAM. `OTR_UnloadAll` gates the canvas FLUX→HuMo handoff.

## The 7-stage pipeline (all under scripts/, all stdlib + ffmpeg, no
new frameworks)

**1. build_silent_test_episode.py** — reads an upstream LLMDirector ledger,
expands the Scene > Shot > Beat > Clip hierarchy (l2-2026-04-25 schema),
injects "lip-synching RADIO" atmospheric beats (a synthetic cast member
playing a 1940s console radio whose grille HuMo lip-syncs to ambient audio
during non-dialogue beats), writes `silent_test_<id>/ledger.json + meta.json`.

**2. render_flux_batch.py --mode bundled** — POSTs ONE ComfyUI /prompt
graph with N parallel chains:
`CheckpointLoaderSimple → CLIPTextEncode pos+neg → EmptyLatentImage
(batch_size=1) → KSampler (euler/simple/20 steps) → VAEDecode →
SaveImage` with per-target filename_prefix. 51 renders for an episode
(7 portraits + 44 composites) in 37 min on FLUX-dev-fp8 at 1024×1024.
Bundle saves the inter-prompt setup tax. Shared
CheckpointLoaderSimple + shared negative CLIP encode across all chains.
Ledger-driven targets — RADIO portrait skips per-shot composites, beats
inside a shot share one composite for cinematic continuity.

**3. verify_flux_coverage.py** — globs `output/otr/stills/`, prints
PRESENT/MISSING per ledger target, exits non-zero if incomplete.

**4. render_humo_batch.py** — Pattern B sequential /prompt orchestrator
(one prompt per ledger line). HuMo 14B fp8 e4m3fn scaled (Kijai) +
lightx2v 4-step distill LoRA + ModelSamplingSD3 shift=8 + Whisper Large
v3. Frame counts must be 4n+1 (Wan VAE temporal compression);
`humo_length_for_dur(dur_s)` snaps. Per-clip mp4 →
`output/otr/videos/<id>/humo_<line>_*.mp4`. ~6:15 cold per clip, model
stays warm across submits via ComfyUI's model cache.

**5. render_episode_concat.py** — pure ffmpeg. Reads ledger, finds
`humo_<line>_*.mp4` per beat in order, **pre-trims each to its line's
dur_s** (default on, stream copy, ~1 frame keyframe tolerance) so
concat video timeline matches proc-gen audio frame-accurately.
Two-pass: (a) concat-demuxer video-only, (b) mux master audio +
**embed `.vtt` as mp4 mov_text subtitle stream** + sidecar .vtt.

**6. render_compose_frame.py** — pure ffmpeg. Optional vintage radio
cabinet PNG overlay (alpha-cutout, HuMo plays inside speaker grille) +
filament-glow vignette + showvolume VU strip lower edge. x264 medium
crf 18.

**7. render_upscale_batch.py** — POSTs one ComfyUI graph: `LoadVideo →
GetVideoComponents → SeedVR2LoadDiTModel (3B fp16) +
SeedVR2LoadVAEModel (ema_vae_fp16) → SeedVR2VideoUpscaler
(resolution=1080, batch_size=33 4n+1, temporal_overlap=3,
color_correction=lab, blocks_to_swap=32) → CreateVideo (preserves
source fps + audio) → SaveVideo`. Auto-promotes result to flat
`output/otr/episodes/` delivery folder for media server.

## Output layout

```
output/
├── otr/
│   ├── stills/                 (FLUX renders)
│   ├── videos/<episode_id>/    (HuMo per-clip + intermediates)
│   └── episodes/               (FLAT, final 1080p deliverables only)
└── old_time_radio/             (legacy v1.5 audio episode lineage)
```

## Architecture notes

- Workflow JSON (`otr_scifi_16gb_full.json`, 32 nodes / 50 links) drives the
  canvas demo: audio pipeline (Bark+Kokoro+MusicGen+AudioGen → SceneSequencer
  → AudioEnhance → EpisodeAssembler → SignalLostVideo audio-only fallback) +
  FLUX batch + ONE HuMo demo clip. Multi-clip HuMo + concat + compose +
  upscale all run out-of-band via the orchestrator scripts.
- Proc-gen audio path and HuMo video path are **parallel sinks in the canvas,
  merged at concat (stage 5 post-process)**.
- 0 orphan nodes in the workflow. End-to-end I/O trace clean per audit.

## Review questions

1. **Are there real, implementable optimizations** (not theoretical, not
   FA chasing, not cloud) that would speed up or improve this on a 16 GB
   Blackwell laptop?
2. **Architectural gotchas I'm missing?** Timing edge cases, race
   conditions, stale-cache hazards across the orchestrator chain.
3. **Foldering sensible** (`output/otr/{stills,videos,episodes}/`) or is
   there a cleaner alternative?
4. **Decoupled audio + HuMo paths** (parallel sinks, merged at concat) —
   good design or should they weld earlier?

Tone: terse. Skip rewrites in different frameworks, skip cloud, skip FA.
Be specific about anything that's actually wrong.
