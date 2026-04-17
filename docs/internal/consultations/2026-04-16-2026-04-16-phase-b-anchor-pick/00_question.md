# Question -- 2026-04-16

# Phase B Anchor Image Model Pick — Round Robin

**Date:** 2026-04-16
**Project:** ComfyUI-OldTimeRadio (SIGNAL LOST), branch `v2.0-alpha`
**Owner:** Jeffrey Brick

## The decision

Pick the image model for Phase B "anchor frame" generation. Each episode has roughly 6-15 shots; one anchor frame per shot drives a downstream motion-clip stage (LTX-2.3, max 10-12 s per clip). Anchor frames must be deterministic-ish (seedable per shot), period-faithful to a 1980s VHS / CRT / Miracle-Mile aesthetic, and runnable in a subprocess on a single 16 GB laptop GPU.

## Hard constraints

- **GPU:** RTX 5080 Laptop, 16 GB VRAM, Blackwell `sm_120`. Single GPU. No cloud, no API.
- **VRAM ceiling:** 14.5 GB peak real-world target. **Anchor model cannot exceed this.** No weight streaming, no quantization-chasing — Jeffrey's rule is "if it fits, fits; if it doesn't, pick something smaller."
- **Stack:** Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SDPA + SageAttention only. **Flash Attention 2/3 NOT available.** Do not propose anything that requires it.
- **Process model:** All visual generation runs in a sidecar subprocess via `multiprocessing.get_context("spawn")` (C3). Model loads in the worker, not the main ComfyUI process. `CheckpointLoaderSimple` is forbidden in the main graph (C2).
- **Audio invariant:** Episode audio output must remain byte-identical to the v1.5 baseline (C7). The anchor stage cannot perturb the audio path in any way — it runs in parallel and is composited at the end.
- **No paid services, no API keys.** Local weights only.

## Soft constraints (the aesthetic + workflow)

- 1980s VHS / over-the-air UHF / CRT-glow look. Miracle Mile vibes (apocalyptic-but-cozy LA, late-night radio).
- Environments and atmospheres only — characters and faces are a known failure mode (Silent Lip Bug / C6). IP-Adapter for environments only, never characters.
- Anchor frames will be passed to LTX-2.3 for motion. Model output should look "still-frame-from-a-shot," not "trailer poster."
- Prompt source is the deterministic shotlist generator (already shipped in `otr_v2/hyworld/shotlist.py`). Each shot has a camera adjective + scene context.
- Want sane caching: identical prompt + seed should produce the same anchor across runs.
- Want subprocess startup time + per-frame inference time both bounded — a 6-shot episode should produce all anchors in well under 10 minutes wall clock.

## Candidates

I want each model honestly assessed against the constraints + aesthetic, with the trade-offs spelled out:

1. **SDXL 1.0 + a CRT/VHS LoRA stack** — well-trodden, ComfyUI-native, ~10 GB VRAM in fp16. LoRA ecosystem is huge for the period look. fp8 unlocks more headroom.
2. **FLUX.1-schnell** — 4-step distilled, ~12 GB in fp8, strong prompt adherence, weak LoRA ecosystem at the time of writing. Apache 2.0.
3. **FLUX.1-dev** — 12 B params, strong but heavy, may not fit 14.5 GB at fp16. Non-commercial license.
4. **SD 1.5 + LoRA stack** — small, fast, less faithful, huge LoRA library. Cheapest fallback.
5. **Diffusion360 / panoramic SDXL** — pano output may help with environment establishing shots.
6. **Anything else relevant in 2026** that I'm missing — please name it.

## What I want from you

For each viable candidate, give me:

1. **VRAM footprint** at typical OTR resolution (1024 wide for env shots, fp8 / fp16 / bnb-nf4 as appropriate).
2. **Aesthetic match** for the SIGNAL LOST look — strengths and known weaknesses.
3. **LoRA / IP-Adapter compatibility** in 2026 — is the period-style LoRA ecosystem actually there?
4. **Per-frame wall-clock** estimate on RTX 5080 Laptop (or comparable).
5. **License + practical risk** (license, model availability, weights still distributed).
6. **Subprocess load behavior** — typical first-load time, whether it will VRAM-spike during the load itself.

Then **rank them** for the Phase B anchor role and **name your top pick**. If you'd reach for something I didn't list, say so and justify it.

Be honest about disagreement with the other model's answer (your reviewer will read both). Don't hedge for politeness — Jeffrey wants the most grounded answer, not the most agreeable one.
