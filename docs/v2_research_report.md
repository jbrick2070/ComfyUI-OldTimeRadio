# Architectural Crossroads for Generative Video Pipelines
## Deterministic 3D Simulation Versus Latent-Space Diffusion

> **Author:** Jeffrey Brick  
> **Date:** 2026-04-10  
> **Status:** Research Phase — informing OldTimeRadio v2.0 Visual Drama Engine  

---

## Executive Summary

This report deconstructs the two primary architectures for translating algorithmic
text into synchronized audio-visual outputs within a 16GB VRAM envelope:

- **Path A: Deterministic 3D Sim-Renderer** (Blender/Unity) — asset-driven, infinite
  temporal scaling, absolute character consistency, low VRAM.
- **Path B: LTX 2.3 Latent Diffusion** (22B DiT) — cinematic photorealism, bounded
  temporal context (~20s), extreme VRAM management required.

Both paths mandate an **audio-first architecture** where the narrative timeline is
mathematically locked before visual rendering begins — a pattern OldTimeRadio v1.5
already implements.

---

## Key Architectural Insights

### 1. Audio-First Is Non-Negotiable

Both paths require the audio timeline (dialogue, SFX, environmental beds) to be
fully generated, normalized to 48kHz, and timestamped before visual rendering
consumes VRAM. OTR v1.5 already does this.

### 2. JSON State Machine > Natural Language Prompting

Natural language introduces unacceptable variance in temporal logic, anatomical
continuity, and environmental persistence. The LLM Director must output rigid JSON
manifests that drive all downstream nodes deterministically.

OTR's existing `production_plan_json` pattern extends naturally to `visual_plan`.

### 3. VRAM Power Wash Protocol

The report validates OTR's existing sequential handoff pattern:
- Pre-emptive offload before loading new models
- Immediate migration/deletion after task completion  
- Active gc.collect() + torch.cuda.empty_cache() reclamation

This maps exactly to `_flush_vram_keep_llm()` and `force_vram_offload()`.

### 4. Path A Advantages (3D Sim-Renderer)

| Advantage | Detail |
|---|---|
| **Absolute consistency** | Rigged meshes cannot hallucinate structural changes |
| **Infinite clip length** | Load ending geometric state as frame 0 of next scene |
| **Low VRAM** | Fits inside 8-12 GB budgets |
| **Deterministic lip-sync** | Rhubarb NG maps audio → visemes → blendshapes |
| **Multi-character native** | Independent meshes, no cross-contamination |

**Tradeoff:** Requires upfront asset library (models, rigs, environments).

### 5. Path B Advantages (LTX 2.3 Diffusion)

| Advantage | Detail |
|---|---|
| **Cinematic quality** | Photorealistic from base weights |
| **Zero assets needed** | Generates geometry on the fly |
| **FeatureExtractorV2** | Separated audio/video embeddings for latent lip-sync |
| **IP-Adapter anchoring** | CLIP Vision embeddings force character identity |

**Tradeoffs:**
- 22B params requires FP8/GGUF quantization on 16GB
- ~20s context window; recursive chaining needed for longer
- "Silent Lip Bug" from aspect ratio tensor misalignment
- Multi-character scenes need timestamped localized masking

---

## Decision Matrix

| Feature | Path A (3D) | Path B (LTX 2.3) |
|---|---|---|
| Visual Consistency | Absolute | Fragile (IP-Adapter/LoRA dependent) |
| VRAM Footprint | Very Low (8-12 GB) | Extreme (14.5+ GB with FP8) |
| Clip Length | Infinite | Bounded (~20s, recursive chaining) |
| Lip Sync | Reliable (Rhubarb) | Temperamental (aspect ratio sensitive) |
| Aesthetic | Stylized/animated | Cinematic/photorealistic |
| Setup Friction | Heavy upfront (assets) | Heavy runtime (debugging) |

---

## Hybrid Architecture (Proposed for OTR v2.0)

The report's analysis suggests the optimal path is BOTH forks composed:

1. **Path A** for character rendering (deterministic, consistent, low VRAM)
2. **Path B** for establishing shots / backgrounds (cinematic, no consistency issue)
3. **Composite** via FFmpeg layers, synced to audio timeline
4. **CRT overlay** as aesthetic post-process (existing v1.5 capability)

This preserves character consistency (Path A's strength) while delivering cinematic
environments (Path B's strength), all within 16GB via sequential VRAM handoff.

---

## Integration Points with OTR v1.5

| OTR v1.5 Component | v2.0 Extension |
|---|---|
| `production_plan_json` | Add `visual_plan` section with scene descriptions, camera, characters |
| `force_vram_offload()` | "VRAM Power Wash" — same pattern, extended to 3D/diffusion models |
| SceneSequencer | Add visual track (item type 6) alongside audio items |
| `video_engine.py` (FFmpeg) | Composite 3D + diffusion layers over audio timeline |
| 48kHz audio master | Direct input to Rhubarb (Path A) or FeatureExtractorV2 (Path B) |

---

*This document is the foundational reference for all v2.0 architectural decisions.
No code will be committed until the alpha spike proves feasibility.*
