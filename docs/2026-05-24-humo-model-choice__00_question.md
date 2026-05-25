# HuMo model-choice for ComfyUI-OldTimeRadio — round-robin question (2026-05-24)

Round-robin consultation input. Tracked as BUG-LOCAL-265.

## Context

OTR (ComfyUI-OldTimeRadio) is an open-source ComfyUI custom-node pipeline that
generates "old time radio"-style video episodes: a writer LLM produces a
script, then audio (Bark / Kokoro TTS + MusicGen), then FLUX character
portraits, then **HuMo** renders audio-driven lip-sync video clips for each
character line, then VideoComposite assembles the episode.

OTR is published on GitHub and is meant to run on other people's machines, not
only the developer's. Developer rig: RTX 5080 Laptop, 16 GB VRAM, 64 GB system
RAM, Windows, fully local / offline.

HuMo (bytedance-research/HuMo, Wan2.1-based) ships in two sizes: **HuMo-1.7B**
and **HuMo-17B** (also repackaged by Kijai as a "14B" build — the 14B is the Wan
base param count; functionally the same 17B-class model).

## The problem

In the full OTR pipeline the HuMo phase thrashes. HuMo-17B/14B fp8 (~16.5 GB)
loads ~88% offloaded to system RAM on the 16 GB card and renders at
140-279 s/it (a single clip took 3h43m in one soak).

## Evidence — bracket test (bare native ComfyUI HuMo workflow, zero OTR code)

| Model | Settings | VRAM behaviour | Speed | Quality |
|---|---|---|---|---|
| HuMo-17B/14B fp8 (~16.5 GB) | 6 steps + lightx2v distill LoRA | staged 16.5 GB, offloads ~10 GB to RAM | 44 s/it bare; 140-279 s/it in-pipeline | best |
| HuMo-17B Q3_K_M GGUF (9 GB) | 6 steps | only 2.1 GB offload | 145 s/it (GGUF per-step dequant tax) | not judged |
| HuMo-1.7B fp16 (3.3 GB) | 20 steps, distill LoRA bypassed (it is 14B-only) | fully resident, zero offload | 13 s/it | operator rated 20-step output "acceptable" |

Per-clip wall time at each model's proper settings: 14B ≈ 4:25, 1.7B ≈ 4:23 —
essentially identical. So the choice does not turn on speed.

## Root cause of the in-pipeline thrash

The 14B fp8 runs fine *bare* (44 s/it). Inside OTR it thrashes because, when the
HuMo phase loads, the pipeline still holds ~14 GB of VRAM in models loaded
earlier in the run (writer LLM, MusicGen, FLUX) via OTR's own loaders — outside
ComfyUI's `comfy.model_management`, so `unload_all_models()` cannot evict them.
HuMo then gets ~1.7 GB of the 16 GB card. ("Lever 1" = fixing that residue.)

## The decision

Which HuMo model should OTR use, given it ships open-source to varied consumer
hardware?

- **Option A — HuMo-1.7B (low-VRAM).** 3.3 GB, fully resident on a 16 GB card,
  no thrash in or out of the pipeline, ~4:23/clip at 20 steps. Quality is lower
  than 17B but the operator rated a 20-step render "acceptable" and expects it
  to improve with real FLUX reference portraits (the smoke used a generic test
  image). Runs on far smaller cards / less system RAM.
- **Option B — HuMo-17B/14B (high quality).** Best quality, but ~16.5 GB;
  requires a 16 GB+ card AND large system RAM (for the ~10 GB offload). Inside
  OTR it additionally needs the lever-1 fix (free the pipeline VRAM residue
  before the HuMo phase — a VRAM-budget change).
- **Option C — tiered.** Ship HuMo-1.7B as the default (broad compatibility,
  removes the thrash for every user who clones OTR), and expose HuMo-17B as an
  opt-in for high-VRAM users via a model-tier widget on the HuMo node.

## Constraints

- OTR is 100% local, open-source, offline-first, shipped to unknown user
  hardware. CLAUDE.md sets a 14.5 GB VRAM peak ceiling for the dev rig.
- Many users will have less than 16 GB VRAM and less than 64 GB system RAM.
- OTR's other stages (writer LLM, FLUX, MusicGen, the umt5 text encoder
  ~5-6 GB) already make the pipeline VRAM-heavy regardless of the HuMo choice —
  so HuMo-1.7B widens compatibility a lot but does not make OTR a low-end app.

## Questions for the consultant

1. For an open-source pipeline shipped to varied consumer GPUs, is Option C
   (1.7B default + 17B opt-in) the right call, or is there a reason to prefer
   A or B outright?
2. Is the lever-1 VRAM-residue fix still worth doing if HuMo-1.7B becomes the
   default — or does it become low-priority?
3. Anything being missed — HuMo-1.7B quality specifically on audio-driven
   lip-sync, GGUF as a viable middle option despite the dequant tax, or risks
   in the tiering mechanism itself?
