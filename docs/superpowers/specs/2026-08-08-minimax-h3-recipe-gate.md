# MiniMax H3 — Recipe Gate (H3 Gate 0)

**Date:** 2026-08-08
**Author:** Claude (Cowork mode)
**Status:** Unblocked, not started. Licensing cleared 2026-08-07 — see `docs/licensing/MINIMAX_H3_AUTHORIZATION.md`.
**Verdict up front:** H3 is worth testing, but it is a **15-second shot generator**, not an episode renderer, and the only VRAM path that plausibly fits our 14.5 GB ceiling is 4-bit. Measure before designing anything.

This document is the testing protocol. It follows the same discipline as the HyWorld Gate 0 in `ROADMAP.md`: **no integration code until the gate produces measured numbers.** Every figure below marked *(reported)* comes from secondary sources and is a hypothesis to falsify on our own hardware, not a fact to build on.

---

## Platform Pins (ground truth — anything contradicting these is wrong)

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, no API keys, no paid services.
- VRAM ceiling: 14.5 GB real-world target.
- **Audio is king. Full narrative output must never break, shorten, or degrade.**
- `trust_remote_code=False` enforced globally on all model loaders.

---

## 1. What H3 actually is

MiniMax H3 (Hailuo 3.0), released 2026-07-31, weights published 2026-08-03. One omni-modal transformer that takes text, images, video and audio and returns **video with native stereo audio, up to 2K, up to 15 seconds per clip**. 33B parameters. ComfyUI shipped day-0 support.

Two access paths exist. Only one is legal *and* pin-compliant for us:

| Path | Pin-compliant? | Licensed for us? |
|---|---|---|
| MiniMax platform API (`MiniMax-H3`) | ❌ — violates "no API keys, no paid services" | n/a |
| **Local open weights** | ✅ | ✅ **only** via the 2026-08-07 authorization |

The US is an Excluded Territory under the H3 Community License. Without the authorization on file, the local path would be unlicensed too. Read `docs/licensing/MINIMAX_H3_AUTHORIZATION.md` §5 before shipping anything derived from this gate.

---

## 2. The four hard questions

These decide whether H3 is viable here. Answer them in order; a failure at #1 or #2 ends the gate.

### Q1 — Does any variant fit 14.5 GB?

Reported footprints, worst to best:

| Variant | Disk | Peak VRAM | Source |
|---|---|---|---|
| BF16 base | ~123.6 GB | far over | *(reported)* |
| NVFP4 pruned+quantized | 31.7 GB | **26.9 GB** on RTX 5090 | *(reported)* |
| Smallest official variants | 42.5 GB min download | — | *(reported)* |
| 4-bit NF4 + DiffSynth-Studio | — | **~8 GB claimed** | *(reported)* |
| Community quant/prune collections | — | targeted at 16–24 GB cards | *(reported)* |

The NVFP4 number is the alarming one: **26.9 GB peak is 12+ GB over our ceiling**, on a desktop 5090 with more headroom than our laptop 5080. Only the 4-bit NF4 path is in the running, and an "8 GB minimum" claim almost certainly means 480p/short-frame-count, not 2K/15s. Treat the 8 GB figure as marketing until measured.

Budget note: 42.5 GB of weights is a real disk cost. `models/` is gitignored, so this never touches the repo — but confirm free space before downloading.

### Q2 — Do the quantized kernels exist for sm_120 + torch 2.10 + CUDA 13 on Windows?

H3's memory reductions lean on int8 convrot quantization and custom kernels. **This is the exact trap Flash Attention 2 set on this platform**: the technique exists, the wheel does not. Verify wheel availability for Blackwell laptop + Windows *before* downloading 42.5 GB. Cheapest possible check, highest possible cost if skipped.

### Q3 — Does it load under `trust_remote_code=False`?

README standing rule #7 enforces this globally and says it may not change without explicit security review. If the H3 loaders require remote code, that is a **hard stop and an escalation**, not a flag to flip in passing. Native ComfyUI day-0 support may sidestep this entirely — check the ComfyUI path first.

### Q4 — Where does a 15-second clip fit in a 30+ minute episode?

It does not, directly. H3 is a shot generator. The only sane placement is the **v2 visual sidecar** — see `2026-04-12-otr-v2-visual-sidecar-design.md` — generating per-scene clips against Director output, cut to the existing audio master.

Two consequences follow immediately:

- **Mute H3's native audio.** OTR masters its own 48 kHz stereo from Bark, Kokoro, MusicGen and procedural SFX. H3's native stereo track fights that master and cannot be allowed to shorten or displace narrative audio. Video track only. "Audio is king" is not negotiable for a visual experiment.
- **H3 overlaps HyWorld.** ROADMAP items 3, 4, 6 and 8 are all blocked on HyWorld Gate 0, which is itself unrun. Running two visual bets in parallel on one 16 GB GPU splits attention and VRAM budget. **Decide explicitly which bet leads** before either consumes real effort.

---

## 3. Gate protocol

Run in order. Record peak VRAM and wall time for every step. Stop at the first hard failure and record why.

1. **Kernel/wheel survey.** Confirm quantized-inference support for sm_120 + torch 2.10 + CUDA 13 + Windows. Pass/fail. *(Do this before any download.)*
2. **Disk check.** Confirm ≥ 60 GB free before pulling weights.
3. **Acquire the smallest viable quantized variant.** Pin the exact revision hash — §VII.1 of the license disclaims any support or update obligation, so upstream can move without notice.
4. **Loader smoke test.** Load under `trust_remote_code=False`. Record peak VRAM at load, before any generation.
5. **Minimum-viable generation.** One short clip at the lowest supported resolution and frame count. Record peak VRAM, wall time, and whether the OTR pipeline's own weights can coexist or must be fully evicted first.
6. **Target-shape generation.** One clip at the resolution and duration a real scene sidecar would need. Record the same three numbers.
7. **Handoff test.** Confirm `_flush_vram_keep_llm()` discipline holds — Director weights cleared before H3 runs, H3 cleared before Bark runs. This is where a 16 GB card actually dies.
8. **Audio isolation.** Confirm the native stereo track can be discarded cleanly without affecting video output.

Write results to `docs/superpowers/specs/h3_gate0_results.md`. No integration code before that file exists.

---

## 4. Gate decision

- **All pass** → H3 becomes a candidate for the v2 visual sidecar. Write the recipe, then re-open the H3-vs-HyWorld decision with real numbers on both sides.
- **Fits VRAM but too slow** → park it. Record the wall-clock number; revisit when quantization improves. A visual pass that triples episode render time fails the "queue it and walk away" promise in the README.
- **Q1 or Q2 fails** → stop. Record the measurement, add H3 to the discarded-ideas list with the reason, and let HyWorld Gate 0 proceed unblocked. This is the most likely outcome given 26.9 GB peak on a larger card.
- **Q3 fails** → escalate to explicit security review. Do not flip `trust_remote_code`.

---

## 5. Out of scope for this gate

- Any public release of an H3 node. That is a separate decision with its own licensing review — `MINIMAX_H3_AUTHORIZATION.md` §5 explains why the MIT repo cannot carry H3 rights to downstream users.
- Distributing weights or Model Derivatives through this repo. Never.
- The MiniMax platform API. Violates the offline-first pin.
- Using H3 or its Outputs to train or improve any other model. Prohibited by license §V.3.

---

## 6. References

- `docs/licensing/MINIMAX_H3_AUTHORIZATION.md` — the clearance and its limits
- `2026-04-12-otr-v2-visual-sidecar-design.md` — where H3 would slot in
- `ROADMAP.md` — platform pins, HyWorld Gate 0 protocol this gate mirrors
- MiniMax H3 Community License — `https://huggingface.co/MiniMaxAI/MiniMax-H3/LICENSE`
- ComfyUI day-0 H3 support — `https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui`
