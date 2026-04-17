# Phase B Anchor Model — Synthesis & Decision (REVISED)

**Date:** 2026-04-16
**Round A:** ChatGPT (gpt-5.4), 83 s — full grounded recommendation: SDXL + LoRA stack
**Round B:** Gemini (gemini-3.1-pro-preview) — **HTTP 429**, free-tier daily quota exhausted; no second opinion this round
**Synthesis by:** Claude
**Reality check:** Verified what weights are actually on disk before committing to a path

## Decision (final)

**Phase B v0 ships on SD 1.5 (already on disk), single-pass, no LoRAs.** Revisit SDXL only if SD 1.5's broadcast-grade fidelity reads as *wrong-muddy* on a real OTR shotlist instead of *right-muddy* for SIGNAL LOST.

This intentionally over-rules ChatGPT's first-place pick (SDXL + curated LoRAs) — see "Why I'm pivoting" below.

## What's actually on disk

`C:\Users\jeffr\Documents\ComfyUI\models\checkpoints\`:

| File | Size | Use for anchor gen? |
|------|------|---------------------|
| `v1-5-pruned-emaonly.ckpt` | 4.3 GB | **Yes — Phase B v0 model** |
| `sd3.5_large_fp8_scaled.safetensors` | 15.0 GB | No — exceeds 14.5 GB ceiling at peak |
| `ltx-2.3-22b-distilled-fp8.safetensors` | 29.5 GB | Downstream motion model, not anchor |
| `ltx-2-19b-*` (3 variants) | 27-43 GB ea | Downstream motion model, not anchor |
| `ltx-video-2b-v0.9.safetensors` | 9.4 GB | Downstream motion model, not anchor |
| `stable-audio-open-1.0.safetensors` | 4.9 GB | Audio path, not anchor |

`C:\Users\jeffr\Documents\ComfyUI\models\loras\`:

| File | Size | Use? |
|------|------|------|
| `ltx-2-19b-distilled-lora-384.safetensors` | 7.7 GB | LTX-only; not for anchors |
| `ltx-2-19b-lora-camera-control-dolly-left.safetensors` | 327 MB | LTX-only; not for anchors |

**No SDXL checkpoint, no SD-style LoRAs.** SDXL would need a deliberate download (~7 GB for base + ~150 MB per LoRA). Not blocking, but also not free.

## Why I'm pivoting from ChatGPT's top pick

ChatGPT ranked SD 1.5 #2 with the framing "best fallback / safety valve, not best primary." That ranking assumed SDXL was equally available. It isn't, and the project rules tilt the trade-off the other way:

1. **"Smallest change with the largest payoff"** (CLAUDE.md round-robin section). SD 1.5 is `0` extra setup. SDXL is one HuggingFace download + node-graph wiring + LoRA selection + cache schema commitment.
2. **"If a senior pair-programmer would just do it, just do it"** (CLAUDE.md). Senior pair-programmer faced with SD 1.5 already loaded would write the anchor_gen module against it tonight, see what comes out, then upgrade if needed. They would not block on a download.
3. **SD 1.5's "muddy / broadcast artifact" failure mode is plausibly an asset for SIGNAL LOST.** ChatGPT acknowledged this verbatim: *"the lower-fidelity prior actually helps for 'broadcast artifact' aesthetics."* The SIGNAL LOST aesthetic is literally degraded analog signal — SD 1.5 may be a feature, not a bug.
4. **Massive LoRA library exists for SD 1.5** (ChatGPT also acknowledged). If style needs reinforcement, period LoRAs are cheap and well-understood there.
5. **VRAM headroom is huge.** ~4-7 GB peak vs. SDXL's 9-13.5 GB. That headroom protects us if Phase B work later wants to keep an LLM resident in the same worker, or stack IP-Adapter, etc. Matches Jeffrey's "if it fits, fits" rule.

If after one real-shotlist render SD 1.5 looks too generic-AI-muddy rather than degraded-signal-muddy, the upgrade path to SDXL is well-trodden and ChatGPT's analysis still holds. We have not foreclosed it.

## Where ChatGPT and I agree (locked in)

- **FLUX.1-dev: rejected.** Too heavy, license sketchier, no payoff for this role.
- **FLUX.1-schnell: rejected as default.** Aesthetic too clean for SIGNAL LOST, weaker LoRA ecosystem, near-ceiling VRAM behavior conflicts with Jeffrey's no-dragon rule.
- **Diffusion360: niche tool only, not the default.** Pano output doesn't match "still frame from a shot."
- **Stable Cascade / Sana / PixArt: skip.** Ecosystem maturity in ComfyUI matters more than benchmark novelty for this build.
- **No 5-LoRA stacks ever.** Start with 0-1 LoRA. Three is already too many.
- **Anchor-gen runs in spawned sidecar worker, model loaded once per episode, torn down at episode end.** Matches C3 + the existing worker.py pattern.
- **Cache key:** `(model_hash, lora_set_hash, prompt, negative_prompt, seed, width, height, sampler, steps, cfg)`, SHA-256 → cached PNG path. Same machine + same key = same PNG.

## Concrete plan for the Phase B kickoff session

1. Add `otr_v2/hyworld/anchor_gen.py` — sidecar module, loads SD 1.5 once per episode.
2. Wire `worker.py` to call `anchor_gen.generate_for_shotlist(shotlist)` between shotlist parse and the existing Ken Burns motion stub. Each shot's anchor PNG replaces the current solid-color placeholder as the input frame to Ken Burns.
3. Anchor cache lives at `io/hyworld_out/<job_id>/anchors/<sha256>.png` with a small `cache_index.json` mapping shot_id → cached PNG.
4. **No VRAM coordinator gate.** SD 1.5 fits with room to spare. `_flush_vram_keep_llm()` between phases is sufficient. Holds Jeffrey's no-coordination-expansion rule.
5. **Determinism contract:** same `(seed, prompt, model)` on the same machine + driver = same PNG. Cross-machine bitwise identity is not promised.
6. **Audio path untouched.** Worker runs in parallel with audio pipeline, composited at the end. C7 holds.
7. Generate one real episode, look at the anchors. **Decision gate:** if visual fidelity is the limiting factor on episode quality, source SDXL 1.0 base + 1 period LoRA (Hugging Face, public weights). If it's not, ship Phase B as-is.

## Open items (not blocking Phase B kickoff)

- **Round-robin tool dir bug:** output went to `2026-04-16-2026-04-16-phase-b-anchor-pick` (date double-prefixed). One-line fix in `_consult_round_robin.py`. Low priority, queued.
- **Gemini free-tier quota:** locked out today. Either bill the API for ~$5 or just rely on Claude as second opinion until tomorrow.
