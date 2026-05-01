# Question -- 2026-04-30

# Round-robin question -- 2026-04-30

TWO RELATED ARCHITECTURE DECISIONS for OTR SIGNAL LOST (ComfyUI radio-drama generator on RTX 5080 Laptop / 16 GB VRAM Windows / torch 2.10 / CUDA 13 / Blackwell sm_120):

## QUESTION 1: Animated background layer model pick for 16 GB tier (LTX or alternatives)

**Context:** Adding a 12 fps slow-mo animated background layer (3-layer composite under HuMo lip-sync foreground). The renderer runs SIDECAR -- HuMo unloads before the background model loads, so peak VRAM is just the background model alone, not summed.

**Candidates we've evaluated:**
- **LTX 2.3 (1.1 release)**: underlying 22B params, smallest GGUF Q5_K_M ~14 GB. Fits 16 GB sidecar tightly.
- **LTX-2 19B Kijai distilled**: Q4_K_M ~12 GB. Fits 16 GB sidecar with more headroom.
- **LTX 0.9 2B fp16**: ~5 GB. Fits trivially. Original roadmap pick.
- **Wan 2.2 5B**: 8 GB tier video model. Could reuse but 8 GB tier has no background layer in our final design.

**Constraints:** Distilled 4-8 step, 8n+1 frame counts (LTX has 8x temporal VAE compression), per-scene granularity (1-2 clips per scene of ~8s each), ffmpeg `setpts=PTS*2,fps=12` post-process for vintage slow-mo. Stability matters MORE than max params (this is JUST a background layer underneath foreground HuMo characters).

**Question 1:**
- Which LTX variant is the best 16 GB sidecar choice TODAY (April 2026)?
- Is "1.1" the current LTX 2.3 release, or has Lightricks shipped something newer?
- Any quantization gotchas on Blackwell sm_120 (FP8 / NVFP4 readiness)?
- Are there NON-LTX alternatives we should consider for animated backgrounds at this VRAM budget?

## QUESTION 2: TTS model expansion candidates

**Context:** OTR currently uses Bark + Kokoro for character voices. The pipeline produces master mix audio that drives HuMo lip-sync (audio-conditioned video). NOT replacing the pipeline -- adding more voice models to the palette so users pick per character.

**Memory has these deferred candidates:**
- CosyVoice 2 (Apache-2.0) -- first pick
- Qwen3-TTS -- second
- Fish Speech -- REJECTED (non-commercial license; OTR stays MIT)

**Constraints:**
- License must be MIT-compatible (we don't vendor GPL)
- Must run on 8 GB AND 16 GB tiers (tier-independent)
- VRAM ceiling 14.5 GB
- No cloud APIs (100% local)
- Phoneme control / pronunciation accuracy matters because output drives HuMo lip-sync

**Question 2:**
- As of April 2026, what are the strongest local TTS candidates for vintage radio-drama character voices?
- CosyVoice 2 vs CosyVoice 3 -- what's the current production-grade release?
- Any newer Apache-2.0 / MIT TTS we should consider that landed in the last 6 months?
- Any TTS with explicit period-style controls (1940s broadcast, mid-century radio aesthetic)?
- Any candidates to AVOID (license issues, Windows-only quirks, sm_120 / Blackwell incompatibility, VRAM blowup)?

## For both questions

Prefer the smallest change with the largest payoff. Cite specific HuggingFace repos, version tags, or commit SHAs where possible. Flag uncertainty rather than bluffing.
