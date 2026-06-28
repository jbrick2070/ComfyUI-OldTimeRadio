# OTR resource-optimization + portability -- OPEN-ENDED review (r1)

QUESTION: Is OTR using the MOST optimized model / precision / quant / VRAM-management choices to run
SMOOTHLY on an RTX 5080 (16 GB VRAM) with REAL HEADROOM -- and does it DEGRADE GRACEFULLY so a more
modest box (a person with ~32 GB SYSTEM RAM and a smaller-or-no GPU) can run it too? This is an
open-ended review: read the real repo, ground every claim, propose + critique optimization + portability
levers. (Interpretation note: "32 GB RAM" = 32 GB system RAM / CPU-offload portability, NOT a 32 GB-VRAM
GPU. If the code implies a different natural target, say so.)

## What to assess (ground in the real files)
1. HEADROOM on the 5080. The hard ceiling is `wrapper_bridge.VRAM_CEILING_MB = 14500` of 16 GB = ~1.5 GB
   headroom. Is that enough for stable operation (driver/desktop/other apps), or do the heavy engines
   (HuMo 14B fp8, LTX-AV, Wan) ride the ceiling? Is the ceiling a SAFE default or already tight?
2. MODEL / PRECISION / QUANT choices -- are they the most optimized for quality-per-VRAM?
   - Video: LTX-AV (distilled-1.1 GGUF; the 2026-06-26 bakeoff picked Q3_K_M @ ~15.1 GB -- is that
     leaving headroom or maxing the card? would Q2_K / a smaller quant free headroom at acceptable
     quality?). HuMo 14B fp8 (vs the 1.7B). Wan ti2v.
   - Audio: bark / kokoro / musicgen / stable-audio-3 / indextts2 -- residency + reclaim.
   - LLM: mistral-nemo / gemma-4-12b local (Ollama) / OpenRouter remote -- VRAM vs offload.
3. VRAM MANAGEMENT -- model reclaim/eviction between beats (reclaim_idle_models, free_after_use,
   the BUG-291 detach), the per-engine residency, single-resident discipline. Is anything held that
   could be evicted to widen headroom?
4. PORTABILITY / GRACEFUL DEGRADATION. Is there a real lower-tier path for a modest box? The
   2026-06-10 switchable-workflow plan = 1 master + generated tiers (16gb / 8gb / cpu_floor) +
   capability profiles. Did it SHIP, or is OTR effectively 5080-only? Can a 32 GB-system-RAM box run a
   cpu_floor / 8 GB tier (CPU offload, smaller quants, still-image floor) end to end, or does it OOM /
   require the 5080? What is the realistic minimum hardware?

## Constraints / invariants
100% local default (OpenRouter is opt-in); single resident heavy engine <= 14.5 GB; UTF-8 no BOM; SFW;
don't propose touching the frozen audio spine for headroom. Quality stays the operator's eyeball; this
review is about RESOURCE EFFICIENCY + ACCESSIBILITY, not re-litigating model picks for quality alone.

## Deliver
The optimization levers that would (a) widen 5080 headroom without losing quality, and (b) make a
modest-box (32 GB RAM, small/no GPU) run realistic -- ranked, with the code-grounded gotchas (what is
already handled vs what is missing), and an honest statement of the REAL minimum hardware today.
