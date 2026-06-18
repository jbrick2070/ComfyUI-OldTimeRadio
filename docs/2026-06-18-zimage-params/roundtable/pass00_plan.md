# Z-Image-Turbo writing/parameter plan -- match the Flux look (DRAFT for roundtable)

**Goal (operator, 2026-06-18):** configure the `z_image_turbo` image engine so its output is
*visually similar to the production Flux engine* (`flux_gen1`) -- ideally reusing the SAME composed
prompts -- by choosing the right knobs (steps / cfg / sampler / scheduler / shift / resolution /
negative) and deciding how (or whether) to rewrite the prompt text. Produce a converged
**writing-parameter plan**: the exact engine config + the prompt-construction rule.

This is UPSTREAM image work (before video). OTR's image layer is model-agnostic: each engine is a
`prompt -> .png` adapter registered in `nodes/_otr_image_engines`. We want Z-Image to be a
drop-in, look-matched peer of Flux so the operator can pick it per role (and so the sub-8GB tier
has a commercial-clean image lane -- Flux.1-dev is non-commercial).

---

## GROUNDED FACTS (read from the code + web, June 2026)

### Flux (the target look) -- `flux_gen1.py`, PRODUCTION
- Loader: ONE `CheckpointLoaderSimple` on `flux1-dev-fp8.safetensors` (~13 GiB).
- Graph: ckpt -> CLIPTextEncode(pos) + CLIPTextEncode(neg) -> **FluxGuidance(guidance=3.5)** on the
  POSITIVE -> EmptyLatentImage -> KSampler -> VAEDecode.
- Knobs: **steps 20, cfg 1.0, FluxGuidance 3.5, sampler euler, scheduler simple, denoise 1.0.**
  cfg=1.0 means the NEGATIVE prompt is inert; Flux.1-dev takes richness from the guidance embedding.
- Dims: portrait 832x1216 default; landscape 1472x832; request w/h plumbed through (aspect-aware).
- License: Flux.1-dev = BFL **non-commercial** (`commercial_clean=False`).

### The prompt that BOTH engines receive -- `compose_still_prompt` (`_otr_story_brief_helpers.py`)
A comma-joined **5-layer tag string** (NOT natural-language sentences):
`subject, setting(top-2), framing-hint, era-tail(still profile), STYLE_TAIL_DEFAULT[, IMAGE_GRADE_TAIL,
RADIO_BROADCAST_TAIL][, NO_TEXT_CLAUSE]`
- `STYLE_TAIL_DEFAULT` = "cinematic, 35mm film look, subtle film grain, ..."
- `IMAGE_GRADE_TAIL` = "anamorphic lens, heavy vignette, muted color grade, sharp focus"
- `RADIO_BROADCAST_TAIL` = "35mm film grain, broadcast-distressed cinematic aesthetic, centered composition"
- `NO_TEXT_CLAUSE` = "no on-screen text" (scene kinds only; portraits omit)
- Scene stills append both grade tails; portraits keep just STYLE_TAIL_DEFAULT.
This tail stack is what gives Flux the muted, filmic, grainy SIGNAL-LOST look we want to preserve.

### Z-Image-Turbo (the engine to configure) -- facts
- 6B params, Scalable Single-Stream **DiT (S3-DiT)**, Alibaba Tongyi. **Distilled to 8 NFEs (8 steps)**,
  sub-second, fits 16GB. Apache-2.0 (**commercial-clean**).
- ComfyUI-CORE split-file model (NOT a sidecar anymore -- the current `z_image_turbo.py` stub's cu128
  sidecar path is STALE): `UNETLoader(z_image_turbo_bf16.safetensors)` + `CLIPLoader(qwen_3_4b.safetensors,
  type=<z-image/qwen>)` + `VAELoader(ae.safetensors)`. **Text encoder = Qwen3-4B (an LLM)** -- this is the
  big difference from Flux's CLIP/T5.
- Web-recommended knobs: **8 steps; cfg 1.5-2.0** (distilled, so high cfg 4+ hurts; some run cfg 0/1);
  **AuraFlow-style schedule** (`ModelSamplingAuraFlow` shift, exactly like `lumina_image`); **1024x1024**
  best (native 1328x1328); negative prompt usable at cfg>=1.5.
- The closest in-repo template is `lumina_image.py`: UNETLoader + CLIPLoader + VAELoader +
  ModelSamplingAuraFlow + EmptySD3LatentImage + KSampler + VAEDecode (an AuraFlow flow model, same family).

---

## MY DRAFT VERDICT (to be pressure-tested by the panel)

### A. Architecture
Drop the stale sidecar; run Z-Image **in-process** via `wrapper_bridge`, cloning `lumina_image.py`'s
split-file AuraFlow graph (UNETLoader/CLIPLoader/VAELoader + ModelSamplingAuraFlow + EmptySD3LatentImage
+ KSampler + VAEDecode). All knobs env-overridable + request dims honored (V-7 determinism).

### B. Starting knobs (env defaults)
- `OTR_ZIMAGE_STEPS` = **8** (distilled; 8 NFE is the design point)
- `OTR_ZIMAGE_CFG` = **2.0** (so the negative prompt is LIVE -- a lever Flux@cfg1.0 doesn't have)
- `OTR_ZIMAGE_SHIFT` (ModelSamplingAuraFlow) = **3.0** (start; sweep 1-6)
- `OTR_ZIMAGE_SAMPLER` = **euler**, `OTR_ZIMAGE_SCHEDULER` = **simple** (match Flux's pairing; AuraFlow
  shift carries the schedule shape)
- NO FluxGuidance node (Flux-specific). Z-Image gets adherence from real cfg + the negative prompt.
- Dims: honor request w/h; default to a ~1MP equivalent at Flux's aspect (portrait ~832x1216 ->
  consider snapping toward Z-Image's ~1MP sweet spot if quality suffers).

### C. Prompt strategy (the "writing parameter")
Reuse the SAME `compose_still_prompt` output (same subject + tails) so the LOOK tails (grade/grain/
broadcast) carry over. BUT: Z-Image's text encoder is Qwen3 (an LLM), which tends to prefer
natural-language description over comma-tag salad. **Draft call: reuse the Flux prompt AS-IS first**
(simplest, look-matched), and only add an optional "naturalize" pass if A/B shows the tag-string
underperforms. Put the muted/filmic grade into the POSITIVE tail (as today); use the now-live NEGATIVE
to suppress Z-Image's tendency toward over-clean/over-saturated output (e.g. "oversaturated, glossy,
clean digital, plastic skin, blurry") to push it toward the Flux filmic grade.

### D. The look-match risk
Z-Image is photoreal/clean by default; Flux.1-dev with the grade tails is muted/filmic/grainy. The
match is mostly: (1) keep the grade tails in the positive, (2) use the negative to kill clean-digital
sheen, (3) possibly a light post grade. The panel should rank which lever matters most.

---

## QUESTIONS FOR THE PANEL (converge on these)
1. Is in-process AuraFlow (lumina clone) the right architecture, or is anything Z-Image-specific
   needed (e.g. a dedicated `ModelSamplingAuraFlow` shift value, a different CLIPLoader `type`, or a
   latent node other than EmptySD3LatentImage)?
2. Best starting knobs to MATCH Flux's look: steps (8 vs 10-12), cfg (1.5 vs 2.0 vs higher), shift
   value, sampler/scheduler pairing.
3. Prompt strategy: reuse the Flux comma-tag prompt AS-IS, or does the Qwen3 encoder need a
   natural-language rewrite of the same content? If rewrite, give the rule.
4. Negative-prompt content to push Z-Image's clean look toward Flux's muted filmic grade.
5. Resolution: honor Flux's 832x1216 / 1472x832 directly, or snap to Z-Image's ~1MP sweet spot and
   upscale? Quality vs aspect-consistency tradeoff.
6. What's the smallest A/B ladder (one variable at a time) to converge the look-match on the 5080,
   judged on raw PNGs?
```
