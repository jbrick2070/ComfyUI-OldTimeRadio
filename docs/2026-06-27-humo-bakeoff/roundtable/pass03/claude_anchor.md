# Claude anchor -- r3 (wiring / integration / sequencing), written before the panel

VERDICT: the R2 slices are wireable into the existing harness with NO eng_humo edit, but
three wiring details decide success: (1) the honest VRAM meter must run IN the server
process, (2) the no-LoRA leg changes graph STRUCTURE so it must be env-driven not literal-
patched, (3) the GGUF leg needs a distinct loader node whose MODEL output must still accept
the LoRA patch + WanHuMo audio cross-attn.

## Wiring per slice (grounded in the harness + _build_graph)
- **A. Honest VRAM meter (in-process).** The runner submits over HTTP /prompt, so it
  CANNOT see the server's `torch.cuda.max_memory_allocated`. WIRE a second sibling node
  `OTR_BakeoffVramProbe` in `custom_nodes/otr_bakeoff_helper` as an IMAGE passthrough on
  the VAEDecode->SaveImage edge that, on execute, prints
  `torch.cuda.max_memory_allocated()/memory_reserved()` (lazy torch import; always-dirty
  IS_CHANGED) so the per-leg peak is logged where the runner already greps the server log.
  The existing OTR_BakeoffReclaim resets nothing memory-wise, so the post-decode probe
  captures the true sampler+decode high-water. KEEP the external nvidia-smi number too
  (report both: true-allocated vs reserved).
- **B. GGUF loader leg.** `_build_graph` node `unet` = `UNETLoader{unet_name,weight_dtype}`.
  The builder must EMIT a different node for the gguf leg: `UnetLoaderGGUF{unet_name:
  <gguf>}` feeding the SAME downstream (`lora.model <- gguf` then ModelSamplingSD3). Do it
  in the BUILDER (translate-time), not eng_humo. WIRING RISK: `UnetLoaderGGUF`'s MODEL must
  still accept `LoraLoaderModelOnly` (lightx2v) + the WanHuMo audio cross-attn -- the
  1-frame smoke is the gate.
- **C. No-LoRA / steps leg = ENV-driven, not literal-patched.** Unlike `cfg` (one KSampler
  literal), dropping the LoRA changes STRUCTURE: `_build_graph` only omits the `lora` node
  and rewires `modelsampling.model <- unet` when `skip_lora` (OTR_HUMO_LORA_NAME in
  none/skip/off). So the runner must SET `OTR_HUMO_LORA_NAME=none` + `OTR_HUMO_STEPS=<n>`
  (or the 17B-namespaced vars for the 1.7B tier) in the build-time env per leg, then build,
  then restore -- NOT post-patch the prompt. Verify via the manifest (lora=None, steps=n).
- **Frame matrix wiring.** Runner generates `(leg x frames)` from frames=[49,97,177];
  `build_leg_prompt(frames=)` already exists; thread the frame count into label / manifest /
  frames_prefix / out_clip / gates so the three points don't collide on disk.

## Invariants to guard (wiring)
- New probe node: always-dirty IS_CHANGED, lazy torch import (cold-import clean), pure
  passthrough (no tensor mutation), SFW. Lives in the sibling pack (NOT the OTR pack).
- Output stays ALWAYS-SILENT; SaveImage terminal unchanged; the reclaim node still asserts
  sampler survival.
- The harness stays HTTP/diagnostic; any PROMOTION re-expresses through
  `wrapper_bridge.run_graph` + workflow JSON + 16gb_full profile in ONE change (deferred).

## Sequencing (kill-gated)
A (meter + alloc-conf A/B across the matrix) -> if true-allocated <13.5 STOP (14B promotable)
-> else B (gguf feasibility 1-frame smoke -> matrix) ; C (mouth ceiling) runs in PARALLEL
as an independent quality track ; D (dep probe) only if mouth needs a model swap.

CUT for r3: any production wiring; an automated mouth metric; building the lip-sync adapter.
