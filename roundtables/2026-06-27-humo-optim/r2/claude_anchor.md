CLAUDE ANCHOR -- HuMo r2 (optimization settings + decision gate) -- with a HF-grounded variant finding

VERDICT: the decision gate is answerable, and there is a stack-native "newer variant" lever the 5/21
fp8 path never had: a GGUF quant of HuMo-17B.

CONFIRMED / GROUNDED:
- 5/21 = fp8 14B + lightx2v LoRA + shift 8, staged 16,531 MB -> OOM-thrash (the BUG-265 demote cause).
  The fp8 is the HEAVY form of the quality tier.
- OTR ALREADY runs GGUF UNets (LTX via City96 + the ComfyUI-GGUF loader). So the stack SUPPORTS a GGUF
  HuMo UNet today -- dropping the fp8 14B for a Q3_K_M/Q4_K_M GGUF of HuMo-17B is stack-compatible and
  cuts the UNet VRAM hard (17B Q4 ~9-10 GB vs fp8 14B ~14 GB unet) [ASSUMPTION on exact sizes -> the
  bakeoff measures]. Repos exist: VeryAladeen/Wan2_1-HuMo_17B-GGUF, calcuis/humo-gguf, Alissonerdx.
- The 5/21 blocker was 14B-fp8 + umt5-TE (~5.2 GB, CS-4) co-residency. Free it via lazy TE detach
  (CS-4-open) or a GGUF umt5.

IMPLEMENTATION ANSWER TO THE GATE (ranked):
1. BEST -- swap the fp8 14B for a GGUF HuMo-17B (Q3/Q4) via the existing GGUF UNet loader path (mirror
   the LTX gguf wiring in the workflow JSON, SAME change). This is the "newer variant that runs on the
   stack" the operator asked for -- the 5/21 quality tier (17B-class) at a fraction of the VRAM.
   Add a GGUF-17B leg to the bakeoff; if peak <= ~13.5 GB -> PROMOTE in 16gb_full.
2. + lazy umt5-TE detach / GGUF TE to reclaim the 5.2 GB co-resident block.
3. SETTINGS: extend the bakeoff 3-way to the GGUF-17B -- distill-LoRA 6-step (fast) vs no-LoRA ~25-step
   (max quality); cfg only on no-LoRA; shift 8 default (needs OTR_HUMO_SHIFT to sweep); native 480x832.

NEWER-VARIANT VERDICT: official ByteDance HuMo (1.7B + 17B) UNCHANGED since 2025-09-18 -- no new base
release. The actionable "new variant" = the GGUF-17B quants (stack-native, lower VRAM). Recommend a
GGUF-17B bakeoff leg as the prime path to "stable + HQ + reliable" without the 5/21 thrash. If even the
GGUF-17B can't hold <= ~13.5 GB with the TE, KEEP 1.7B and harden its settings (de-blue cfg applied).

[VERIFY-AT-BUILD] exact GGUF-17B quant sizes + real peak under the AV stack; ComfyUI-GGUF compatibility
with the HuMo audio-conditioning path (HuMo's audio cross-attn vs a plain Wan GGUF loader).
