# Roundtable pass01 judgment -- LTX 2.3 motion model on a 5080 16GB

Panel: gpt-5.5, gemini-3.1-pro, grok-4.3 (+ Claude as panelist/judge). Spend ~$0.14.
All three returned **VERDICT: no** to the "just swap to 22B" framing. Strong convergence.

## CONFIRMED (grounded -- accepted)
1. **The 22B-distilled fp8 (23.5 GB) does NOT fit OTR's 14.5 GB resident ceiling**
   (gpt+gemini+grok, unanimous). 23.5 GB transformer + 8.8 GB gemma >> 16 GB card.
   Goofer runs it on a 16 GB 5080 ONLY via DynamicVRAM block-swap/streaming -- which
   means peak may stay under the card but each step streams a 23.5 GB model =
   very slow, and it breaks OTR's "single resident heavy <=14.5 GB" invariant for
   a multi-beat episode. => The 22B is impractical for OTR's per-clip pipeline.
2. **The 22B file on disk is transformer-ONLY** (no bundled VAE/CLIP); the current
   graph pulls VAE from the checkpoint loader (`W("checkpoint",2)`). A 22B path
   needs a separate LTX-2.3 VAE + the gemma CLIP loader + a different sampler node
   (`LTXVBaseSampler`) -- a NEW graph, not the v0.9 graph. None of that is wired in
   `eng_ltx_video.py` today (grok cited the exact `_node_candidates`/`_ckpt_path`).
3. **The camera-control dolly LoRA is a 19B/22B LoRA, not on disk, and not wired**
   (all three). It is model-matched -- it will NOT apply to the 2B v0.9 base. So it
   cannot rescue v0.9; it only matters if we move to the matching LTX-2.3 base.
4. **fps drift**: OTR hardcodes `target_fps=25`; Goofer uses **35**. Feeds
   `LTXVConditioning.frame_rate` -- a real conditioning difference (gpt).
5. **257 DOES decode at 1472x832** -- I validated it live this session (MAD 5.95,
   clean decode, ~8 min/clip). Resolves the panel's "233 safe ceiling" caution.

## MISREAD / where the panel is uninformed (rejected or qualified)
- gemini+grok say "v0.9 already reaches REAL motion (MAD 4.2-5.3) via
  euler_cfg_pp+length, so stay on v0.9." They grounded on the MAD doc but did NOT
  see the clips. **Operator eyeball: those long v0.9 clips WARP / morph and look
  nothing like Goofer; MAD rewarded the warp.** So "v0.9 already solves it" is
  REJECTED -- MAD is necessary but not sufficient; the visual is the gate.
- gemini "0.75 = red mush, keep 1.0": that code comment is for v0.9 @ 1472x832.
  Goofer uses 0.75 @ 768x512 on the 22B. Strength is per-model/res -- do not adopt
  globally (gpt agrees). UNVERIFIABLE on the target model -> A/B at build.

## THE REAL TENSION (synthesis)
Real LTX-2.3 motion lives in the 22B; the 22B busts OTR's VRAM ceiling. v0.9 fits
but its motion is poor (operator-confirmed). The camera LoRA is model-matched, so
it can't help v0.9. => Neither extreme works.

## RECOMMENDATION (Claude, judge)
**Target the 13B-0.9.7-distilled fp8 -- the LTX-2.3 motion model that FITS 16 GB.**
- It is Goofer's own documented "optional, higher quality" model and the natural
  sweet spot between v0.9 (fits, weak motion) and 22B (strong motion, won't fit).
  fp8 13B is ~13 GB-class -> can sit resident under ~14.5 GB with the encoder
  offloaded first (encode -> free gemma/t5 -> load transformer -> tiled VAE decode).
- NOT on disk -> one download (HuggingFace `ltxv-13b-0.9.7-distilled-fp8`), with
  hash + license recorded, fail-closed if absent (no runtime download). Operator
  lifted the strict-local rule, so a one-time fetch is allowed.
- Prove it in the smoke FIRST (the fast harness already exists): 13B fp8 + the
  distilled chain (cfg 1.0 + the 8 sigmas) + gemma encoder + i2v ConditionOnly,
  strength A/B {1.0, 0.75}, length {97,169,257}, fps 35, 768x512. Gate on the
  OPERATOR EYEBALL (real motion + still preserved + no warp), not MAD alone.
- Only if 13B fp8 also fails the ceiling/motion bar do we consider the 22B with
  explicit block-swap (accepting it breaks the resident-ceiling invariant for an
  opt-in "max quality" lane, operator-gated).
- Build hygiene the panel is right about: encode->free-encoder->load-transformer
  sequencing; record peak NVML next to MAD in the smoke; per-model/res strength
  A/B; expose OTR_LTX_LENGTH; do not wire the missing 19B camera LoRA into the
  base build (separate, model-matched, optional).

## OPEN (operator decision)
Download the 13B-0.9.7-distilled fp8 (~one fetch) to smoke-test the LTX-2.3 motion
that actually fits 16 GB? That is the convergent answer: not v0.9 (too weak), not
22B (won't fit), but the 13B-distilled fp8 in between.
