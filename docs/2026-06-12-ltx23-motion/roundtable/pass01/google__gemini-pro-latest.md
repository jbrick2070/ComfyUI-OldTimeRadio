<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

- VERDICT: no. The proposed 22B model physically cannot fit in 16GB VRAM, and the grounding proves the existing 2B model already achieves the desired motion via simple config changes.

- MUST-FIX BEFORE BUILD:
  1. [Questions for the panel: 1. VRAM fit] The 23.5 GB 22B fp8 transformer physically exceeds the 16 GB GPU VRAM. It will OOM during sampling even if the Gemma encoder is completely offloaded. Fix: Abandon the 22B model upgrade and retain the 2B v0.9 model.
  2. [What we run TODAY] The claim that v0.9 2B is "motion-limited" is directly contradicted by `2026-06-12-ltx-motion-sweep.md`, which proves 2B v0.9 achieves "REAL dynamic motion" (MAD 4.21 - 5.30) simply by switching the sampler to `euler_cfg_pp` and increasing the length. Fix: Apply the sweep doc's recommendation (`OTR_LTX_SAMPLER_NAME = euler_cfg_pp`).
  3. [The reference that DOES move] The plan proposes `cond_strength 0.75`, but `eng_ltx_video.py` (`_node_candidates_i2v` docstring) explicitly states that at 0.75, the model "re-noises the still into red mush at 1472x832". Fix: Keep `cond_strength` at 1.0, as validated in the sweep doc.
  4. [The reference that DOES move] The plan proposes a length of 257, but the sweep doc's "OPEN CAVEAT" warns that at the production resolution of 1472x832, 257 must be decode-validated, as 233 is the known safe ceiling to avoid a VAEDecode tensor mismatch. Fix: Validate 257 at 1472x832 or cap the length at 233.

- CUT THESE (over-engineering):
  1. [The reference that DOES move] The Gemma encoder, 22B distilled model, and Dolly LoRA. The grounding proves the existing 2B v0.9 + T5 stack already solves the motion requirement when configured correctly, making a 34GB+ model stack completely unnecessary.