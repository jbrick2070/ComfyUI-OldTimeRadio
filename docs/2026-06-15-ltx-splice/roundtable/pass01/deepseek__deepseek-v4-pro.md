<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. Phase 0/Phase 1 sequencing is contradictory — the plan would brick the engine before Phase 1 replaces it, making the required green suite impossible. Several other gaps and inconsistencies exist.

MUST-FIX BEFORE BUILD:
1. [Section 3.0] Phase 0 says "delete … the dead 2B/T5/VAEDecode recipe" from `LtxVideoEngine`, yet claims the engine will still serve three uses and pass the suite green before Phase 1 begins. The grounding excerpt shows `_node_candidates` depends on `CheckpointLoaderSimple` and `CLIPLoader`; removing them would break `load()`/`render_clip()`. Fix: Phase 0 must only delete `LtxOrbitEngine` and its references (registry row, capability entry, imports, etc.). The live engine’s graph scaffold must remain until Phase 1 swaps in the GGUF/Gemma recipe. If Phase 0 must remove the old recipe, combine Phase 0 and Phase 1 into a single commit.  

2. [Section 3A] Phase 1 bullet “ltx_orbit (LtxOrbitEngine): no change needed …” contradicts the Phase 0 deletion of `LtxOrbitEngine`. Remove that bullet.  

3. [Section 3A] The plan sets `_sampler_mode()` default to `distilled`. The grounding snippet shows the operator reverted that to `ksampler` on 2026‑06‑15 because distilled produced insufficient motion (0.73 framediff vs 7.85). Overriding that without explicit sign‑off risks reintroducing static-looking clips. MUST‑FIX: confirm with operator that the new GGUF recipe warrants the distilled default, or provide a fallback to `ksampler`.  

4. [Section 3.0/3A] The plan does not specify how the engine will resolve paths for the GGUF unet, Gemma encoder, LoRA, and video VAE. Current methods `_ckpt_path()` and `_text_encoder_name()` are to be removed; without replacements, `_ckpt_name()`, `_use_distilled_lora()`, and the new `assert_usable()` checks cannot work. MUST‑FIX: define new path‑resolution logic (e.g., env vars or folder‑search) analogous to what `eng_ltx_av.py` already does for the same model files.

SHOULD-FIX:
1. [Section 3A] Canvas defaults `_LTX_DEFAULT_W` / `_LTX_DEFAULT_H` are `(768, 512)` but the target recipe uses `832×480`. Update the constants.  
2. [Section 4] The BUG‑413 fix says “make the GGUF recipe the real production default (remove/flip `OTR_ENABLE_LTX_VIDEO` dark‑gate)”. The grounding shows the gate already defaults to on (`os.getenv(…, "1") == "0"`). Clarify that no further flag change is needed, only model‑presence checks.  
3. [Section 3C] If the recipe adds no new widget, the plan correctly says no JSON change is needed; if it does, the positional append rule is fine. Ensure this is documented.  

OPTIONAL / NICE-TO-HAVE:
- Open question #2 (Q4 vs Q3 quant for per‑shot batch) should be answered before finalizing the recipe to avoid VRAM surprises across many shots.  
- Explicitly verify that the distilled LoRA remains beneficial at 832×480 (open question #3) and that the GGUF distills enough motion to avoid static‑looking video.  

CUT THESE (over-engineering):  
None. The clean‑break approach is appropriate.  

[ASSUMPTION] The GGUF/Gemma model files are already present on the production box because `ltx_av` engines use the same artifacts. If not, add a note that model procurement must happen before Phase 1 deployment.