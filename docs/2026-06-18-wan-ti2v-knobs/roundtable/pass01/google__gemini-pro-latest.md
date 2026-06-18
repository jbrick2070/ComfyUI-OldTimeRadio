<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The code mathematically guarantees an OOM on 8GB cards, and the proposed A/B test candidates require nodes that do not exist in the graph spec.

MUST-FIX BEFORE BUILD:
1. [Current recipe / GPU smoke] **Guaranteed OOM on 8GB.** The smoke test confirms 33 frames @ 832x480 hits a 13.1 GB NVML peak. The grounding code hardcodes `_TI2V_MIN_FRAMES = 33` and clamps the 25fps default to 33. This will instantly OOM the 8GB floor. **Fix:** Change `_TI2V_MIN_FRAMES` to `17` (or `21`, preserving 4n+1) to fit the 8GB ceiling.
2. [Candidate recipes] **Missing LoRA graph wiring.** Candidates B and C require a LoRA, but `eng_wan_ti2v.py`'s `_node_candidates()` and `_build_graph()` completely lack a `LoraLoader` node. The A/B test cannot be run. **Fix:** Add `LoraLoader` to `_node_candidates()` and wire it between `unet` and `modelsampling` in `_build_graph()`.
3. [Candidate recipes] **GGUF + LoRA incompatibility.** ComfyUI's standard `LoraLoader` cannot apply `.safetensors` LoRAs (like LightX2V) to a `UnetLoaderGGUF` model without specialized custom nodes (e.g., `LoraLoaderGGUF`). **Fix:** Either add the specific GGUF LoRA loader to `_node_candidates()` (violating the "core nodes only" constraint) or drop the LoRA candidates for the GGUF-based floor.

SHOULD-FIX:
1. [Questions for the panel - #4] **MPS Sampler crashes.** `uni_pc` and `sa_solver` rely on complex math that frequently causes NaNs or CPU-fallbacks on Mac (MPS). `MoEKSampler` is a custom node, violating the "CORE ComfyUI" constraint. **Fix:** Set the default sampler to `euler` and scheduler to `simple` (or `beta`). Euler is the only mathematically rock-solid cross-platform choice.
2. [Questions for the panel - #5] **VAE Decode VRAM spikes.** Decoding even 17 frames of 832x480 at once in `VAEDecode` can cause a massive VRAM spike at the very end of the render. **Fix:** [ASSUMPTION] Ensure ComfyUI is running with `--vae-vram-kb` limits or that temporal tiling is explicitly enabled for the VAE decode step, otherwise it will fail-closed on 8GB Macs during readout.

OPTIONAL / NICE-TO-HAVE:
- [Questions for the panel - #2] Keep GGUF for the Mac/AMD floor. An fp8 safetensors model upcasts to fp16/fp32 on MPS (which lacks native fp8 support), immediately blowing past 8GB. GGUF dequantizes layer-by-layer and is the only viable 8GB Mac path.

CUT THESE (over-engineering):
1. **Candidates B and C (Lightning LoRA).** Chasing a distill LoRA for the absolute lowest-end floor introduces severe GGUF-patching risks, licensing validation overhead, and graph complexity. Cut them. Candidate E (Euler, lower shift, reduced frame count) is the simplest, most robust path to a working 8GB cross-platform floor.