<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan drops the sidecar but forgets to update the usability gate, meaning the engine will permanently fail closed looking for a deleted sidecar.

MUST-FIX BEFORE BUILD:
1. [Section A] Defect: Dropping the sidecar breaks `assert_usable` in `z_image_turbo.py`, which currently checks `SIDECAR_ENV` for a python executable. Fix: Rewrite `assert_usable` to check for the new UNET/CLIP/VAE weight files instead, mirroring `lumina_image.py`'s `MODEL_ENV` check.
2. [Section B] Defect: The plan relies on split-file loading but fails to define the environment variables to locate them. Fix: Define and read `OTR_ZIMAGE_UNET`, `OTR_ZIMAGE_CLIP`, and `OTR_ZIMAGE_VAE` in the params dictionary so the operator can point to the weights.
3. [Section A] Defect: Blindly cloning Lumina's `EmptySD3LatentImage` assumes Z-Image uses a 16-channel SD3 VAE. If it uses a standard 4-channel VAE (like Flux's `EmptyLatentImage`), the graph will crash or output noise. Fix: Verify Z-Image's latent channel count and use `EmptyLatentImage` if it is 4-channel.

SHOULD-FIX:
1. [Section A] Defect: The `CLIPLoader` type `<z-image/qwen>` is written as a placeholder. Fix: Identify the exact string ComfyUI expects for Qwen3 in `CLIPLoader` (e.g., `qwen`, `qwen2`), or the graph will fail to build.

OPTIONAL / NICE-TO-HAVE:
- [Section C] Since Qwen3 is an LLM, passing a comma-separated tag string might severely underperform compared to natural language. Consider adding a simple formatting pass if the initial A/B test fails.

CUT THESE (over-engineering):
- None.

[ASSUMPTION] Assuming ComfyUI currently supports Qwen3 natively in its `CLIPLoader`.
[ASSUMPTION] Assuming Z-Image is mathematically compatible with `ModelSamplingAuraFlow` just because it is an S3-DiT. Verify the shift formula matches.