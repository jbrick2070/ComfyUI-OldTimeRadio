<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan lacks critical implementation details (CLIPLoader type, model-path env vars, negative-prompt env var, concrete default resolution) and assumes compatibility of AuraFlow shift / Qwen3 CLIP type without verification. Not build-ready as-is.

MUST-FIX BEFORE BUILD:
1. [A. Architecture / B. Starting knobs] The plan does not specify the `CLIPLoader` `type` argument for the Qwen3-4B text encoder; the placeholder “<z-image/qwen>” is not a known ComfyUI type. Without the exact string, the encode will fail. Fix: determine and document the correct type (e.g., from Z-Image’s official ComfyUI workflow) and hardcode it in the Z-Image engine graph definition.
2. [A. Architecture / B. Starting knobs] Model weights must be located but no environment variables are listed for the diffusion model, CLIP, and VAE files. The plan must define `OTR_ZIMAGE_UNET`, `OTR_ZIMAGE_CLIP`, `OTR_ZIMAGE_VAE` (or equivalent) with default basenames, mirroring the `lumina_image` pattern. Without them, the engine cannot instantiate loaders.
3. [B. Starting knobs / D. Look-match risk] The plan intends to use a negative prompt but does not provide an environment variable to supply it. The engine code needs `OTR_ZIMAGE_NEGATIVE` (or similar) to feed the `neg` CLIPTextEncode node, as done in `lumina_image` with `OTR_LUMINA_NEGATIVE`.
4. [A. Architecture] Application of `ModelSamplingAuraFlow` to Z-Image Turbo is an unchecked assumption. Verify that Z-Image’s sigma schedule is indeed AuraFlow-compatible; if not, the generated images will be distorted. The fix: confirm against official Z-Image ComfyUI nodes or documentation; if a different shift node is required, replace it in the plan.
5. [B. Starting knobs] The default resolution rule “~1MP equivalent at Flux's aspect … consider snapping” is too ambiguous to code. Define explicit default width/height (e.g., 1024×1536 portrait, 1536×1024 landscape, or 1024×1024 square) and whether aspect-ratio matching or cropping is performed, so the engine is deterministic.
6. [D. Look-match risk] The mention of a “light post grade” is unresolved – is it an in-engine step or external? If it belongs inside the engine, specify the exact post-processing (e.g., LUT, color-matrix) and its control; otherwise, remove it from the engine spec to prevent ambiguity.

SHOULD-FIX:
1. [B. Starting knobs] Provide the exact default values for all `OTR_ZIMAGE_*` knobs (steps, cfg, shift, sampler, scheduler) as the initial running configuration, informed by a minimal verified smoke test, rather than starting values that need a sweep.
2. [C. Prompt strategy] Since Qwen3 may behave poorly with comma-tag prompts, add an optional natural-language rewrite step (controlled by an env flag, e.g., `OTR_ZIMAGE_NATURALIZE=1`) and a concrete rewrite rule, to avoid a dead-end at first run.
3. [B. Starting knobs] Explicitly document that the engine must call `reclaim_idle_models()` after decoding (like `flux_gen1` and `lumina_image`) to respect the single-resident VRAM ceiling. The plan’s “cloning lumina” implies it but does not state it.
4. [B. Starting knobs] Specify the `weight_dtype` for `UNETLoader` – the plan’s “bf16” weights may require explicit `"bf16"` rather than `"default"` to avoid forced upcast and possible OOM.

OPTIONAL / NICE-TO-HAVE:
- Add a resolution-snapping logic that adjusts input dimensions to the model’s native latent aspect ratio, gated by an env flag (`OTR_ZIMAGE_SNAP_RES=1`), as a future optimization.
- Document the expected VRAM budget and working set size for the in-process engine to help operators plan concurrent workloads.

CUT THESE (over-engineering):
1. [B. Starting knobs] The suggestion to upscale when resolution deviates from sweet spot is unnecessary complexity for the initial build. Cut it; upscaling can be a downstream tool.
2. [D. Look-match risk] Remove the “light post grade” from the engine configuration; the engine should output raw PNGs. Grading belongs in the post‑pipeline, not inside the prompt‑to‑image adapter.

Mark [ASSUMPTION]:
- That ComfyUI’s CLIPLoader supports a Qwen3-4B type with a specific string; no grounding excerpt shows this.
- That ModelSamplingAuraFlow is the correct sigma-shift node for Z-Image Turbo.
- That EmptySD3LatentImage’s latent dimensions are compatible with Z-Image’s VAE and DiT.
- That the Qwen3 encoder may prefer natural language over comma-tag prompts – unverified and based on external knowledge about LLMs.