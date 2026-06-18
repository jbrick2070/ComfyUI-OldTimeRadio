<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is not build-ready because [A]/[B] still leave unresolved Comfy node schema, stale sidecar replacement details, exact loader/type names, fail-closed checks, negative plumbing, cleanup, and resolution defaults.

MUST-FIX BEFORE BUILD:
1. [A / z_image_turbo.py] The current grounded `z_image_turbo.py` still gates on `OTR_ZIMAGE_SIDECAR` and `render_image()` raises `NotImplementedError`. That directly contradicts [A] “drop the stale sidecar; run Z-Image in-process.” Concrete fix: replace the sidecar stub with a Lumina-style in-process engine: `_zimage_params()`, `_node_candidates()`, `_build_zimage_graph()`, lazy `wrapper_bridge.resolve_graph_classes()`, `wrapper_bridge.run_graph()`, `images_to_uint8()`, and return `frames[0]`.

2. [A / Z-Image facts] `CLIPLoader(qwen_3_4b.safetensors, type=<z-image/qwen>)` is not a buildable spec: `<z-image/qwen>` is a placeholder, not an exact ComfyUI `CLIPLoader` type string. Concrete fix: verify via ComfyUI `/object_info` or the installed node schema and write the exact literal `type` value into the plan and implementation. If there are multiple candidates, add ordered candidates only after verifying the real class/input names.

3. [A / Z-Image facts] The graph is only described as “clone `lumina_image.py`,” but Z-Image-specific node/input compatibility is still an open question in Q1. `ModelSamplingAuraFlow`, `EmptySD3LatentImage`, `UNETLoader`, `CLIPLoader`, `VAELoader`, and their input keys must be exact or `wrapper_bridge.run_graph()` will fail at runtime. Concrete fix: add a checked Z-Image graph spec equivalent to `lumina_image._build_lumina_graph()`, including exact node classes, output slots, and input keys. Mark any unverified node as a blocker, not a question.

4. [A / z_image_turbo.py] The fail-closed usability gate is wrong after dropping the sidecar. Grounded code checks only `OTR_ZIMAGE_SIDECAR`; the in-process split-file plan needs model-file gates. Concrete fix: define `MODEL_ENV`, `CLIP_ENV`, and `VAE_ENV` for Z-Image, e.g. `OTR_ZIMAGE_UNET`, `OTR_ZIMAGE_CLIP`, `OTR_ZIMAGE_VAE`, with defaults only if Comfy can resolve basenames. Update `assert_usable()` to fail closed on at least the diffusion model path, and preferably all three files, instead of checking a sidecar python.

5. [A] The plan omits the post-render residency cleanup required by the existing in-process engines. Grounded `flux_gen1.py` and `lumina_image.py` both call `_wb.reclaim_idle_models(...)` in a `finally` block. Concrete fix: add the same `finally: _wb.reclaim_idle_models(reason="z_image_turbo post-decode")` around the Z-Image graph run, otherwise the image model can remain resident before downstream video work.

6. [B / C] The plan depends on a live negative prompt at `cfg=2.0`, but does not define actual negative-prompt plumbing or a default. Saying “e.g. oversaturated…” is not enough for a build. Concrete fix: add `OTR_ZIMAGE_NEGATIVE` to `_zimage_params()` and set an exact default string, or explicitly default it to empty and state negative tuning is not part of the first build. If the intended default is active, write the exact string.

7. [B / Goal] Resolution handling is not exact despite the stated goal requiring “the exact engine config.” [B] says “honor request w/h” and also “consider snapping toward Z-Image’s ~1MP sweet spot if quality suffers.” That is not an implementable rule. Concrete fix: choose one initial rule. Smallest drop-in fix: honor request `width`/`height` exactly, use env defaults when absent, and do no snapping/upscale in the engine. If snapping is required, specify the deterministic aspect-preserving formula and where it happens.

8. [B] The scheduler default is internally inconsistent with the proposed Lumina/AuraFlow clone. Grounded `lumina_image.py` uses `scheduler="normal"` with `ModelSamplingAuraFlow`; [B] sets `OTR_ZIMAGE_SCHEDULER=simple` to match Flux. Concrete fix: either default to `normal` for the AuraFlow-style graph, or verify that `simple` is valid and better for Z-Image with `ModelSamplingAuraFlow` before making it the default. Do not justify it as “clone Lumina” while changing this core sampling parameter.

9. [A / B] [ASSUMPTION] The memory target is unresolved and may break the stated deployment tier. The document says Z-Image is for the “sub-8GB tier” / commercial-clean lane, but the proposed split-file path uses `z_image_turbo_bf16.safetensors` plus `qwen_3_4b.safetensors` and VAE. A 6B bf16 diffusion model alone is roughly 12 GB before text encoder/VAE/runtime overhead. Concrete fix: specify the actual intended weight precision and expected resident VRAM on the target 5080; if the bf16 split-file graph exceeds the single-resident ceiling, use an FP8/quantized model or change the tier claim.

SHOULD-FIX:
1. [A] Add the same cold-import discipline as grounded Flux/Lumina: no torch/comfy imports at module scope; only import `wrapper_bridge` inside `load()` / `render_image()`. This is implied by “in-process” but not stated in the build steps.

2. [A] Match the peer method signature unless the dispatcher is verified otherwise. Grounded `FluxGen1ImageEngine.render_image(self, request, prepared=None)` and `LuminaImage2Engine.render_image(self, request, prepared=None)` accept `prepared=None`; grounded Z currently requires `prepared`. Concrete fix: use `prepared=None` in the rewritten Z engine.

3. [A] Define model basename/path behavior. Grounded Lumina uses `os.path.basename()` before passing names to Comfy loaders. Concrete fix: specify whether Z loader params pass basenames or absolute paths, and make it match Comfy folder resolution.

4. [B] Reconsider default `shift=3.0`. It is a guess plus “sweep 1-6.” Concrete fix: pick a small first-pass set, e.g. 3.0 and 6.0 only, then lock one default after raw PNG comparison.

5. [C] Keep prompt reuse outside the engine implementation. The engine should accept a prompt string; prompt naturalization should be a separate upstream composition option so content-addressing/determinism stays understandable.

6. [C] The proposed negative includes “blurry,” while the positive tails include film grain / broadcast distress. That may suppress desired analog softness if Qwen interprets it broadly. Concrete fix: start with negatives aimed at color/material only, e.g. “oversaturated, glossy, clean digital, plastic skin, waxy skin, sterile studio lighting, cartoon, illustration, text, watermark,” and add blur only if Z outputs soft images.

7. [Questions / validation] The “smallest A/B ladder” is asked but not specified. Concrete fix: define a fixed seed set, fixed prompts, and one-variable matrix before build validation; otherwise the operator cannot converge reproducibly.

OPTIONAL / NICE-TO-HAVE:
- Add Z-specific log line mirroring Lumina/Flux: width, height, seed, steps, cfg, shift, sampler, scheduler.
- Add `engine_version` bump only if any prompt-construction behavior changes outside the engine.
- Add a CPU unit test for `_zimage_params()` and `_build_zimage_graph()` like the Lumina graph can support.

CUT THESE (over-engineering):
1. [C] Cut the optional natural-language rewrite from the first build. It adds another variable before the base graph/knobs are proven. Safe to cut because the stated first strategy is reuse `compose_still_prompt` as-is.

2. [D] Cut “possibly a light post grade” from the initial writing/parameter plan. It is not an engine parameter or prompt-construction rule, and it will mask whether Z-Image sampling/prompting actually matches Flux.

3. [B / Q5] Cut snap-and-upscale for the first build. Honor request dimensions exactly first. Safe to cut because Flux parity depends on aspect/dimension consistency, and upscale introduces a second model/process path.

4. [B] Cut the broad `shift` sweep `1-6` for initial validation. Use two or three fixed candidates after the graph is working. Safe to cut because a wide sweep delays finding build/runtime failures in the core adapter.