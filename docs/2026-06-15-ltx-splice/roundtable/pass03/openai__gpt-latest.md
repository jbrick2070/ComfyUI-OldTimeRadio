<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still contains contradictory build instructions and an unimplementable sampler fallback against the grounded engine shape.

MUST-FIX BEFORE BUILD:
1. [§0, §2, §4A, §9.1, §9.2] The document still says “replicates exact node graph + values” and §4A still says “Replace EVERY W('checkpoint',0) with W('lora',0)”, which directly conflicts with §9.1/§9.2 and would create a LoRA self-reference if followed mechanically. Concrete fix: edit §0/§2/§4A so the only allowed graph rewrite is:
   - `unet` node: `UnetLoaderGGUF{unet_name:_unet_name()}`
   - `lora` node: `model: W("unet",0)`, never `W("lora",0)`
   - downstream sampler/guider model inputs: `W("lora",0)`
   - `img2vid.vae` and `vaedecode.vae`: `W("videovae",0)`
   - no “exact graph” wording except model/encoder/decode plus sigma values.

2. [§7, §9.8] “fall back to ksampler for that path/per role” is not implementable as written. Grounding: `_sampler_mode()` reads only global `OTR_LTX_SAMPLER`; it has no request/path/role parameter. Concrete fix: either require one global shipping default whose rows pass the full smoke matrix for all production roles/paths, or explicitly add a request/role-aware sampler selector and class-resolution path. Do not leave “per-path fallback” in the plan unless implementing that mechanism.

3. [§4A, §7, §9.8] The smoke matrix will be invalid if it flips `OTR_LTX_SAMPLER` after `load()`. Grounding: `load()` caches `self._classes = resolve_graph_classes(self._node_candidates_sampling())`; non-i2v `render_clip()` reuses that cache. If the engine is loaded under `ksampler`, then the distilled graph can later be built without cached `samplersel/noise/guider/sampleradv` classes. Concrete fix: specify that each sampler-mode smoke uses a fresh engine/load after setting env, or change `render_clip()` to resolve the active candidate set every time the mode changes.

4. [§4A, §9.3] The GGUF usability gate is underspecified. Grounding: current `assert_usable()` only calls `_assert_stack_ready()`, which checks SageAttention and `_installed()`; node classes are only resolved in `load()`. The new graph depends on `UnetLoaderGGUF`, `LTXAVTextEncoderLoader`, `VAELoader`, `LoraLoaderModelOnly`, and `VAEDecodeTiled`. Concrete fix: add an `assert_usable()` node-class gate mirroring `eng_ltx_av.py` before declaring the engine usable, and make `_installed()`/`assert_usable()` validate all five weight artifacts: GGUF unet, Gemma encoder, projection ckpt, video VAE, LoRA.

5. [§9.3] “min-size floors” are required but not specified for the new `ltx_video` assets. Grounding: `eng_ltx_av.py` defines explicit `_FLOOR_UNET`, `_FLOOR_ENCODER`, `_FLOOR_VIDEO_VAE`, `_FLOOR_AUDIO_VAE`; the splice plan does not give concrete floors for the Q4_K_S unet or LoRA. Concrete fix: define explicit byte floors for GGUF unet, Gemma encoder, projection ckpt, video VAE, and LoRA, then use them in `_weight_paths()`/`_installed()`/`assert_usable()`.

6. [§5] The full-frame i2v requirement can still be defeated by existing key precedence. Grounding: `_init_image_path()` reads `asset_refs` in order `("still", "init_image", "image")`. If `still` contains a portrait and `init_image` contains the full-frame FLUX scene, LTX will use the portrait. Concrete fix: either make render_driver guarantee `asset_refs["still"]` is always the full-frame scene for LTX, or change/add an LTX-specific init-image resolver that prefers the full-frame scene key. Add the regression with both a portrait ref and scene ref present. [ASSUMPTION] The portrait-vs-scene collision depends on render_driver asset naming, which is not grounded here.

7. [§6, §9.7] The banned/residual grep instructions are contradictory and will false-fail. §6 asks for residual `VAEDecode` grep, but the desired node class is `VAEDecodeTiled`. Concrete fix: delete the raw-substring grep language from §6 and require exact structured assertions only: graph/candidate class exactly equals banned names or exactly equals legacy `VAEDecode`/`CheckpointLoaderSimple`/`CLIPLoader`.

8. [§4B, §8, §9.9] `commercial_clean` is still left as “verify/flip pending” in earlier sections even though §9.9 says decide now and the grounded protocol includes `commercial_clean`. Concrete fix: set `LtxVideoEngine.commercial_clean` to the decided value in this splice, or explicitly add a ticket and state profile-filter behavior is out of scope. Do not leave the runtime policy bit as an open build-time guess.

SHOULD-FIX:
1. [§2, §9.3] Normalize exact asset names. §2 names `gemma_3_12B_it_fp4_mixed` and `ltx-2.3-22b-dev` without extensions, while grounded `eng_ltx_av.py` uses `gemma_3_12B_it_fp4_mixed.safetensors` and `ltx-2.3-22b-dev.safetensors`. Concrete fix: use the exact filenames the loader receives.

2. [§4A] Update stale frame/decode comments together with the graph. Grounding: current comments still describe 1472x832 decode behavior, `VAEDecode`, v0.9/2B/T5, and default-OFF/dark. Concrete fix: include comments/docstring cleanup in the same Phase 1 change so future gates do not preserve the wrong operational assumptions.

3. [§3] Phase 0 says remove every `ltx_orbit` reference, but grounded `eng_ltx_video.py` has shared comments mentioning `LtxOrbitEngine` in `_LOOP_VIA_REVERSE_DEFAULT` and `render_clip`. Concrete fix: explicitly include those comments in the grep cleanup, not just class/registry entries.

4. [§4A, §9.5] The VRAM keep-set instruction still says “verify which key holds the live model.” Concrete fix: make the implementation test assert the exact retained patcher key under `free_after_use=True`, then document the final keep set (`{"lora", terminal}` or `{"unet","lora",terminal}`) in the plan.

5. [§4C, §5] render_driver line references are not grounded here. Concrete fix: add “verify:” tasks for `build_request_from_shot`, canvas assignment, and full-frame FLUX still routing instead of treating those line references as established facts.

6. [§7] The motion floor is undefined. Concrete fix: state the numeric threshold and measurement method, or tie it to the prior known framediff target. Otherwise the smoke can pass/fail subjectively.

OPTIONAL / NICE-TO-HAVE:
- Add a small pure unit test for `_node_candidates_sampling()` in both modes proving `lora` is always present and legacy loader class names are absent.
- Add a pure graph-shape test for `_build_graph_i2v()` proving `img2vid.vae == W("videovae",0)` and sampler latent comes from `W("img2vid",0)`.

CUT THESE (over-engineering):
1. [§4D, §9.10] Cut the GGUF-quant-widget warning entirely. The plan already says there are no LTX nodes in the JSON, and §9.10 correctly says this warning is unnecessary.
2. [§8, §9.10] Cut “No ModelSamplingLTXV” from required build steps. It is a verify-at-build/perf/look note, not an implementation step for the chosen mini model/decode splice.
3. [Status] Cut the roundtable/model-name provenance line. It does not constrain build behavior and cannot be verified against source.