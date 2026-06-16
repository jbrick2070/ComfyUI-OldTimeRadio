<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still has contradictory graph requirements and at least one instruction that can create an invalid self-referential LoRA graph.

MUST-FIX BEFORE BUILD:
1. [4A] “Replace EVERY W("checkpoint",0) with W("lora",0)” can make `lora.model` point to `W("lora",0)` if applied mechanically. In the grounded current code, `model_wire = W("checkpoint", 0)` is used as the input to `graph["lora"]`. Concrete fix: specify:
   - `graph["unet"] = ...`
   - `graph["lora"].inputs.model = W("unet", 0)`
   - sampler/guider `model` inputs use `W("lora", 0)`
   - never replace the LoRA node’s own model input with its own output.

2. [4A] The plan does not define separate resolvers/names for the GGUF UNet versus the projection checkpoint required by `LTXAVTextEncoderLoader`. Grounding from `eng_ltx_av.py` shows these are separate: `_unet_name()` returns a `.gguf`, while `_projection_ckpt()` returns `ltx-2.3-22b-dev.safetensors`. Concrete fix: add explicit `LtxVideoEngine` methods/envs:
   - `_unet_name()` / `OTR_LTX_VIDEO_UNET` default `ltx-2.3-22b-dev-Q4_K_S.gguf`
   - `_projection_ckpt()` / `OTR_LTX_VIDEO_PROJECTION_CKPT` default `ltx-2.3-22b-dev.safetensors`
   - `_video_vae_name()` / `OTR_LTX_VIDEO_VAE`
   - `_encoder_name()` / `OTR_LTX_VIDEO_TEXT_ENCODER`
   Then wire `UnetLoaderGGUF{unet_name: _unet_name()}` and `LTXAVTextEncoderLoader{ckpt_name: _projection_ckpt()}`. Do not reuse `_ckpt_name()` for both.

3. [0, 2, 4A] The document contradicts itself on whether the engine must replicate the mini JSON’s exact graph or only swap model/encoder/decode while keeping production sampling. Section 0 says “replicates its exact node graph + values” and “No shim”; Section 4A says “MODEL-ONLY swap — do NOT flip the sampler default” and keep `ksampler`. Concrete fix: rewrite the target contract as one of:
   - “Exact mini graph only when `OTR_LTX_SAMPLER=distilled`; default `ksampler` is intentionally not source-of-truth and must be separately smoke-proven,” or
   - “Exact mini graph is the only production graph; default flips to distilled after gate.”
   As written, a coder cannot satisfy both.

4. [2, 4A] The plan says the source-of-truth distilled chain uses `ManualSigmas`, but the grounded engine currently uses the in-adapter `_SigmasFromValues` shim and injects it after node resolution. That violates Section 0’s “No shim” / exact-node-graph claim. Concrete fix: either:
   - replace `_SigmasFromValues` with a real `ManualSigmas` node candidate and verified inputs, or
   - explicitly state that `_SigmasFromValues` remains an allowed internal compatibility shim and the graph is not exact in that node.

5. [4A, 7] Keeping `ksampler` as the default with `UnetLoaderGGUF + LoraLoaderModelOnly + no ModelSamplingLTXV` is not proven by the provided source. Grounding only shows the current `ksampler` path was built for `CheckpointLoaderSimple`, while `eng_ltx_av.py` uses `ModelSamplingLTXV` after GGUF UNet. [ASSUMPTION] KSampler may or may not accept the GGUF LTX model object directly. Concrete fix: before making this the default path, add an explicit build gate: `OTR_LTX_SAMPLER=ksampler` must render at least one t2v and one i2v LTX clip with the GGUF+LoRA graph, above the motion floor, without `ModelSamplingLTXV`. If it fails, default cannot remain `ksampler` for the new model graph.

6. [6] The proposed grep test for residual `VAEDecode` will falsely fail after the intended change because the new required class is `VAEDecodeTiled`, which contains the substring `VAEDecode`. It will also hit grounded comments/docstrings unless every mention is removed. Concrete fix: make the test exact/structured, e.g. assert no node candidate or graph class equals `"VAEDecode"` and no `("VAEDecode",)` tuple remains; do not grep the raw substring `VAEDecode`.

7. [4A, 5] The i2v graph still has a grounded `W("checkpoint", 2)` in `graph["img2vid"].inputs.vae`. The plan says replace every `W("checkpoint",2)`, but this is easy to miss because `_build_graph_i2v()` overlays the base graph after calling `_build_graph()`. Concrete fix: add an explicit required diff/check: both `graph["img2vid"].inputs.vae` and `graph["vaedecode"].inputs.vae` must be `W("videovae", 0)`, and no `W("checkpoint", 2)` remains anywhere in `eng_ltx_video.py`.

SHOULD-FIX:
1. [4A] “Path/usability rewrite” is too loose for a production fail-closed gate. Grounding from `eng_ltx_av.py` includes node-class resolution and weight-size sanity floors. Concrete fix: mirror that pattern for `ltx_video`: check required node classes in `assert_usable()`, resolve model files through `folder_paths`/fallback, and add minimum file-size floors for GGUF UNet, Gemma encoder, video VAE, and LoRA.

2. [7] The motion smoke is underspecified for the actual decision. It says it proves “distilled isn’t static,” but Section 4A keeps `ksampler` as default. Concrete fix: define a matrix:
   - sampler: `ksampler`, `distilled`
   - path: t2v, i2v
   - role: announcer, music, scene/per-beat
   Default shipping requires `ksampler` pass; default flip requires `distilled` pass.

3. [5] “Full-frame still” cannot be proven by “path exists” and “not portraits dir” alone. Concrete fix: regression must open the image and assert dimensions/aspect match the LTX canvas/full-frame scene still policy, not just the directory name. Also verify: `build_request_from_shot()` and the FLUX still pool behavior, because `render_driver.py` is not included in grounding.

4. [4A] The LoRA behavior should be made mechanically explicit after removing the `"22b" in _ckpt_name()` gate. Concrete fix: delete or neuter `_use_distilled_lora()` so no stale checkpoint-name logic remains, require `_distilled_lora_file()` success in `_installed()`/`assert_usable()`, and always include `"lora"` in `_node_candidates_sampling()` for both sampler modes.

5. [4A] Update `render_clip()` residency text and logic together. Grounding currently keeps/results `"checkpoint"`. Concrete fix: after the graph rewrite, use `keep={"lora", self._TERMINAL}` and `model = results.get("lora", (None,))[0]`; remove or update comments saying `"checkpoint"` is kept for patcher teardown.

6. [3] Phase 0 says delete `LtxOrbitEngine` and its registry/capability entries, but imports that auto-register engines are not shown. Concrete fix: verify no package importer, docs generator, or test expects `LtxOrbitEngine` in `__all__` or in static engine-name snapshots after deletion. [ASSUMPTION]

7. [4B] `commercial_clean` is left as “verify-at-build.” Grounding says the registry protocol reads `commercial_clean`. Concrete fix: decide the value in this plan before the build, or explicitly defer profile-filter behavior out of this splice. Do not leave a runtime policy bit as an open-ended build-time guess.

OPTIONAL / NICE-TO-HAVE:
- [4A] Rename stale `_ckpt_*` methods instead of repurposing them; using `_unet_*` / `_projection_*` will prevent future GGUF-vs-checkpoint confusion.
- [7] Record the measured VRAM peak per sampler mode, not just one “single resident” number, because `ksampler` and `distilled` have different residency profiles.

CUT THESE (over-engineering):
1. [4D] The discussion of appending a GGUF-quant widget to `otr_scifi_16gb_full.json` can be cut. The same section says no JSON node change and env/default is enough; keeping the warning adds no build step.
2. [6] Raw grep over the whole engine for broad strings like `VAEDecode` is too brittle. Replace with structured assertions over node candidates/graph class values; this is safer and avoids false positives from `VAEDecodeTiled` and comments.
3. [8] The speculative Gemma `device="cpu"` / projection-ckpt tweak belongs in a post-smoke performance note, not the build splice. It is safe to cut from the required plan because Section 2’s recipe explicitly uses `device="default"` and Section 8 says not to change it unless a full episode shows trouble.