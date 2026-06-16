<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan contradicts itself on deleting vs keeping `ltx_orbit`, leaves the current 2B/T5 loader path partially alive, and does not specify enough graph/registry/role changes to produce the stated one-engine GGUF LTX surface.

MUST-FIX BEFORE BUILD:
1. [3.0 vs 3A] Contradiction: Phase 0 says delete `LtxOrbitEngine`, but [3A] says "`ltx_orbit` (`LtxOrbitEngine`): no change needed — inherits everything; keep its no-auto-loop override." These cannot both be true. Concrete fix: remove the [3A] `ltx_orbit` bullet entirely and state that all `ltx_orbit` behavior, docs, registry row, exports, tests, and workflow/profile references are removed in Phase 0.

2. [3.0] Wrong sequencing: Phase 0 says to delete the dead 2B/T5 recipe scaffolding inside `LtxVideoEngine` before Phase 1 splices GGUF/Gemma. In the grounded code, `LtxVideoEngine` currently depends on `_ckpt_path()`, `_text_encoder_name()`, `_installed()`, `CheckpointLoaderSimple`, `CLIPLoader`, and `W("checkpoint", 2)` VAE outputs. Removing that scaffolding in a separate “green” Phase 0 commit without adding the GGUF/VAE/Gemma replacement will leave `ltx_video` broken. Concrete fix: Phase 0 should only delete `ltx_orbit` and references. Move all 2B/T5 removal into the same Phase 1 commit that adds the GGUF/Gemma/VAE graph.

3. [0, 3.0, 3A] The stated “three uses” / “per-character” goal is not matched by the grounded `LtxVideoEngine.roles`. Current `ltx_video` roles are `("scene_broll", "background_abstract", "music_visual", "announcer_visual")`; `character_video` exists only on `LtxOrbitEngine`. If `ltx_orbit` is deleted, no shown LTX engine supports `character_video`. Concrete fix: either add `"character_video"` to `LtxVideoEngine.roles` and verify its requests supply `text_prompt`, or remove “per-character” from the goal and workflow expectations.

4. [3A] The graph rewiring is underspecified and will leave invalid VAE wires. Current `_build_graph()` and `_build_graph_i2v()` use `W("checkpoint", 2)` as the VAE input for `VAEDecode` and `LTXVImgToVideoConditionOnly`. `UnetLoaderGGUF` will not provide the old checkpoint tuple with VAE at slot 2. Concrete fix: add a separate graph node, e.g. `"vae": {"class": "vae", "inputs": {"vae_name": ...}}`, then change every VAE wire to `W("vae", 0)`: `img2vid.inputs.vae`, terminal decode input, and any other checkpoint-slot-2 references. Also update `keep={...}` and patcher extraction if the model node key changes from `"checkpoint"`.

5. [3A] `load()` / `_installed()` are not included in the required replacement, but grounded `load()` still calls `_installed()`, and `_installed()` currently checks the old 2B checkpoint plus T5 encoder. Updating only `assert_usable()` is insufficient. Concrete fix: replace `_ckpt_path()`, `_ckpt_name()`, `_text_encoder_name()`, `_installed()`, `_assert_stack_ready()` messages, and `load()` error text with GGUF UNet + Gemma encoder + distilled LoRA + video VAE checks.

6. [3A, 7.1] Runtime-gate policy is unresolved. [0] says “no runtime gate”; [3A] says remove/flip `OTR_ENABLE_LTX_VIDEO`; [7.1] reopens keeping it as a kill switch. Grounded code still has `requires_flag = "OTR_ENABLE_LTX_VIDEO"` and `assert_usable()` raises when the env is `"0"`. Concrete fix: decide before coding. For the clean-break rule as written, remove the usability flag gate and set `requires_flag = None` or equivalent. If an ops kill switch is mandatory, document it explicitly as an accepted exception and update [0].

7. [3A] The canvas claim is false against the grounded engine defaults. [3A] says “Canvas 832x480 ... already correct,” but grounded constants are `_LTX_DEFAULT_W = 768` and `_LTX_DEFAULT_H = 512`. Concrete fix: either change engine defaults to 832x480 or specify the exact request/workflow/render_driver path that always passes 832x480 for every LTX shot. If relying on render_driver, add a test that `LtxVideoEngine._dims()` receives 832x480 for announcer/music/per-beat requests.

8. [3A, 7.4, 7.5] The build-critical node inputs and model filenames are still open questions. The plan names `LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed, ltx-2.3-22b-dev)`, `UnetLoaderGGUF`, `VAELoader`, and `LoraLoaderModelOnly`, but does not give the exact input names expected by the installed Comfy nodes. Current code uses `{"clip_name": ..., "type": "ltxv"}` for `CLIPLoader`; that will not necessarily match `LTXAVTextEncoderLoader`. Concrete fix: before coding, capture `/object_info` or `INPUT_TYPES()` for every target node and put the exact graph input dicts in [3A]. [ASSUMPTION] Without that capture, the first GPU run is likely to fail on unexpected/missing keyword arguments.

9. [3A] “Distilled LoRA unconditional wiring” is not enough unless the candidate set and missing-file behavior also change. Grounded `_node_candidates_sampling()` only includes `"lora"` when `_use_distilled_lora()` returns true, and `_use_distilled_lora()` silently omits the LoRA with only a warning if the file is missing. Concrete fix: for the target 22B GGUF path, always include `"lora"` in candidates, make the LoRA file a hard `MISSING_MODEL` failure in `_installed()` / `assert_usable()`, and remove the “rendering WITHOUT it” path.

10. [3.0, registry.py CAPABILITIES] The registry capability row cannot simply drop the old `ltx-video-2b` model requirement. Grounded `CAPABILITIES["ltx_video"]["model_requirements"]` is `["ltx-video-2b"]`; the plan says delete the constant but does not specify the replacement row. Concrete fix: replace it with the actual new asset ids, e.g. GGUF UNet, Gemma encoder, distilled LoRA, and video VAE, and update `vram_estimate_mb` if Q4/Q3 changes the peak. Also delete only the `ltx_orbit` row.

11. [5] License metadata is not reconciled with code. The plan says the new path is MIT-friendly / license-clean, but grounded `LtxVideoEngine.commercial_clean = False`. Concrete fix: either set the adapter/profile metadata to the correct clean value or document why this field intentionally remains false and verify no commercial-clean profile filter blocks `ltx_video`. [ASSUMPTION] This matters if `commercial_clean` participates in profile selection.

SHOULD-FIX:
1. [2, 3A] Target recipe says `ManualSigmas(8-step distilled)`, but grounded code currently uses custom in-adapter `_SigmasFromValues` and injects `"sigmas"` after resolver. Decide whether exact workflow parity requires real `ManualSigmas`. Concrete fix: either add `ManualSigmas` to candidates and graph with verified inputs, or explicitly state `_SigmasFromValues` is the accepted equivalent and keep it.

2. [3B, 4] “Ensure the FLUX `radio_bookend` still is generated/stamped” is not a concrete test. Current `_use_i2v()` falls back to text-to-video if no on-disk init image exists. Concrete fix: add a regression that builds an announcer/music LTX request and asserts `asset_refs["still"]` or equivalent exists on disk before `render_clip()`.

3. [6] License validation is only described as a rule, not a gate. Concrete fix: add a simple workflow/code scan in validation for banned class names: `ClownSampler_Beta`, `MultimodalGuider`, `VHS_VideoCombine`, and any RES4LYF/VHS package references.

4. [3A] Stale docstrings/comments will become actively misleading. Grounded `eng_ltx_video.py` has many comments saying v0.9, 2B, T5, `CheckpointLoaderSimple`, and default-off/dark. Concrete fix: update the module docstring and graph comments in the same Phase 1 commit so future operators do not follow the old recipe.

5. [7.2] Q4 vs Q3 default is still open, but [2] freezes Q4_K_S. Concrete fix: pick one default before implementation. If Q3 is allowed, make it an explicit alternate asset with separate VRAM expectation, not an unresolved default.

6. [3C] “Append a GGUF quant knob if needed” conflicts with the clean-break “one engine/one recipe” goal. Concrete fix: do not add a workflow knob for quant unless there is a hard requirement; use code/env defaults for operator experiments.

OPTIONAL / NICE-TO-HAVE:
- Add a CPU-only unit test for graph construction that asserts the new graph contains `unet`, `vae`, `encoder`, `lora`, tiled decode, and no `CheckpointLoaderSimple` / `CLIPLoader`.
- Add a small role-compat test proving `ltx_video` is selectable for every intended role after `ltx_orbit` deletion.

CUT THESE (over-engineering):
1. [7.1] Cut the kill-switch debate unless the operator explicitly accepts an exception to “no runtime gate.” It keeps the old dark-lane behavior alive and directly conflicts with [0].
2. [3C] Cut the JSON quant-pick knob for the first splice. The target recipe is already frozen; adding a user-facing widget creates widget-order risk without being required to replace the engine.
3. [6] Cut “commit and push per green chunk” as a build-readiness requirement for this splice. Keep separate commits if desired, but the load-bearing requirement is passing validation/smoke; push sequencing does not make the code safer.