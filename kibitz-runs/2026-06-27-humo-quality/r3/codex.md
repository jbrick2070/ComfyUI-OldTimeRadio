VERDICT: no. Live contracts disagree with the plan in the GGUF loader, runner manifest path, LoRA-disable wiring, and promotion target.

MUST-FIX BEFORE BUILD:
1. [Step B] Defect: GGUF input contract is wrong. The installed `UnetLoaderGGUF` takes required `unet_name`, not `gguf_name`; it also has no `weight_dtype`. See `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF\nodes.py:135-143` and `:150`. Concrete fix: emit `{"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": <gguf file>}}`, and validate against live `/object_info`. Existing OTR GGUF code uses this contract in `nodes\_otr_video_engines\eng_wan_i2v.py:215-218`.

2. [Step B] Defect: runner validation is hard-wired to `UNETLoader`, so a GGUF leg will crash or false-fail even if the builder is fixed. `assert_checkpoints()` checks `("UNETLoader", "unet_name", meta["unet"])` in `scripts\run_humo_bakeoff.py:319-327`; `build_manifest()` dereferences `_find(prompt, "UNETLoader")` in `scripts\run_humo_bakeoff.py:361-365`. Concrete fix: add `loader_class` / `loader_param` to `meta`, use `UnetLoaderGGUF.unet_name` for GGUF, and make manifest extraction loader-agnostic.

3. [Step C] Defect: `lora=none` is not a literal rewrite like `cfg`. The current graph includes `LoraLoaderModelOnly` when the engine default has a LoRA; disabling it requires deleting that node and rewiring `ModelSamplingSD3.model` from LoRA output back to UNET output. See cfg-only rewrite in `scripts\build_humo_bakeoff_workflow.py:164-171`, LoRA inclusion in `nodes\_otr_video_engines\eng_humo.py:232-234` and `:263-271`. Concrete fix: implement a per-leg no-LoRA graph branch or post-build node deletion plus rewire; update `meta["lora"]` and checkpoint assertions.

4. [Step A] Defect: the proposed “probe node on the latent edge” is ordered before the heavy sampler if placed like `OTR_BakeoffReclaim`; it will not capture sampler peak. The latent splice is `WanHuMoImageToVideo slot 2 -> reclaim -> KSampler.latent_image` in `scripts\build_humo_bakeoff_workflow.py:145-154`, and the actual HuMo submit happens after optional sentinel in `scripts\run_humo_bakeoff.py:602-611`. Concrete fix: reset peak stats immediately before each HuMo prompt, then log `max_memory_allocated()` / `memory_reserved()` after KSampler or after prompt completion. For sentinel, reset after the LTX sentinel and before the HuMo prompt if the number is meant to be HuMo-specific.

5. [Per-idea promotion edit] Defect: the named promotion target does not match live production wiring. `config\profiles\16gb_full.json:11-24` has `other_beats_visual: "humo_1.7B"` and `video_render_engine: "humo_1.7B"`, not `humo_1.7B_169`; the saved workflow has node 87 `other_beats_video_model` as `visualizer (16:9)` and node 92 `engine` as `humo_1.7B` in `workflows\otr_scifi_16gb_full.json:1`. Also episode mode ignores node 92’s `engine` widget and renders from the ShotLock ledger path: `nodes\otr_video_render_batch.py:127-134`. Concrete fix: promote by setting `role_overrides.other_beats_visual` to the winning engine and applying that into node 87; update node 92 only for single/soak parity, not as episode authority. Do not change announcer unless intentionally moving it off LTX.

SHOULD-FIX:
1. [Step B] “ONE-FRAME smoke” is not actually one frame through this harness. `build_leg_prompt()` quantizes with HuMo min/max in `scripts\build_humo_bakeoff_workflow.py:130-131`, and HuMo min is 33 frames in `nodes\_otr_video_engines\eng_humo.py:53-54`. Rename this to a min-frame smoke or create a separate verified 1-frame diagnostic path.

2. [Frame matrix] Current result gates only require `frame_count > 0`, so a 49/97/177 matrix run can pass with the wrong saved frame count. See `scripts\run_humo_bakeoff.py:636-655`. Concrete fix: assert `result["frame_count"] == manifest["length"]` for each matrix cell.

3. [Step A] The allocator A/B knob must be recorded in the manifest; otherwise results are not traceable. `boot_server()` builds an env in `scripts\run_humo_bakeoff.py:190-201`, but `build_manifest()` currently records model/render fields only in `scripts\run_humo_bakeoff.py:371-385`. Add `OTR_BAKEOFF_ALLOC_CONF` and effective `PYTORCH_CUDA_ALLOC_CONF`.

OPTIONAL / NICE-TO-HAVE:
Use the existing GGUF loader-mode pattern from `nodes\_otr_video_engines\eng_wan_i2v.py:155-172` and `:215-218` as the local implementation model.

CUT THESE (over-engineering):
1. [Step B] Separate “GGUF file exists on disk” gate as a standalone build gate. The live `UnetLoaderGGUF` dropdown/object_info check is the real loadability contract; raw file existence can pass while folder_paths cannot resolve it.

2. [Per-idea promotion edit] Treating `OTR_VideoRenderBatch.engine` as an episode promotion gate. In episode mode it is bypassed by `_render_episode()`; keep it only as a single/soak default if needed.