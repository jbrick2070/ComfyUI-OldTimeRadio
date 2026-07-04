VERDICT: yes-with-fixes — close the Step A meter contract, promotion gate, GGUF leg schema, and mouth/no-LoRA acceptance ambiguity before handing to a builder.

MUST-FIX BEFORE BUILD:
1. [Step A] CUDA peak reset/read is still under-specified. A “post-VAEDecode passthrough probe OR server-side log” gives two incompatible implementations, and a post-VAEDecode node alone cannot reset before the sampler. Current harness submits one HTTP prompt at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\run_humo_bakeoff.py:608-611. Concrete fix: define one graph contract: latent passthrough reset node after `WanHuMoImageToVideo`/`OTR_BakeoffReclaim` and before `KSampler`, plus image passthrough read/log node after `VAEDecode` and before `SaveImage`, logging max allocated + reserved. For sentinel, the reset node naturally runs after the LTX sentinel because HuMo is submitted second.

2. [Sequencing / Step A kill-gate] “if true max_memory_allocated < 13.5 GB STOP (14B promotable)” violates the repo’s resident VRAM ceiling unless NVML/reserved behavior is also acceptable. Production ceiling is 14500 MB at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\wrapper_bridge.py:37; the bakeoff currently records NVML peak and only reports `within_box_14500` at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\run_humo_bakeoff.py:649-655. Concrete fix: promotable only if all frame-matrix and sentinel cells have true allocated <=13.5 GB AND NVML peak <=14.5 GB under the same effective allocator env, or the plan explicitly says a production wrapper re-run with that env is required before promotion.

3. [Step B] GGUF harness boundary is not specific enough. The builder currently translates `eng_humo._build_graph()` output, and that graph hardcodes `UNETLoader` + `weight_dtype` at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\eng_humo.py:166-171 and :235-238. Concrete fix: add an explicit harness-only leg schema, e.g. `loader_mode=gguf`, `loader_class=UnetLoaderGGUF`, `loader_param=unet_name`, `unet=<resolved basename>`, and state whether GGUF is inferred from `.gguf` or forced by the leg. Do not leave “mirror eng_wan_i2v” as the implementation contract.

4. [Step C] Mouth/teeth acceptance was lost from the locked plan. Current runner records ffprobe, blue-cast, and soft face metrics only at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\run_humo_bakeoff.py:645-655; none measure mouth interior, teeth realism, lip closure, or sync. Concrete fix: add fixed plosive/vowel audio clips, side-by-side montage output, and an operator rubric gate before any no-LoRA/steps result can be called a mouth win.

5. [Step C] Env namespace is ambiguous enough to test the wrong graph. 14B uses `OTR_HUMO_LORA_NAME`, `OTR_HUMO_STEPS`, `OTR_HUMO_CFG` at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\eng_humo.py:134-140 and :190-195; 1.7B uses `OTR_HUMO_17B_LORA_NAME`, `OTR_HUMO_17B_STEPS`, `OTR_HUMO_17B_CFG`, with wide cfg override `OTR_HUMO_17B_169_CFG` at :497-515 and :536-538. Concrete fix: add a per-tier env table and manifest asserts for lora/steps/cfg actually loaded.

SHOULD-FIX:
1. [Sequencing] “B … C in PARALLEL” should say CPU/build prep may be parallel, but GPU/headless legs run serially with the repo reset rules. The harness uses one :8000 server and resident VRAM.

2. [Promotion wiring] Keep promotion through `config/profiles/widget_mapping.json`, not raw node-id patching only. The mapping for `role_overrides.other_beats_visual` targets `OTR_VideoDirector.other_beats_video_model`; node 92 is only `slot_overrides.video_render_engine`.

3. [Production-control caveat] If comparing against the shipping production control, include a portrait `humo_1.7B` control leg, not only `humo_1.7B_169`. The profile pins `humo_1.7B` at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\config\profiles\16gb_full.json:14 and :23, while the current bakeoff class map only includes `_169` controls at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\build_humo_bakeoff_workflow.py:67-71.

OPTIONAL / NICE-TO-HAVE:
Add exact manifest fields for CUDA peak stats: reset marker id, allocated peak MB, reserved peak MB, NVML peak MB, baseline MB, and effective allocator env.

CUT THESE:
1. [Sequencing] None beyond the cuts already in the document. The remaining work is gated and tied to the stated fit/mouth goals.

VERIFY-AT-BUILD checklist:
1. Live `/object_info` confirms `UnetLoaderGGUF` exists and its required input is `unet_name`; no `gguf_name`, no `weight_dtype`.
2. GGUF leg manifest proves the resolved loader class/param/file match the built prompt.
3. GGUF min-frame smoke proves `WanHuMoImageToVideo` accepts the GGUF-loaded model with audio conditioning.
4. Frame matrix `[49, 97, 177]` runs for viability legs; every result asserts `frame_count == manifest.length`.
5. Step A records CUDA allocated/reserved peaks after a same-prompt reset, plus NVML peak and effective `PYTORCH_CUDA_ALLOC_CONF`.
6. Sentinel resets CUDA peak after LTX and before HuMo.
7. No-LoRA/steps legs assert actual lora absence, steps, cfg, and model tier in the manifest.
8. Mouth review uses fixed plosive/vowel clips and operator side-by-side rubric.
9. Any promotion re-expresses the winner through the in-process `wrapper_bridge.run_graph` path, edits `workflows\otr_scifi_16gb_full.json` and `config\profiles\16gb_full.json` in the same change, then runs workflow validator, JSON round-trip, link/widget audit, suite, Bug Bible, and B7.