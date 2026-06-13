<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The implied 22B/gemma/i2v build path is not asset-complete, not wired in the shown adapter, and has no validated VRAM/length acceptance path; the safer v0.9 motion fix still needs target-resolution decode validation.

MUST-FIX BEFORE BUILD:
1. [Reference Goofer] / [Q1] vs [eng_ltx_video._node_candidates] / [eng_ltx_video._build_graph]: The document treats the 22B distilled + Gemma path as runnable, but the shown adapter still builds the v0.9/T5 graph: `CheckpointLoaderSimple` + `CLIPLoader(type="ltxv")` + `KSampler`/`SamplerCustomAdvanced`. The excerpt explicitly says the 22B Gemma path “requires different ckpt path + different graph topology for LTXVBaseSampler; left as operator-gated.” Concrete fix: either remove 22B from this build and scope the build to the existing v0.9 graph, or add a separate verified 22B graph using the installed wrapper’s actual node classes/widgets for Gemma loading, transformer loading, base sampler, VAE, and i2v conditioning. Do not reuse the current T5 graph for 22B.

2. [On disk already] / [Q1]: The selected 22B distilled file is “transformer ONLY,” but the asset list does not include a separate LTX 2.3 VAE, and the current graph gets the VAE from `W("checkpoint", 2)`. A transformer-only file will not satisfy that contract as written. Concrete fix: name the exact VAE file already on disk, add its loader to the 22B graph, and smoke-test `VAEDecode`; otherwise do not select the transformer-only 22B variant.

3. [Hard constraints] / [Q1]: The <=14.5 GB live ceiling is not enforceable from the plan as written. The document asks what block-swap/sequential-offload/tiled-VAE/fp8-on-the-fly is required, but no concrete offload mechanism, node settings, or acceptance command is specified. The shown `render_clip` only visibly calls `assert_vram_within_ceiling` after `run_graph` and encoding; verify whether `wrapper_bridge.run_graph` samples peak NVML during the forward. Concrete fix: define the exact offload strategy and add a smoke that records peak host NVML across text encode, sampling, and VAE decode, failing if peak >14.5 GB.

4. [On disk already] / [Q2] vs [Hard constraints]: The camera-control dolly LoRA is not on disk, while the hard constraints require 100% local/offline-first. Any build path that depends on that LoRA cannot run offline as written. Concrete fix: either exclude the camera LoRA from the first build, or add a pre-build asset step that places the exact LoRA filename under the expected local LoRA directory with license/hash verification, then fail closed if absent. No runtime download.

5. [Reference Goofer] / [On disk already]: The document conflates at least two different LoRA roles: the on-disk `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` and the missing camera-control dolly LoRA. The shown code only knows about the distilled LoRA via `OTR_LTX_DISTILLED_LORA`; it does not wire a camera-control LoRA. Concrete fix: specify separate load order, filenames, strengths, and compatibility for distilled LoRA vs camera-control LoRA, or remove the missing camera LoRA from the build.

6. [Q3] / [2026-06-12-ltx-motion-sweep.md Recommended LTX default]: The recommended motion unlock is `euler_cfg_pp` plus length 257/305, but the sweep only proves 257/305 at 768x448. The caveat explicitly says 257 at 1472x832 must be decode-validated and that 233 may be the safe ceiling there. The code default still caps/floors to 169. Concrete fix: before changing defaults, run the exact target resolution(s) and lengths: at minimum 768x512x257 for this document’s target, and 1472x832x233/257 if episode output remains relevant. Then update `OTR_LTX_MAX_FRAMES` / `OTR_LTX_MIN_DECODE_FRAMES` or code defaults accordingly.

7. [Reference Goofer] vs [eng_ltx_video.target_fps] / [eng_ltx_video._build_graph]: The reference uses fps 35, but the shown adapter hardcodes `target_fps = 25` and passes `frame_rate=float(self.target_fps)` into `LTXVConditioning`. If the goal is to reproduce Goofer motion, this is a behavioral mismatch. Concrete fix: either set the target to 35 for the comparison path or explicitly keep 25 and rerun the motion sweep at 25 so the selection is not based on a different conditioning fps.

8. [Reference Goofer] / [Q3] vs [eng_ltx_video._build_graph_i2v] / [2026-06-12-ltx-motion-sweep.md]: The document highlights i2v `cond_strength=0.75`, but the grounded current i2v path defaults to `OTR_LTX_I2V_STRENGTH=1.0`, and the sweep conclusion says strength 1.0 stayed on while motion came from `euler_cfg_pp + length`. The code comments also state 0.75 re-noised into mush at 1472x832 for the current path. Concrete fix: do not adopt 0.75 globally. Treat strength as a per-model/per-resolution variable and require an A/B at the selected model/resolution proving both motion and still-anchor preservation.

9. [Reference Goofer]: The Goofer workflow is the main claimed moving reference, but no Goofer source/workflow JSON is included in the grounding. The node names, loader behavior, and memory behavior are therefore not build-verifiable here. Concrete fix: include the exact Goofer workflow JSON or source excerpt and the model/LoRA loader node settings used for the successful clip.

SHOULD-FIX:
1. [Q1] / [Q3]: Add an explicit memory order for text encoding: encode prompts/images, release Gemma/T5, then load/run transformer, then tiled VAE decode. The document asks whether this is required but does not decide. If 22B is pursued, this sequencing is probably the difference between fitting and OOM [ASSUMPTION].

2. [Q4]: Do not mention possible “13B-distilled fp8” or “LTX-2 2B successor” as candidate build inputs unless exact filenames, loader support, and local availability are provided. Current grounding provides no such asset or graph.

3. [What we run TODAY] / [2026-06-12-ltx-motion-sweep.md]: MAD alone is not enough for final selection because the document already observed high-MAD warps. Add a pass/fail visual or automated anchor check: first-frame/still similarity, object identity retention, and warp/flicker rejection.

4. [Hard constraints]: Determinism is stated but not specified for the new offload/fp8 path. Add a render-twice test using the same seed and compare frame hashes or bounded pixel deltas. If using dynamic offload or fp8 kernels, deterministic behavior must be proven on the 5080 stack.

5. [eng_ltx_video.render_clip]: Width/height are floored to multiples of 32. 768x512 is safe, but any future requested canvas may silently shrink. Log the final dimensions in the smoke output and include them in the motion/VRAM result table.

6. [Q5]: STG/skip-layer guidance is mentioned but no node support is grounded. Keep it out of the build plan unless the installed wrapper exposes exact nodes/widgets and a smoke proves benefit.

OPTIONAL / NICE-TO-HAVE:
- Add a matrix table with rows for model, encoder, sampler, length, resolution, strength, fps, LoRA set, peak VRAM, MAD, and visual verdict.
- Add explicit environment presets, e.g. `ltx_v09_motion_safe`, `ltx_22b_experimental`, instead of mixing env knobs manually.
- Record ComfyUI wrapper version/commit used for every smoke, not just ComfyUI core version.

CUT THESE (over-engineering):
1. [Q1] / [Q4]: Cut the 22B/Gemma path from the first build. The grounded sweep already shows the existing v0.9 path can reach “REAL” motion at longer length with `euler_cfg_pp`; validating that is smaller and directly compatible with the shown adapter.

2. [Reference Goofer]: Cut the spatial upscaler from model selection. It increases memory/runtime and is downstream of motion generation; it is not needed to decide sampler/model/length.

3. [Q2]: Cut the missing camera-control LoRA from the first offline build. It cannot run under the stated offline constraints until the asset is locally present and wired.

4. [Q5]: Cut STG/skip-layer guidance exploration for this build. No grounded implementation is shown, and it expands the search space before the known sampler/length issue is closed.