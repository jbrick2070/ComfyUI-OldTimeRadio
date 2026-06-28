VERDICT: no. It is a decision prompt, not a build-ready coding plan; the target engine id, VRAM hook, workflow wiring, and peak gate are unresolved.

MUST-FIX BEFORE BUILD:
1. [OPERATOR DECISION GATE] “flip 1.7B->14B” is not an implementable target. There is no registered `humo_14B`; the registered 14B ids are `humo` and `humo_14B_169` in `nodes/_otr_video_engines/registry.py:141,156` and `nodes/_otr_video_engines/eng_humo.py:80,557`. Current workflow node 87 has `other_beats_video_model="visualizer (16:9)"`, while node 92 has `engine="humo_1.7B"` in `workflows/otr_scifi_16gb_full.json`. Fix: state the exact target id and aspect: `humo_14B_169 (16:9)` for the current 16:9 other-beats path, or `humo (portrait)` if portrait is intended; patch both `config/profiles/16gb_full.json` and the real workflow nodes in the same change.

2. [r2 questions 1 / lazy umt5 detach] The proposed “detach after conditioning, before HuMo forward” does not happen today. `HuMoEngine.render_clip` calls `_wb.run_graph(graph, classes)` without `free_after_use` at `nodes/_otr_video_engines/eng_humo.py:349`, then calls `reclaim_idle_models` only after decode at `eng_humo.py:361`; that is too late to reduce sampler peak. Fix: make HuMo use the existing executor release path, e.g. `_wb.run_graph(graph, classes, free_after_use=True, keep={"unet","lora","modelsampling", self._TERMINAL})`, or add an explicit executor hook after `pos`/`neg` and before `humo`/`ksampler`. Add a CPU test that asserts HuMo uses `free_after_use` and does not dangle retained MODEL patchers.

3. [Deliver / expected peak] The plan asks for peak <= ~13.5 GB, but HuMo does not currently measure render-window peak. It only does an instantaneous post-reclaim guard via `_MC.assert_vram_within_ceiling("humo-render")` after `reclaim_idle_models` in `eng_humo.py:361-365`; other engines use `VramPeakProbe` and thread `vram_peak_mb` through, e.g. `nodes/_otr_video_engines/eng_ltx_av.py:677,694,701`. Fix: wrap HuMo `run_graph`/encode with `motion_common.VramPeakProbe`, return `vram_peak_mb`, include it in `canonicalize`, and gate promotion on measured peak <= 13500 MB.

4. [r2 questions 1 / GGUF TE or UNET] GGUF is not supported by the current HuMo graph. HuMo candidates are `UNETLoader`, `CLIPLoader`, and safetensors names in `eng_humo.py:172,186-198`; LTX’s GGUF path is a different graph using `UnetLoaderGGUF` in `eng_ltx_video.py:568` / `eng_ltx_av.py:456`. Fix: either cut GGUF from this build, or first verify real `/object_info` node classes and add HuMo-specific GGUF loader candidates plus tests. Do not assume “like LTX” applies.

5. [r2 questions 2] Several requested tuning knobs are not interfaces. `shift` is hardcoded to `8.0` in `eng_humo.py:269-271`; 14B quant is effectively the hardcoded/default UNET filename in `eng_humo.py:73,116`; 1.7B only overrides unet/lora/steps/cfg in `eng_humo.py:497-515`. Fix: define actual per-tier methods/env vars for any knob to be tuned, or explicitly mark it non-tunable and leave it out of the implementation.

SHOULD-FIX:
1. [Constraints / invariants] The hard code path defaults to a 14.5 GB ceiling, not the promotion target of 13.5 GB: `motion_common.py:40,47,219`. Registry estimates for `humo` and `humo_14B_169` are 14000 MB in `registry.py:141,156`. Fix: run the promotion smoke with `OTR_VRAM_CEILING_MB=13500` or add a separate promotion assertion at 13500 MB.

2. [r2 questions 1 / BUG-291 drop] “How much did real peak already drop” has no specified harness or baseline artifact. The existing HuMo tests explicitly say live VRAM is not covered in `tests/test_video_humo.py:11-12`. Fix: name the exact headless workflow run, log location, and metric source before using any peak comparison to decide promotion.

3. [r2 questions 3] “NEWER MODEL CHECK” is not code-grounded from the repo. verify: current external model availability, license, ComfyUI wrapper support, Blackwell/cu13 support, and local smoke. Keep this as a separate research gate, not a coding prerequisite for the HuMo fix.

OPTIONAL / NICE-TO-HAVE:
Add a small regression that parses `workflows/otr_scifi_16gb_full.json` and asserts the profile’s `role_overrides.other_beats_visual` and `slot_overrides.video_render_engine` agree with node 87/node 92 after profile application.

CUT THESE (over-engineering):
1. [r2 questions 1] Cut GGUF HuMo TE/UNET from the first build. No current HuMo loader path supports it, and the root candidate fix is early encoder release using existing `run_graph(free_after_use=True)`.
2. [r2 questions 2] Cut “14B no-LoRA ~25-step” from the promotion path. It is not the stated 5/21 quality target, increases runtime materially, and does not solve the umt5 co-residency defect.
3. [r2 questions 3] Cut adopting a newer model in this build. A research probe is fine; swapping the production talking-face engine is a separate dependency/license/wrapper validation project.