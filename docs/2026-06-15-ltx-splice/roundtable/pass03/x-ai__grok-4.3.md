<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The splice requires invasive rewrites to node keys, resolver methods, graph builders, VRAM residency, and usability gates that the current eng_ltx_video.py + registry.py implementations do not support and that the plan leaves under-specified.

MUST-FIX BEFORE BUILD:
1. [4A eng_ltx_video.py] _build_graph and _build_graph_i2v both hard-wire "checkpoint" (and W("checkpoint",2) for VAE) plus conditional "lora" only under _use_distilled_lora(); the GGUF splice changes the primary model node to "unet" + unconditional "lora" + separate "videovae". Concrete fix: rewrite both builders to emit the four new loader nodes from 9.3, wire lora.model = W("unet",0), downstream model inputs = W("lora",0), vae inputs = W("videovae",0), and delete every W("checkpoint",*) reference.
2. [4A + 9.5 render_clip:805] keep={"checkpoint", self._TERMINAL} and results.get("checkpoint") will return None after the node rename, dropping the patcher and leaking VRAM. Concrete fix: change to keep={"unet","lora",self._TERMINAL} (or the proven resident key) and model = results.get(<correct key>, (None,))[0]; add the NVML drain assert from the smoke.
3. [4A + 9.3 _installed/assert_usable/_ckpt_path] still only check a single 2B checkpoint + T5; the splice needs four GGUF/Gemma/LoRA/VAE files with min-size floors. Concrete fix: replace with the four new *_name() resolvers + _weight_paths() style checks mirroring eng_ltx_av.py exactly; update the MISSING_MODEL message.
4. [4A _node_candidates + _node_candidates_i2v] still list "checkpoint","encoder","vaedecode"; i2v path inherits the same mismatch. Concrete fix: emit the GGUF set ("unet","videovae","te","lora","vaedecode":("VAEDecodeTiled",)) in both, plus the existing distilled sampling nodes.
5. [9.6 + 4A _use_distilled_lora] still contains the "22b" in _ckpt_name() gate and only adds "lora" conditionally. Concrete fix: delete the method (or neuter to always True), always add the lora candidate, and require the LoRA file in _installed/assert_usable.
6. [4B registry.py:CAPABILITIES] "ltx_video" row still says model_requirements=["ltx-video-2b"] and vram_estimate_mb=12500. Concrete fix: update to the GGUF + Gemma + LoRA + video VAE list and re-measure post-smoke.
7. [Phase 0 + Phase 1] LtxOrbitEngine inherits from LtxVideoEngine and is deleted in Phase 0 while Phase 1 mutates the base class. Concrete fix: delete LtxOrbitEngine + its registry entry first (Phase 0), then perform the base-class splice (Phase 1) in a single commit after green suite.

SHOULD-FIX:
1. [5 + 9.8] The motion-acceptance smoke is defined as a 2×2×3 matrix but the plan never states the exact frame-diff threshold or how a per-role fallback to ksampler is wired back into _sampler_mode(). Add the numeric floor and the fallback logic.
2. [9.9] commercial_clean remains False with an open "verify-at-build" note; registry protocol already reads it. Set the value (or file the one-line ticket) before the splice commit.
3. [4A canvas defaults] _LTX_DEFAULT_W/H are still 768/512 in the source. The plan changes them to 832/480; perform the assignment.
4. [9.7] The anti-regression test must be an exact-tuple assert (no "VAEDecode" etc.); implement it as a unit test that would have caught the old node names.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line comment in _build_graph_i2v noting that it calls the new _build_graph and therefore inherits the GGUF wiring automatically.
- Document the four new OTR_LTX_VIDEO_* env vars in the module docstring.

CUT THESE (over-engineering):
1. §4D "no GGUF-quant-widget" warning — the JSON contains no LTX nodes, so the paragraph is dead weight and can be deleted with zero behavioral change.
2. All "verify-at-build only if episode trouble" language in §8 for ModelSamplingLTXV / temporal_size / device=cpu — these are already scoped to post-smoke perf notes in 9.10 and do not need to appear as build steps.