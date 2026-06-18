<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan leaves critical implementation choices unspecified (CLIP loader default, tiled decode gating, sampler whitelist values) and includes out-of-scope guards that contradict the “Lightning LoRA is CUT” statement.

MUST-FIX BEFORE BUILD:
1. [P2 Portability] Specify the default CLIP for the floor. The plan says “a GGUF umt5 encoder … or an fp16 umt5 safetensors” but never picks one. The floor must have a single default. Fix: state that the floor default is the GGUF Q5_K_M (`umt5-xxl-encoder-Q5_K_M.gguf`) with `CLIPLoaderGGUF`, and that the loader class will be selected by a new env `OTR_WAN_TI2V_CLIP_LOADER` (gguf|safetensors) defaulting to gguf, analogous to the UNET loader mode. Update `_node_candidates` to use `("CLIPLoaderGGUF",)` when clip loader is gguf.
2. [P1 VRAM] Define the env-gating for `VAEDecodeTiled`. The plan says “env-gated, default-on for the floor tier” but gives no env var name or detection of “floor tier”. Fix: introduce `OTR_WAN_TI2V_VAE_DECODE_TILED` (default `1`). In `_node_candidates`, if enabled, set `"vaedecode"` to `("VAEDecodeTiled",)`; otherwise `("VAEDecode",)`. Document that the floor tier expects this on.
3. [P2 Portability] Specify the sampler whitelist. The plan says “core euler + simple” and “drop uni_pc/sa_solver/MoEKSampler” but does not list allowed values. Fix: define the whitelist as `["euler"]` (the only portable, deterministic sampler for the floor). Add a check in `assert_usable` that raises `EngineUnusable` if `OTR_WAN_TI2V_SAMPLER` is set to a non‑whitelisted value.
4. [P1 VRAM] Define `_TI2V_DEFAULT_FRAMES` explicitly. The plan says “add `_TI2V_DEFAULT_FRAMES` and use it as the fallback” but omits the value. Fix: set `_TI2V_DEFAULT_FRAMES = 17` (the new 8GB‑tier minimum) and change the fallback in `render_clip` from `self.target_fps` to `_TI2V_DEFAULT_FRAMES`.

SHOULD-FIX:
1. [P3 Reliability] Remove the CFG/steps coupling guard for distill. Distill LoRA is cut from the floor (Lightning LoRA is CUT), so this guard is dead code. If needed later, add it when distill is introduced.
2. [P3 Reliability] Remove the license guard for LoRA. No LoRA path exists in the floor; the guard is dead code.
3. [P1 VRAM] Change `VramPeakProbe` interval from `1.0` to `0.1` seconds as the plan suggests, to capture finer peak memory spikes.
4. [P2 Portability] Verify that `VAEDecodeTiled` is available in the target ComfyUI environment (core or custom). If it is a custom node, document the dependency explicitly; otherwise the floor may break on a stock ComfyUI install.

OPTIONAL / NICE-TO-HAVE:
- OOM retry logic (P3). If the new defaults (17 frames, tiled decode, GGUF CLIP) already keep peak under 8 GB, the retry is unnecessary. The existing `assert_peak_within_ceiling` provides a hard fail. Adding retry complexity is not required for the floor goal.

CUT THESE (over-engineering):
1. CFG/steps coupling guard (P3) – distill not in floor, dead code.
2. License guard for LoRA (P3) – no LoRA path, dead code.
3. OOM pre‑flight estimate / catch‑and‑retry (P3) – the primary VRAM reductions should prevent OOM; the existing peak assertion suffices. This adds complexity without clear benefit for the floor.

[ASSUMPTION] ComfyUI‑GGUF custom node provides `CLIPLoaderGGUF` and supports umt5 GGUF files (plan states “same dep family, already required” – verify).
[ASSUMPTION] `VAEDecodeTiled` is available in the target ComfyUI environment (plan does not specify source – verify).