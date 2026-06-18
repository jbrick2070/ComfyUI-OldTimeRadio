<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is solid for an 8GB floor, but misses a critical variable update in the code that will allow upstream requests to bypass the frame clamp and OOM.

MUST-FIX BEFORE BUILD:
1. [Section 3] Defect: The plan says "clamp the floor tier so an upstream target_frame_count can't push 33+". However, `eng_wan_ti2v.py` defines `_TI2V_MAX_FRAMES = 177`. If you only change `_TI2V_MIN_FRAMES` to 17, `_wb.quantize_frames_4n1` will still allow requests up to 177 frames, causing an immediate OOM on 8GB. Fix: Change `_TI2V_MAX_FRAMES = 17` in the module scope (or introduce an env override for the max).
2. [Section 1] Defect: The plan mandates "OFF fp8" and "default basename -> the GGUF umt5", but `eng_wan_ti2v.py` currently hardcodes the default in `_loader_names` to `"umt5_xxl_fp8_e4m3fn_scaled.safetensors"`. Fix: Update the fallback value in `_loader_names()["clip"]` to `"umt5-xxl-encoder-Q5_K_M.gguf"`.
3. [Section 6] Defect: The plan requires `assert_usable` to "fail-closed unless the resolved VAE basename is the approved Wan2.2 name". The current code uses a blacklist (`if not vae_base or vae_base == _WAN21_VAE_BASENAME:`). Fix: Change the condition in `assert_usable` to a strict whitelist: `if vae_base != "wan2.2_vae.safetensors":`.

SHOULD-FIX:
1. [Section 1] Defect: The plan says to reconcile the clip inputs dict per loader. [ASSUMPTION] If `CLIPLoaderGGUF` uses a different input key (e.g., `clip_name` vs `gguf_name`) than the standard `CLIPLoader`, `_build_graph` must conditionally set the correct key in the `"inputs"` dict based on `self._clip_loader_mode()`.
2. [Section 4 & 5] Defect: If the config resolver is used in `assert_usable` to validate the sampler, it must gracefully handle `None` or empty strings from `os.environ.get()` by substituting the `"euler"` default before validation, otherwise it will falsely fail-closed on unset environments.

OPTIONAL / NICE-TO-HAVE:
- [Section 2] When adding `tile_size` to `VAEDecodeTiled` in `_build_graph`, expose it as an env var (e.g., `OTR_WAN_TI2V_TILE_SIZE=256`) rather than hardcoding it, as Mac MPS and 8GB Nvidia cards often have slightly different optimal tile sizes for VAEs.

CUT THESE (over-engineering):
1. [Section 5] Range-checking `cfg` and `shift` in `assert_usable`. It is safe to cut because ComfyUI's internal nodes (like `KSampler` and `ModelSamplingSD3`) already cast and clamp floats. Only validate the `sampler` and `scheduler` strings in `assert_usable` to prevent missing-node/key errors mid-render.