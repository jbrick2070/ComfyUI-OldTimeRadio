VERDICT: build-ready as-is? yes-with-fixes.
One line why: Capability validation will fail-closed due to omitted dictionary fields, and cairo's 4-channel BGRA output will corrupt the 3-channel RGB ffmpeg encoder pipeline without channel conversion.

MUST-FIX BEFORE BUILD:
1. [registry.CAPABILITIES["viz_mxc_mandala"]] Omitted capability schema fields. The plan lists only (cpu_ok, required_toolchain, model_requirements) for the capabilities row. However, `validate_declaration` in `nodes/_otr_shared/capability_profiles.py:L260-273` enforces that all declarations contain exactly all keys in `_DECL_KEYS`. Omiting `vram_class`, `vram_estimate_mb`, or `requires_sidecar` will raise a fatal `ProfileError`. Fix: Use the complete dictionary structure:
   ```python
   "viz_mxc_mandala": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None, "requires_sidecar": False, "cpu_ok": True, "model_requirements": []}
   ```
2. [Tests / eng_viz_mandala surface->numpy] Interface contract mismatch on color channels for ffmpeg input. The plan specifies wrapping cairo's buffer as a 4-channel array: `np.ndarray((h,w,4),uint8,buffer=surface.get_data(),strides=(stride,4,1))`. However, the downstream ffmpeg encoder helper `encode_silent_mp4` in `nodes/_otr_shared/scope_draw.py:L399-425` is hardcoded to expect 3-channel `rgb24` frames. Direct injection of a 4-channel BGRA buffer will desynchronize/corrupt the ffmpeg standard input. Fix: Drop the alpha channel and reverse BGR to RGB:
   ```python
   bgra = np.ndarray((h, w, 4), dtype=np.uint8, buffer=surface.get_data(), strides=(stride, 4, 1))
   rgb = bgra[..., :3][..., ::-1]
   ```
3. [GROUNDING #8 / Open List] Missing step for new CRT post-processing helper. Decision #8 states that CRT post-processing will use a public helper `apply_crt_post_rgb(rgb, ..., rng_key)` in `nodes/_otr_shared/scope_draw.py`. However, the Round 3 wiring list completely omits this target file. Without adding this helper, the mandala engine will raise an `AttributeError` at runtime. Fix: Add a step in the plan to implement and export `apply_crt_post_rgb` in `nodes/_otr_shared/scope_draw.py`.

SHOULD-FIX:
1. [scripts/otr_video_soak.py:~56] Inconsistent `ENGINE_FAMILY` mapping. The plan adds `"viz_mxc_mandala": "abstract"` to `otr_video_soak.py:ENGINE_FAMILY` but omits the preexisting `"viz_mxc_cpu"` engine. While neither is in `_PROFILES`, this creates a configuration drift with `render_driver.py`. Fix: Add both `"viz_mxc_cpu": "abstract"` and `"viz_mxc_mandala": "abstract"` to `scripts/otr_video_soak.py`.
2. [Tests / eng_viz_mandala surface->numpy] Missing `np` namespace prefix for `uint8`. The plan writes the type as bare `uint8` [ASSUMPTION]. This will raise a `NameError` if `uint8` is not explicitly imported from numpy, which is non-standard. Fix: Reference `np.uint8` or use keyword `dtype=np.uint8`.

OPTIONAL / NICE-TO-HAVE:
1. Include a test in `test_video_viz_mandala.py` verifying that missing `cairo` returns the correct LOUD error message mapping to `EngineUsabilityReason.MISSING_MODEL` in `assert_usable`.

CUT THESE (over-engineering):
None. The wiring plan is extremely lean and focused.
