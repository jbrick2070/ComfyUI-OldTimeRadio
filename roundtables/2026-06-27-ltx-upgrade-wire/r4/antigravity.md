VERDICT: yes-with-fixes. The plan is highly mature, but a few critical wiring issues around VRAM peak propagation, filter signature updates, decode env handling, and alpha test validation must be resolved.

MUST-FIX BEFORE BUILD:
1. [VRAM verify] + [Smoke] Defect: The plan states the smoke test must record the NVML render-phase peak VRAM. However, `vram_peak_mb` recorded in the episode report (`nodes/otr_video_render_batch.py:222`) is derived from `render_driver.run_episode`, which gets `_mc.vram_used_mb()` (the post-render instantaneous VRAM) from `render_shot` (`nodes/_otr_video_engines/render_driver.py:1496`), rather than the actual `VramPeakProbe` peak. Furthermore, `_clip_from_raw` in `nodes/_otr_video_engines/eng_ltx_av.py` does not propagate `vram_peak_mb` to the canonical clip dict.
   Concrete Fix:
   - In `nodes/_otr_video_engines/eng_ltx_av.py:render_clip`, return `"vram_peak_mb": peak` inside the returned dict.
   - In `nodes/_otr_video_engines/eng_ltx_av.py:_clip_from_raw`, copy `"vram_peak_mb": int(raw.get("vram_peak_mb", 0) or 0)` into the returned dictionary.
   - In `nodes/_otr_video_engines/render_driver.py:render_shot`, return `clip.get("vram_peak_mb") or _mc.vram_used_mb()` as the final tuple element instead of unconditionally calling `_mc.vram_used_mb()`.

2. [Scaler helper] Defect: The helper signature `_scale_filter(w, h, fps, *, sharpen, pad=True)` requires passing `sharpen` down. However, `_seg_vf` (`nodes/otr_silent_composite.py:319`) and `_encode_segment` (`nodes/otr_silent_composite.py:336`) do not accept a `sharpen` argument. If they are not modified, `assemble_silent_timeline` cannot enforce `sharpen=True` for clips and `sharpen=False` for floor.
   Concrete Fix:
   - Update `_seg_vf` signature to `_seg_vf(w, h, fps, start_frame, sharpen=True)`.
   - Update `_encode_segment` signature to `_encode_segment(fb, src, n_frames, seg_path, *, w, h, fps, start_frame=0, loop=False, sharpen=True)`.
   - In `assemble_silent_timeline` (around lines 579-586), pass `sharpen=True` when calling `_encode_segment` for `kind == "clip"`, and `sharpen=False` for `kind == "floor"`.

3. [Decode env] Defect: The instruction "fail-loud/clamp if overlap>=size or <=0" is a build-blocking ambiguity. Clamping and failing loud produce fundamentally different runtime outcomes (silent correction vs termination).
   Concrete Fix: Resolve this ambiguity to strictly fail-loud. If `OTR_LTX_AV_DECODE_TEMPORAL_SIZE` and `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` are specified but invalid (non-integer, <= 0, or `overlap >= size`), raise a `ValueError` before creating the `VAEDecodeTiled` node. [ASSUMPTION] Clamping is cut.

4. [Tests] Defect: The Alpha test spec (lines 46-47) requires asserting that "edges stay (partially) transparent after format=rgba->scale->unsharp->overlay". However, `nodes/otr_silent_composite.py:455-457` flattens the output to opaque `yuv420p`. The resulting video file contains no alpha channel, making this assertion impossible.
   Concrete Fix: Change the test assertion to verify source-over blend math: assert that the pixel values of the semi-transparent edges in the flattened output show the expected blended contribution of the foreground color and background plate, rather than checking for output alpha.

5. [VRAM verify] Defect: In `render_clip`, if `_wb.run_graph` or `results[self._TERMINAL][0]` raises an exception, `results` and `images` will be undefined. Running cleanup / post-processing steps directly will trigger a `NameError`/`UnboundLocalError`, masking the true failure.
   Concrete Fix: Initialize `results = images = None` before the `try` block. In the `finally` block, stop the probe, and run `_retain_model_patchers` only if `results is not None`. Always execute `_wb.reclaim_idle_models` at the end to ensure VRAM recovery on failure.

SHOULD-FIX:
1. [VRAM verify] Defect: The plan uses `_MC.VramPeakProbe()` without arguments (defaulting to a `1.0` second sampling interval in `nodes/_otr_video_engines/motion_common.py:255`). For short/transient decode steps, a 1-second sample rate is too coarse.
   Concrete Fix: Instantiate as `_MC.VramPeakProbe(interval_s=0.1)` to match the fine-grained sampling pattern used in `nodes/_otr_video_engines/eng_wan_ti2v.py:455`.

2. [Smoke] Defect: The smoke preflight check documents and enforces distilled-native setup (`scripts/run_otr_30word_smoke.py:197-200`) specifically checking `OTR_LTX_AV_UNET == DISTILLED_UNET`. Generalizing it to other recipes will cause a crash if a different recipe/model is used.
   Concrete Fix: Remove the strict `DISTILLED_UNET` string equality assertion in `_preflight_distilled_native_graph` when generalization is active. Check the active recipe from `eng._recipe()` and assert corresponding structures (e.g. presence/absence of `lora`, `sigmas`, `modelsampling`).

OPTIONAL / NICE-TO-HAVE:
- Include the resolved recipe and decode temporal parameters in the log line in `nodes/_otr_video_engines/eng_ltx_av.py:583-584` to aid forensic log analysis.

CUT THESE:
1. [Decode env] Cut "clamp" behavior entirely. Fail-loud is cleaner and prevents operators from running with silent misconfigurations.

VERIFY-AT-BUILD checklist:
1. Confirm that `unsharp` filter capability is verified on the resolved `fb` binary from `_ffmpeg_bin` (not hardcoded `ffmpeg`) within the test suite (e.g., `tests/test_video_render_path_cw4.py`).
2. Verify that the output of `_scale_filter` follows the exact order: `scale -> unsharp (if sharpen) -> pad (if pad) -> fps`.
3. Verify that the spatial tiling parameters default to 512 size / 64 overlap, and temporal tiling parameters default to 128 size / 32 overlap.
4. Verify that temporal decode env overrides are read inside `_build_graph` at runtime, allowing monkeypatch testing without module reloads.
5. Verify that `EPISODES_DIR` contains the completed MP4 files under the correct slug hierarchy (e.g., `<output>/otr/episodes/<slug>/`) after smoke runs.
