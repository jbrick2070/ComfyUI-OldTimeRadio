VERDICT: yes-with-fixes. The plan is not build-ready as-is due to alpha-channel corruption in directory clips, hardcoded smoke-test constraints, and VramPeakProbe usage gaps.

MUST-FIX BEFORE BUILD:
1. [SCALER] Foreground alpha-channel corruption in `_encode_segment_from_dir`.
   Defect: Applying the shared `_scale_filter` consistently to the foreground character paths (`fg_filter` in `nodes/otr_silent_composite.py:444-445`) will corrupt transparency. The helper relies on `pad` with `color=black` (`nodes/otr_silent_composite.py:324-325`), which adds opaque black padding to RGBA frames, destroying the alpha channel and overlay capability.
   Concrete Fix: Add a boolean `pad` keyword argument (default `True`) to `_scale_filter`. Set `pad=False` when calling it for the character directory clip foreground: `fg_filter = ("[1:v]format=rgba,fps=%d,%s[fg]" % (fps, _scale_filter(w, h, sharpen=True, pad=False)))`.

2. [Pre-ship #5] Smoke test preflight checks reject the `sharp_lora` recipe.
   Defect: Pre-ship step #5 requires running the smoke test on the no-env default (`sharp_lora`). However, `scripts/run_otr_30word_smoke.py:197-204` contains hardcoded preflight assertions that fail if `OTR_LTX_AV_UNET` is not explicitly set to `DISTILLED_UNET` or if `OTR_LTX_AV_RECIPE` is unset, preventing the default recipe from running.
   Concrete Fix: Generalize `_preflight_distilled_native_graph` in `scripts/run_otr_30word_smoke.py` to support both `sharp_lora` and `distilled_native` recipes, verifying node configurations dynamically based on the active recipe rather than enforcing hardcoded constraints.

3. [VRAM verify] `VramPeakProbe` lacks context manager interface and is not wired.
   Defect: The plan specifies checking VRAM peak across `run_graph` via `motion_common.VramPeakProbe`. However, `VramPeakProbe` in `nodes/_otr_video_engines/motion_common.py:242-283` is not a context manager and must be started/stopped manually. The current `render_clip` implementation in `nodes/_otr_video_engines/eng_ltx_av.py:621-633` only checks VRAM instantaneously after graph execution, missing the peak.
   Concrete Fix: Construct, start, and stop `VramPeakProbe` around the graph run block in `nodes/_otr_video_engines/eng_ltx_av.py`:
   ```python
   probe = _MC.VramPeakProbe().start()
   try:
       with _ltx_av_vram_reserve():
           results = _wb.run_graph(...)
   finally:
       peak = probe.stop()
   ```
   Then call `_MC.assert_peak_within_ceiling(peak, ...)` before returning.

4. [DECODE] Ambiguous decode environment variables naming.
   Defect: The plan refers to `OTR_LTX_AV_DECODE_TEMPORAL_SIZE` and `_OVERLAP` to control tiling parameters. However, `VAEDecodeTiled` in `nodes/_otr_video_engines/eng_ltx_av.py:556-559` has both a spatial `overlap` (default 64) and a temporal `temporal_overlap` (default 8). An environment variable named `_OVERLAP` is ambiguous and risks miswiring spatial instead of temporal overlap.
   Concrete Fix: Name the environment variable explicitly `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` at module level, keeping the spatial `overlap` at `64` and wiring `temporal_overlap` to the new environment variable.

SHOULD-FIX:
1. [SCALER] Single-base video path in `normalize_to_silent_canonical` ignored.
   Defect: The plan ignores the single-base normalization path in `normalize_to_silent_canonical` at `nodes/otr_silent_composite.py:72-105`, which uses an ad-hoc local `vf` string (lines 85-89) instead of the new shared `_scale_filter` helper.
   Concrete Fix: Refactor `normalize_to_silent_canonical` to construct its video filter using the new `_scale_filter(w, h, sharpen=False)` helper to ensure scaler consistency.

2. [What r1 changed] Test suite environment override mock failures.
   Defect: If the decode and unsharp environment variables (`OTR_LTX_AV_DECODE_TEMPORAL_SIZE`, etc.) are evaluated once at module load time in `nodes/_otr_video_engines/eng_ltx_av.py`, unit tests in `tests/test_video_ltx_av.py` using `monkeypatch.setenv` will fail to verify override behavior because the module constants are already bound.
   Concrete Fix: Access environment variables dynamically at runtime inside the graph-building function `_build_graph` in `nodes/_otr_video_engines/eng_ltx_av.py:447-560`.

3. [Pre-ship #1] Lack of automated ffmpeg `unsharp` capability verification.
   Defect: Pre-ship step #1 adds "ffmpeg supports `unsharp` (preflight/test)", but there is no automated check in the codebase.
   Concrete Fix: Add a check in `nodes/otr_silent_composite.py` preflight that runs `ffmpeg -filters` and parses the output to verify the presence of `unsharp`, failing loud early if it is missing.

OPTIONAL / NICE-TO-HAVE:
- [What r1 changed / DECODE] Log resolved `temporal_size`, `temporal_overlap`, and unsharp parameters in `nodes/_otr_video_engines/eng_ltx_av.py:583-584` to aid in debugging VRAM spills.

CUT THESE (over-engineering):
None.

[ASSUMPTION] Assumed the "custom textured still plate background" under directory clips is part of the character lane and should be sharpened; if it is to be treated like a generated floor background, it must pass `sharpen=False` to `_scale_filter`.
