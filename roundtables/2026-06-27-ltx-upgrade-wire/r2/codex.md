VERDICT: yes-with-fixes. Not build-ready as-is because scaler targeting, decode env names, and smoke/VRAM verification are still ambiguous against the actual interfaces.

MUST-FIX BEFORE BUILD:
1. [Open for r2/r3 / SCALER] The plan leaves `sharpen=True` segment selection open, but the current implementation has separate call sites that must be wired deliberately: `nodes/otr_silent_composite.py:579-587` calls `_encode_segment` for both real clips and floor, while `_encode_black` is separate. Concrete fix: change `_seg_vf` and `_encode_segment` to accept `sharpen=False`; pass `True` only for `kind == "clip"`, `False` for `kind == "floor"`, and leave `_encode_black` unchanged. For `_encode_segment_from_dir` at `nodes/otr_silent_composite.py:403-461`, decide explicitly: foreground directory frames sharpened, floor/black background not sharpened, still plate background verify/decide.

2. [What r1 changed / DECODE] `OTR_LTX_AV_DECODE_TEMPORAL_SIZE + _OVERLAP` is ambiguous because `VAEDecodeTiled` inputs currently include both spatial `overlap` and temporal `temporal_overlap` in `nodes/_otr_video_engines/eng_ltx_av.py:556-559`. Concrete fix: name the env var `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP`, keep spatial `overlap` at 64 unless intentionally changing it, and wire only `temporal_size=128`, `temporal_overlap=32`.

3. [Pre-ship #5 / RECIPE scope] The existing canonical smoke script is hard-coded to distilled_native and rejects the no-env/default sharp_lora path the plan says must be recorded. See `scripts/run_otr_30word_smoke.py:16`, `scripts/run_otr_30word_smoke.py:52`, and `_preflight_distilled_native_graph` at `scripts/run_otr_30word_smoke.py:188-204`. Concrete fix: either add a separate sharp_lora smoke or generalize this script to record `LtxAudioInEngine()._recipe()` and validate the graph for the actual recipe instead of requiring `DISTILLED_UNET`.

4. [Pre-ship #5 / VRAM verify] `VramPeakProbe` is not a context manager; its usable API is `start()`, `stop()`, and `assert_peak_within_ceiling()` in `nodes/_otr_video_engines/motion_common.py:242-295`. Current production `render_clip` still only calls `_MC.assert_vram_within_ceiling()` after `run_graph` and reclaim at `nodes/_otr_video_engines/eng_ltx_av.py:621-633`. Concrete fix: specify and implement one path: either wrap `_wb.run_graph(...)` with `probe = _MC.VramPeakProbe(...).start(); ...; peak = probe.stop(); _MC.assert_peak_within_ceiling(peak, ...)`, or make the smoke harness sample machine-wide VRAM for the whole prompt window and report that it is broader than the exact run_graph window.

5. [SCALER / API correctness] The shared `_scale_filter` contract is underspecified for `-vf` versus `-filter_complex`. `_seg_vf` returns an unlabeled `-vf` string at `nodes/otr_silent_composite.py:319-326`; directory clips need labeled `[0:v]... [bg]` and `[1:v]... [fg]` filters at `nodes/otr_silent_composite.py:428-445`. Concrete fix: define whether the helper returns only the inner `scale,pad,unsharp,fps` chain or accepts optional input/output labels. Add tests for both generated strings before editing call sites.

SHOULD-FIX:
1. [Pre-ship #1 / SCALER] Add an actual ffmpeg proof for `unsharp` in both paths, not just a filter-list check. The directory foreground path uses RGBA before overlay at `nodes/otr_silent_composite.py:444-457`; verify `unsharp` placement preserves alpha/source-over semantics with the existing opaque mesh test pattern in `tests/test_3d_image_streams.py:197-226`.

2. [What r1 changed / SCALER] The plan ignores the single-base normalization scale path in `normalize_to_silent_canonical` at `nodes/otr_silent_composite.py:84-88`, called by `OTRSilentComposite.composite` when there is no clip manifest at `nodes/otr_silent_composite.py:704-707`. [ASSUMPTION] If this path can ever normalize a real LTX clip, it needs the same helper and an explicit `sharpen` decision; if it is floor-only, document that and keep it unsharpened.

3. [Pre-ship #3] Env override tests need to account for import-time constants if implemented like existing module constants in `eng_ltx_av.py:52-90`. Concrete fix: either parse decode env inside `_build_graph()` so `monkeypatch.setenv` works without reload, or make the tests reload `nodes._otr_video_engines.eng_ltx_av` before asserting override behavior.

OPTIONAL / NICE-TO-HAVE:
- Add a small report line in `render_clip` with decode temporal settings next to the existing recipe log at `nodes/_otr_video_engines/eng_ltx_av.py:583-584`, so smoke logs prove recipe and decode knobs together.

CUT THESE (over-engineering):
1. None. The listed tests and smoke gates are proportionate to the GPU/VRAM risk; the issue is missing specificity, not excess scope.