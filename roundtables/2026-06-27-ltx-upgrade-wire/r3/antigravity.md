VERDICT: build-ready-with-fixes. The plan is highly grounded and matches the engine design, but has a missing signature parameter in the scaler and minor safety bugs in the VRAM try/finally scope.

MUST-FIX BEFORE BUILD:
1. [Scaler] Missing `fps` parameter in `_scale_filter` signature:
   - Defect: The plan specifies defining `_scale_filter(w, h, *, sharpen, pad=True)` in [otr_silent_composite.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_silent_composite.py) returning the inner `scale...fps` chain. However, because the chain appends the dynamic `fps` parameter (which varies between callers and manifest specifications), `_scale_filter` requires `fps` in its input arguments.
   - Fix: Change the signature to `_scale_filter(w, h, fps, *, sharpen, pad=True)`.
2. [Scaler] Foreground overlay aspect ratio/padding destruction:
   - Defect: In [otr_silent_composite.py:403-461](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_silent_composite.py#L403-L461), when composite-rendering straight-alpha frames, the foreground overlay must run with `pad=False`. If it is padded with solid black, the transparent borders around the character mesh are replaced with opaque black pixels, completely blocking the background plate.
   - Fix: Ensure `_scale_filter` supports omitting the `pad` filter in the chain (e.g., `pad_str = f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black" if pad else ""`).

SHOULD-FIX:
1. [Smoke] Recipe-specific graph validation logic in smoke preflight:
   - Defect: In [run_otr_30word_smoke.py:196-245](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/run_otr_30word_smoke.py#L196-L245), the preflight method `_preflight_distilled_native_graph()` is generalized to validate whichever recipe runs. However, if the default recipe is `sharp_lora`, the checks that forbid `lora` will fail.
   - Fix: Dynamicize the node validations inside the preflight based on `LtxAudioInEngine()._recipe()`:
     - For `sharp_lora`: Assert `lora` and `sigmas` are present; `modelsampling` and `sched` are absent; guider model input wires from `lora`.
     - For `distilled_native`: Assert `sigmas` is present; `lora`, `modelsampling`, and `sched` are absent; guider model input wires from `unet`.
     - For `m0_base`: Assert `modelsampling` and `sched` are present; `lora` and `sigmas` are absent; guider model input wires from `modelsampling`.
2. [Decode env] Robust environment variable parsing:
   - Defect: In [eng_ltx_av.py:559](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py#L559), casting environment variables `OTR_LTX_AV_DECODE_TEMPORAL_SIZE` and `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` directly via `int(os.environ.get(...))` will raise a `ValueError` and crash the ComfyUI process if these variables are set to malformed or empty values.
   - Fix: Wrap the cast in a `try/except ValueError` block, defaulting to `128` (size) and `32` (overlap).
3. [VRAM verify] Try/finally scope safety for extracting graph results:
   - Defect: In [eng_ltx_av.py:621-633](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py#L621-L633), if `_wb.run_graph` completes successfully but extracting the image batch `results[self._TERMINAL][0]` throws a `KeyError` or `IndexError`, the background `VramPeakProbe` thread will not be stopped, resulting in a thread leak.
   - Fix: Move `images = results[self._TERMINAL][0]` inside the `try` block before the `finally` block (similar to the pattern used in [eng_wan_ti2v.py:456-461](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py#L456-L461)).

OPTIONAL / NICE-TO-HAVE:
1. [Smoke] Parse and report NVML peak from log tails:
   - Recommendation: Have [run_otr_30word_smoke.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/run_otr_30word_smoke.py) parse the tailed server logs for `VRAM render-phase peak` to output the exact peak measured during the run.

CUT THESE:
- None. (No over-engineering detected).

ASSUMPTIONS:
1. [Scaler] [ASSUMPTION] We assume that still-plate background `bg_is_still=True` is the only "real bg" that should be sharpened (`sharpen=True`). When `bg_is_still=False` and `bg_path` is not empty, it is the base floor video (`base_video_path`), which is considered "procgen floor" and must have `sharpen=False` to match the core video segment rules.
2. [Tests] [ASSUMPTION] We assume that the ffmpeg `unsharp` capability check is run inside the test suite (e.g. [test_video_directory_clip.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_directory_clip.py)) rather than at ComfyUI startup/import time, avoiding startup degradation.
