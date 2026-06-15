<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan requires changing a pure helper's signature and contains a fatal math flaw that will cause the boomerang to freeze-frame at the end of the beat window.

MUST-FIX BEFORE BUILD:
1. [THE FORK] `_ltx_frame_length` signature mismatch. Option (A) requires a canvas-aware floor, but `_ltx_frame_length` (L100) only accepts `(target_frame_count, fallback)`. You cannot make it canvas-aware without changing its signature and updating the callsite in `render_clip` (L427) to pass `width` and `height`.
2. [Proposed restore - Step 2] Target length math shortfall. If `target_frame_count` is 169, half is 85. `_ltx_frame_length(85)` snaps to 81 (8n+1). Mirrored length is `2(81) - 1 = 161`. 161 is less than the 169 target, meaning the composite will hold-last-frame (freeze) for the remaining 8 frames, ruining the loop. Fix: The half-ask must be `ceil((target + 1) / 2)` before snapping, and you must verify `2 * snapped - 1 >= target`.

SHOULD-FIX:
1. [Current render path] `OTR_LTX_RENDER_CANVAS` hallucination. The plan claims the engine renders at 832x480 via this env var. `eng_ltx_video.py`'s `_dims` (L314) does *not* read this variable; it reads `request.canvas` and falls back to `_LTX_DEFAULT_W=768` / `_LTX_DEFAULT_H=512`. If 832x480 is required, `_dims` must be updated to actually read the env var.

CUT THESE (over-engineering):
1. [THE FORK] Cut Option A (Canvas-aware floor). The code explicitly warns `do NOT touch the 169f floor` (L211) because it governs both paths. Option C (Safe full-render + use-first-half) requires zero changes to `_ltx_frame_length`, guarantees the decode survives, and avoids the math shortfall entirely by just slicing the proven 169-frame tensor.