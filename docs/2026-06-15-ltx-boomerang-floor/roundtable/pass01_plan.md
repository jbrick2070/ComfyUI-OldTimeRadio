# LTX Boomerang Restore -- HARDENED BUILD SPEC (pass01, roundtable-converged)

3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4, $0.10), Claude-grounded vs the real
`eng_ltx_video.py` + `render_driver.py`. Converged in one pass.

## DECISION (the fork, resolved)
Restore `loop_via_reverse` as a **boomerang-only source-length + in-tensor frame mirror** inside
`LtxVideoEngine.render_clip`. **Do NOT touch the global `_ltx_frame_length` / 169 decode floor** -- it
guards the 1472x832 tiled-decode band and the code says "do NOT touch". Use a hardcoded proven-safe
boomerang minimum (**97** @ 832x480, GPU-proven this session). **REJECTED:** the canvas-aware global floor
(Option A), the runtime probe 73/49 (Option B-probe), and full-render-then-slice (Option C -- shallower
motion, defeats the "more motion" goal; Gemini favored it, rejected on that ground, its decode-safety
caution honored by the 97-floor).

## BUILD STEPS -- `eng_ltx_video.py`, `render_clip` (~L676)
1. **Env + class gate.** `_loop_via_reverse()` static reads `OTR_LTX_LOOP_VIA_REVERSE`: default ON; truthy
   {on,1,true,yes}; false {off,0,false,no}; invalid -> LOUD warn + default ON. Gate ALSO by a class attr
   (e.g. `_LOOP_VIA_REVERSE_DEFAULT`) so `LtxOrbitEngine` (subclasses LtxVideoEngine, inherits render_clip
   unchanged -- CONFIRMED L804) defaults OFF (orbit is opt-in; do not auto-boomerang it). Loop ON iff
   env-on AND engine-allows.
2. **Source-length helper** `_ltx_loop_source_length(target)`:
   - `src_ask = (target + 1) // 2`  (round UP -- fixes Gemini's freeze: target 169 -> naive half 85 ->
     snap 81 -> `2*81-1 = 161 < 169` -> composite hold-last-frame FREEZE for 8f, ruins the loop)
   - `src = ((src_ask - 1)//8)*8 + 1`  (8n+1 snap)
   - `src = max(src, _LTX_LOOP_MIN_DECODE_FRAMES)`  (env `OTR_LTX_LOOP_MIN_DECODE_FRAMES`, default **97** --
     the PROVEN-safe decode @ 832x480, NOT the global 169)
   - `while 2*src - 1 < target: src += 8`  (guarantee the mirrored clip COVERS the beat window -> no freeze)
   - cap at the existing `OTR_LTX_MAX_FRAMES`. Final mirrored length `2*src-1` is 8n+1 (`2*(8k+1)-1 = 16k+1`).
3. **Placement / ordering.** Compute `src` AFTER `width,height = self._dims(request)` + the /32 normalization,
   BEFORE `_build_graph` / `_build_graph_i2v`. When loop ON, pass `src` (not the full `_ltx_frame_length`
   result) as the graph length. Keep the i2v-vs-non-i2v class resolution UNCHANGED (do not reuse the
   non-i2v cached classes for the i2v path).
4. **Mirror in-tensor**, right after `frames = _wb.images_to_uint8(images)` (correct point -- before the
   silent-mp4 encode): `if loop and len(frames) >= 2: frames = np.concatenate([frames, frames[-2::-1]],
   axis=0)`. This drops the duplicate **turnaround (last) frame** -- NOT a "midpoint" (wording fix).
   Exactly equivalent to the 6/3 ffmpeg `[b]reverse,trim=start_frame=1;[a][r]concat`:
   `[0,1,2,3] -> [0,1,2,3,2,1,0]`. Guard `len < 2` (LOUD skip, no crash). Returned `frame_count =
   len(frames)` after mirror.
5. **i2v STAYS ON with the boomerang.** b005 (the target) was i2v+loop and looked right; the reverse half is
   the SAME decoded frames played backward (no re-render, no "unconditioned reversed first frame" -- that
   DeepSeek worry is a MISREAD). The loop join lands on frame 0 == the i2v bookend anchor = seamless. Eyeball
   the turnaround at build, but do NOT disable i2v.
6. **Ledger stamp** `ltx_loop_via_reverse: bool` -- add to the `render_clip` raw return dict and propagate via
   `canonicalize`/`_clip_from_raw` to the ledger (the sink EXISTS: b005's ledger has this exact field; verify
   the plumbing at build -- SHOULD, not a blocker).

## TESTS (CPU; no GPU)
- pure `_boomerang_frames([0,1,2,3]) == [0,1,2,3,2,1,0]` + a 9-frame case; assert `len % 8 == 1`.
- `_ltx_loop_source_length`: 193 -> 97 -> mirror 193; 169 -> src with `2*src-1 >= 169` (NO freeze);
  target < 193 floors src to 97; cap respected; every returned src is 8n+1.
- env parse on/off/1/0/true/false/yes/no/invalid; `ltx_orbit` defaults OFF, `ltx_video` ON.
- `len(frames) < 2` guard fires LOUD.
- fake `images_to_uint8`/encoder proving the encoder gets the MIRRORED array.
- Regression: full suite (4266) + Bug Bible + `test_audio_byte_identical` (video-only change keeps audio GREEN).

## INVARIANTS PRESERVED
global `_ltx_frame_length`/169 floor UNTOUCHED; single-resident <= 14.5 GB (half-render is CHEAPER, not more);
determinism (mirror is deterministic); audio byte-identical (video-only); UTF-8 no BOM; NO workflow-JSON
change (engine-internal); LOUD env/guards. Blur fix intact: 832x480 is set in `render_driver.py` L816
(`OTR_LTX_RENDER_CANVAS` default 832x480) and reaches the engine via `request.canvas` (read in `_dims`) --
the engine does NOT read the env itself (wording corrected vs pass00).

## GPU VERIFY (post-build)
Motion smoke at src=97 OR a real episode -> confirm a 193f 832x480 boomerang, mirror score ~0, motion ~b005
(framediff ~2.3, flow ~0.06), audio byte-identical, render-phase VRAM <= 14.5 GB.
