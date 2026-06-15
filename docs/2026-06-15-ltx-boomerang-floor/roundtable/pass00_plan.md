# LTX Boomerang Restore + Decode-Floor Handling -- Plan to Harden

## Goal
Restore the forward+reverse **boomerang** (`loop_via_reverse`) to the in-process LTX i2v engine
(`nodes/_otr_video_engines/eng_ltx_video.py`) so LTX clips regain the ~4x motion of the 5/30-6/5 era,
WITHOUT regressing the just-shipped blur fix or any invariant. Decide the **decode-floor handling** (the
one real fork) before coding.

## Why (grounded evidence)
- The boomerang was DELETED in the `70d379b` cleanbreak (it lived in the now-removed
  `nodes/batch_ltx_render.py`, BUG-LOCAL-117d). The current `eng_ltx_video.py` has NO loop code.
- The canonical good clip b005 (`signal_lost_chilled_hope_20260603_161926/videos/b005.mp4`, rendered at
  commit `59d9179`) has ledger `ltx_loop_via_reverse: true`, is 832x480 / 193f / 7.72s, and is a
  detectable forward-reverse loop (mirror score 0.65 ~= 0).
- Measured motion (both 832x480, same metric): b005 framediff 2.34 / optical-flow 0.061 vs the current
  single-pass smoke 0.72 / 0.014 (~3-4x). The boomerang is the difference.

## Current render path (grounded against eng_ltx_video.py)
- `render_clip` (~L676): `length = _ltx_frame_length(plan["target_frame_count"], self.target_fps)` ->
  build i2v/t2v graph -> `run_graph(free_after_use=True, keep={checkpoint, TERMINAL})` ->
  `images = results[TERMINAL][0]` -> `frames = _wb.images_to_uint8(images)` (uint8 [N,H,W,C]) ->
  `path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, fps)` -> returns `{out_path, frame_count:n}`.
- `_ltx_frame_length` (~L100): `length = max(_LTX_MIN_FRAMES=9, ask)`, cap `OTR_LTX_MAX_FRAMES`, RAISE
  short asks to `OTR_LTX_MIN_DECODE_FRAMES` (default `_LTX_DECODE_FLOOR_DEFAULT = 169`), snap to 8n+1.
- Floor rationale (code note L90-97): at the **1472x832** landscape canvas the installed wrapper's
  VAEDecode survives ONLY in its tiled band -- 169f/233f decode clean, 121f/137f raise a 256-vs-128
  tensor mismatch. Short asks raised to 169; the composite TRUNCATES long sources to the beat window.
- The blur fix (`4fc4268`) now renders LTX at native **832x480** (`OTR_LTX_RENDER_CANVAS`), i2v strength
  0.75. **GPU-proven this session: a 97-frame render at 832x480 decodes CLEAN (12s).** So the 169 floor
  is a 1472x832 artifact, not required at 832x480.

## Original boomerang to restore (faithful)
Render HALF the target duration, then mirror. 6/3 used ffmpeg post-process:
`[0:v]split[a][b];[b]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];[a][r]concat=n=2:v=1:a=0[out]`
-> output frames = 2*input - 1 (8n+1 preserved: 2*(8k+1)-1 = 16k+1). Frame 0 == last frame == the
radio-bookend still -> seamless loop. Env `OTR_LTX_LOOP_VIA_REVERSE` default on (truthy on/1/true/yes);
ledger field `ltx_loop_via_reverse`. Reproduced EXACTLY this session (97f render + that ffmpeg filter ->
193f @ 832x480, identical to b005).

## Proposed restore (in the current in-process engine)
1. `_loop_via_reverse()` static reads `OTR_LTX_LOOP_VIA_REVERSE` (default on).
2. When on: render ~half-length, then mirror **in-tensor** right after `images_to_uint8`:
   `frames = np.concatenate([frames, frames[-2::-1]], axis=0)` (forward + reverse minus duplicate
   midpoint). Avoids the ffmpeg re-encode entirely (lossless, simpler).
3. `frame_count` = mirrored length; stamp `ltx_loop_via_reverse` for the ledger/trace.

## THE FORK to decide -- decode-floor handling
The half-render needs a length < 169, but `_ltx_frame_length` floors to 169. Options:
- **(A) Canvas-aware floor** -- lower `OTR_LTX_MIN_DECODE_FRAMES` at native 832x480 (e.g. 9-49), keep
  169 at 1472x832. render_clip already has width/height. Faithful half-render; full motion depth.
- **(B) Boomerang-only floor bypass** -- when loop on, compute the half render-length against a separate
  proven-safe minimum (97 proven; probe 73/49 to go lower), independent of the global floor.
- **(C) Safe full-render + use-first-half** -- render the floor-safe full length (169), build the loop
  from the first ceil(N/2) frames + mirror. No floor change, lowest risk, but SHALLOWER motion (uses
  only the first half of the render -> never reaches peak drift the way b005 did).

Working recommendation: (A)+(B) hybrid -- canvas-aware floor at 832x480 with a proven-safe minimum for
the boomerang half-render, falling back to (C) only if a sub-97 decode probe fails.

## Invariants to GUARD (reject any critique whose fix breaks one)
- Single resident heavy engine <= 14.5 GB host NVML (half-render is CHEAPER -- a plus).
- Determinism seed-keyed (same seed -> same clip; the mirror is deterministic).
- Audio byte-identical (`test_audio_byte_identical`) -- the boomerang is VIDEO-ONLY, must not touch the
  master mux / audio path.
- UTF-8 no BOM; SFW; LOUD fail/fallback; NO new workflow-JSON widgets (engine-internal, no JSON change).
- The blur fix (native 832x480 render) must stay intact.

## Questions for the panel
1. Is the in-tensor mirror `frames[-2::-1]` the correct equivalent of the ffmpeg boomerang? Pitfalls:
   the FINAL clip's 8n+1 contract, odd/even frame counts, color/encode parity vs the silent-mp4 encoder,
   the duplicate-midpoint drop.
2. Which decode-floor option (A/B/C) is safest+best given the 169 floor is a 1472x832 artifact and
   97f@832x480 is proven? What is the real minimum-safe decode length to assume at 832x480?
3. Risks we under-weight: composite truncation of a slightly-long boomerang, the i2v loop seam (frame 0
   == frame N), audio-length vs the doubled video length, determinism, VRAM.
4. Anything here that breaks an invariant, or a simpler faithful approach we're missing.
