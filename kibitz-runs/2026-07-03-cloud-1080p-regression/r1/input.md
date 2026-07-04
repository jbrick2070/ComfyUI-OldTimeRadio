# 1080p cloud change -- regression-safety review (does it break legacy local non-1080p?)

## The change under review
Wire 1080p for the CLOUD lane only:
1. `OTR_CLOUD_PIXVERSE_QUALITY=1080p` (cloud Pixverse i2v / word_razzle output tier).
2. Kling's own resolution tier for the cloud talking-head.
3. (Proposed) set `OTR_VideoDirector` `canvas_w`/`canvas_h` = 1920x1080.
   HuMo portrait excepted.

## The two questions to converge on
- Q1: Does the 1080p cloud-video change alter or break any LEGACY LOCAL non-1080p
  flow (ltx_video, ltx_audio_in, HuMo 1.7B portrait, still_flat/still_pan/still_motion,
  viz_green/viz_mxc)?
- Q2: Does using 1080p cloud STILLS hurt a non-1080p LOCAL video model that
  consumes those stills as an init image?

## Anchor findings (Claude, grounded against the real code -- verify, do not trust blindly)
- **Cloud video output resolution is a PROVIDER param, not the canvas.**
  `nodes/_otr_video_engines/eng_cloud_video.py:393` -- CloudWordRazzleEngine sends
  `"quality": os.environ.get("OTR_CLOUD_PIXVERSE_QUALITY", "540p")` to the partner
  node. The partner call sends NO width/height; canvas is used only for
  `canonicalize` metadata (`:165-172`) and duration derivation (`:375-384`).
  => Setting `OTR_CLOUD_PIXVERSE_QUALITY=1080p` touches ONLY the cloud call.
- **Local engines OVERRIDE the request canvas per-family in the render driver --
  they DO NOT read the Director's `canvas_w`/`canvas_h` for their render size.**
  `nodes/_otr_video_engines/render_driver.py:1378-1428`:
  non-face families -> `OTR_VIDEO_LANDSCAPE_CANVAS` (default 1472x832);
  `ltx_video` -> `OTR_LTX_RENDER_CANVAS` (default 832x480);
  `ltx_audio_in` -> `OTR_LTX_AV_RENDER_CANVAS` (832x480/512x288);
  HuMo (audio_driven_face) keeps the 480x832 portrait from `build_request` default.
  This per-family override whitelist is the load-bearing invariant: it decouples
  local render resolution from the global Director canvas.
- **Cloud stills are canonicalized to the role canvas, then remapped into the
  engine's own canvas before a local engine renders.** `canonicalize_image`
  (cover+crop to exact role canvas) + `motion_common.py:_map_into_canvas` (pad/
  fit/crop) map any-size init into the local engine's native render canvas. Init
  image size does not drive local render VRAM; the engine's own render canvas does.

## Anchor verdict (pre-panel)
SAFE for legacy local flows PROVIDED resolution stays PER-LANE:
- cloud video 1080p via the provider `quality` param (isolated);
- cloud stills sized by the cloud-image role canvas (canonicalized/remapped
  before any local engine sees them);
- the single risk is the GLOBAL Director `canvas_w`/`canvas_h` bump. It does NOT
  reach local RENDER resolution (the per-family override whitelist protects
  ltx/humo/still/viz), but it DOES set the deliverable/composite canvas and any
  consumer NOT in that whitelist. So a global 1080p canvas bump is only safe
  because of the render_driver override whitelist -- that whitelist is the
  invariant to guard. A NEW engine added without a per-family override would
  inherit the 1080p canvas and could breach the <=14.5 GB VRAM ceiling.

## What the panel must ground (do not hand-wave)
1. Confirm/refute: does ANY local engine's render path read `policy["canvas"]`
   w/h for its own render size (bypassing the render_driver override)?
2. Confirm/refute: is the global Director canvas consumed anywhere that would
   push a LOCAL heavy engine (ltx/humo/ltx_av) to render at 1080p (VRAM breach)?
3. Confirm/refute: does a 1080p cloud still feeding a local i2v engine change the
   local render canvas or VRAM (vs the current 832x480 default)?
4. Is bumping the Director canvas to 1920x1080 necessary at all for the cloud
   lane, given cloud res comes from the provider `quality` param?

## Invariants (reject any fix that breaks one)
NO fallbacks / dropdown-only defaults; workflow-JSON same-change (BUG-LOCAL-097
positional widgets); audio spine byte-identical (mux LAST); single resident heavy
<= 14.5 GB; UTF-8 no BOM; SFW; suite + Bug Bible + B7 + push per green chunk;
prod/main GATED; HuMo portrait excepted from any landscape bump.
