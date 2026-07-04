VERDICT: no. The plan’s central resolution story is internally wrong: provider 1080p, request/canonical canvas, still canvas, and final composite canvas are separate surfaces, but the plan treats them as one cloud-only switch.

MUST-FIX BEFORE BUILD:
1. [Anchor findings] Defect: “Cloud video output resolution is a PROVIDER param, not the canvas” is false for delivered/canonical output. `nodes/_otr_video_engines/eng_cloud_video.py:163-172` passes request canvas into `canonicalize_video`, and `nodes/_otr_shared/cloud_media_canonical.py:246-270` scales/pads the provider clip to exact `w x h`. If Pixverse returns 1080p but request canvas is 1472x832 or 832x480, the canonical clip is conformed down. Concrete fix: split the plan into provider tier (`OTR_CLOUD_PIXVERSE_QUALITY=1080p`) plus explicit cloud canonical canvas (`1920x1080`) and define which code path supplies that canvas.

2. [The change under review] Defect: setting `OTR_VideoDirector.canvas_w/canvas_h` to 1920x1080 does not reliably make cloud `word_razzle` output 1080p. `word_razzle` is `family = "image_to_video"` in `nodes/_otr_video_engines/eng_cloud_video.py:347-352`; `render_driver` overwrites non-face/wide request canvas from `OTR_VIDEO_LANDSCAPE_CANVAS` at `nodes/_otr_video_engines/render_driver.py:1378-1384`, defaulting to 1472x832. Concrete fix: add or specify a cloud-video-only render/canonical canvas override instead of relying on the Director global canvas.

3. [The change under review] Defect: the proposed 1080p deliverable path is incomplete. Current workflow has `OTR_VideoDirector` node 87 widgets at 832x480 and video picks `viz_green`, not cloud; `OTR_SilentComposite` node 84 still composites at 1472x832; only `OTR_SceneAwareScopes` node 94 is 1920x1080 (`workflows/otr_scifi_16gb_full.json`, nodes 87/84/94). Concrete fix: state whether the target is provider generation only, canonical clip only, or final episode output; then wire the real workflow nodes for the chosen cloud engines and matching composite/output canvas in the same change.

4. [Anchor verdict] Defect: “SAFE for legacy local flows PROVIDED resolution stays PER-LANE” is not implemented by the proposed knobs. `OTR_CLOUD_PIXVERSE_QUALITY` is cloud-only, but any attempt to get cloud still/video canvas via `OTR_VIDEO_LANDSCAPE_CANVAS` would also change still/floor/viz render canvases because `render_driver.py:1378-1384` applies it to every non-face family before only `ltx_video` and `ltx_audio_in` get fixed overrides at `render_driver.py:1394-1428`. Concrete fix: introduce a cloud-specific canvas path or per-engine policy; do not reuse `OTR_VIDEO_LANDSCAPE_CANVAS` for cloud 1080p.

5. [Q2] Defect: “1080p cloud stills” has no concrete mechanism in the change. Scene still dimensions come from `_landscape_still_dims()` reading `OTR_VIDEO_LANDSCAPE_CANVAS` in `nodes/otr_meta_brief_image_prompt.py:339-349`, and cloud image canonicalization uses request `w/h` in `nodes/_otr_image_engines/eng_cloud_image.py:159-179`. Concrete fix: either cut 1080p cloud stills from this change, or add a cloud-image-only still canvas and prove it does not alter local still/viz render canvases.

6. [The change under review] Defect: “Kling’s own resolution tier” is underspecified and not grounded in current adapter inputs. `CloudKlingAvatarEngine` sends `mode` from `OTR_CLOUD_KLING_MODE`, not a resolution field, at `nodes/_otr_video_engines/eng_cloud_video.py:225-240`. Concrete fix: name the exact provider input/env value that selects 1080p, or mark Kling 1080p as verify-at-build until the partner node schema is pinned.

SHOULD-FIX:
1. [What the panel must ground] Add an explicit acceptance matrix: with cloud 1080p enabled, verify `ltx_video` stays 832x480, `ltx_audio_in` stays its recipe canvas, HuMo portrait remains portrait except wide variants, and still/viz local paths do not inherit 1920x1080. Cite/assert `render_canvas` receipts from the clip manifest, not just absence of OOM.

2. [Anchor verdict] The “new engine added without override” risk is real but too vague. Concrete fix: make the invariant executable: any newly registered heavy/local video engine must declare either a fixed native render canvas or an explicit `allow_global_landscape_canvas` capability before it can pass workflow validation. [ASSUMPTION] This may belong in registry tests rather than runtime code.

3. [Invariants] The plan mentions workflow JSON same-change, but not the exact workflow edits. Concrete fix: list node 87 model/widget edits and node 84/94 output-canvas edits, then run `OTR_WorkflowValidator` and link/widget audits against `workflows/otr_scifi_16gb_full.json`.

OPTIONAL / NICE-TO-HAVE:
- Add a short resolution glossary to the plan: provider tier, request canvas, canonical clip canvas, still canvas, composite canvas, final mux output. This would prevent future reviews from conflating them.

CUT THESE (scope / over-engineering):
1. Cut the global `OTR_VideoDirector` 1920x1080 bump as currently framed. It is either ineffective for cloud `image_to_video` because `render_driver` overwrites it, or too broad if later consumers read canonical canvas globally.

2. Cut 1080p cloud stills from the first build unless provider i2v quality demonstrably depends on 1080p init stills. It is safe to cut because Pixverse video quality is already a provider parameter, and the still-size path currently shares knobs with local landscape rendering.

3. Cut Kling 1080p from the same chunk until the exact partner schema is pinned. It is safe to split because Pixverse `word_razzle` already has a concrete `quality` field, while Kling currently exposes only `mode` in the adapter.