VERDICT: no. Phase 0 is not actually runnable as described, and Phase 1 would fail the real VideoRequest shape before reaching a provider.

MUST-FIX BEFORE BUILD:
1. [PHASE 1 -- build / Adapter reality] Cloud adapters read `request.init_image`, but real video requests put the path at `asset_refs["init_image"]`. `eng_cloud_video.py:197-202` uses `_req_get(request, "init_image")`; `render_driver.py:255-257` emits `asset_refs` plus `audio_ref`; `schemas.py:156-158` defines that schema. Concrete fix: make `_CloudVideoBase._init_image_input()` resolve `asset_refs.init_image` like `eng_humo.py:421-427`, and add an integration test that feeds `render_driver.build_request()` into `CloudKlingAvatarEngine._partner_inputs()`.

2. [PHASE 0 -- pin audit] The plan says “Audit the LIVE core’s partner-node catalog,” but the existing pinner cannot do that. `scripts/otr_pin_partner_nodes.py:68` defines manual `CURATED_ROWS`, and `scripts/otr_pin_partner_nodes.py:168` only iterates those rows. Concrete fix: add a non-mutating catalog survey mode that enumerates live `comfy_api_nodes` classes, captures `INPUT_TYPES`/`RETURN_TYPES`, applies the filter, and writes an explicit candidate report including “none found.” Only after that should rows be manually added to `CURATED_ROWS`.

3. [PHASE 1 -- Duration fit] “Duration fit is REQUIRED” is not implemented by the cloud video path. `eng_cloud_video.py:163-184` canonicalizes to canvas/fps and computes `frame_count` from provider duration; it does not trim/loop to `timing.target_frame_count`, which is the timing authority in `schemas.py:97-99` and `schemas.py:142`. Concrete fix: make cloud canonicalization accept `target_frame_count` and exact-fit the clip, or explicitly route through the existing clip-fill policy with tests proving the final manifest delivers the target frames.

4. [PHASE 1 -- Cost] The plan says pricing stamps drive selection and a per-episode cap is mandatory, but runtime video spend is a single global estimate. `eng_cloud_video.py:46-48` reads `OTR_CLOUD_VIDEO_EST_USD` for every row, while `cloud_media_backend.py:224-232` defines `CostQuote` but the adapters do not use it. Concrete fix: add `estimate_cost(request) -> CostQuote` or a row pricing table keyed by `node_key`, duration, and resolution, then reserve that estimate before invoke.

5. [PHASE 0 / PHASE 1A] The current pinned rows do not contain a non-V3 promptable no-audio i2v candidate. `partner_nodes.yaml:156-181` shows `cloud_kling_avatar` needs `sound_file`; `partner_nodes.yaml:282-305` shows `cloud_seedance_2` is `COMFY_DYNAMICCOMBO_V3`; `partner_nodes.yaml:397-424` shows `cloud_wan_i2v` has `first_frame` and V3 `model`, but no top-level prompt. Concrete fix: Phase 0 must explicitly output the surviving row list before Phase 1A. If empty, stop at BLOCKED.

SHOULD-FIX:
1. [PHASE 0 -- conformance test] “profile + conformance test” is underspecified. Existing `tests/test_cloud_video_adapters.py:91-120` tests a hand-built top-level request, not the real schema or a generic profile-to-pin guard. Concrete fix: define the prompt profile data shape and add a test that every emitted kwarg exists in `partner_nodes.yaml` required/optional inputs for that row.

2. [Grounded cloud-i2v reality] `cloud_kling_avatar` sends `"prompt": ""` even when no prompt is present (`eng_cloud_video.py:223-229`), although the pin marks it optional (`partner_nodes.yaml:172`). Concrete fix: omit optional `prompt` when empty unless the provider requires an explicit empty string.

3. [Problem / goal] The fail path says retry once with modified prompt/seed, but no retry owner or state is specified. Concrete fix: define whether this is manual spike procedure or code: if code, record attempt number, seed, prompt revision, provider row, and verdict in the run trace.

OPTIONAL / NICE-TO-HAVE:
- Add a frame-extraction helper for manual legibility review so “every frame readable” has repeatable artifacts, even before OCR exists.
- Record the resolved live catalog commit/hash beside the Phase 0 audit result.

CUT THESE (over-engineering):
1. [PHASE 1 -- Adapter reality] Cut “a prompt-conditioned, non-audio i2v row is a NEW reactivity class” unless the audit proves a different shape. `CloudWanI2VEngine` already models `required_inputs = ("init_image", "text_prompt")` with `reactivity = "mute_only"` in `eng_cloud_video.py:290`; a promptable i2v row can reuse that class pattern.

2. [PHASE 1 -- Prompt-shaped mint] Cut C-worded/C-plate contracts from this build plan. The section itself says it is “not a deliverable here”; keep it in the ideo_word / ideo_word_vid docs, not in the implementation checklist.