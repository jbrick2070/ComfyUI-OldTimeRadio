VERDICT: no. The plan mixes the image-engine and video-engine seams, and the lyric text source is not wired to any current request path.

MUST-FIX BEFORE BUILD:
1. [Proposed approach / Still-to-clip lifecycle] `flux_still` is not the current engine pattern to copy. It is only a legacy alias to `still_pan`; `still_pan`/`still_flat` animate an already-minted still from `asset_refs`, they do not mint images. See `nodes/otr_video_director.py:106-130`, `nodes/_otr_video_engines/cheap_families.py:113-142`, `nodes/_otr_video_engines/cheap_families.py:202-241`. Concrete fix: choose one seam. Prefer an image adapter `cloud_ideogram_v4` in `nodes/_otr_image_engines` plus existing `still_pan`/`still_flat` for video duration. If instead `ideogram_lyric_still` is a video adapter, define a real `render_clip()` that invokes Ideogram, canonicalizes image, then emits a CanonicalClip.

2. [What it touches] Registering only a `CAPABILITIES` row will not make an image engine selectable. Image adapters self-register only when imported from `nodes/_otr_image_engines/__init__.py`; current imports are explicit guarded imports. See `nodes/_otr_image_engines/__init__.py:24-72` and `nodes/_otr_image_engines/registry.py:107-139`. Concrete fix: add the adapter module, guarded import in `__init__.py`, and matching `CAPABILITIES` row in the same change.

3. [Proposed approach] The lyric-card prompt source is not wired. Current image objects get scene prompts from `compose_still_prompt()` or `_compose_char_scene_prompt()`, not spoken-line excerpts; scene prompts explicitly append `NO_TEXT_CLAUSE`. See `nodes/otr_meta_brief_image_prompt.py:917-940`, `nodes/otr_meta_brief_image_prompt.py:1233-1309`, `nodes/_otr_story_brief_helpers.py:456-520`. Concrete fix: add a new `kind`/`card_mode` object path for lyric cards keyed from `lines[].text`, with the 8-word excerpt rule, and bypass the no-text scene composer.

4. [Music beats -- wordless TITLE-MOOD card] The plan does not name the actual data field for “procgen EPISODE TITLE.” Current music/open scene-still creation uses `compose_still_prompt()` from meta/role/beat, not title text. See `nodes/otr_meta_brief_image_prompt.py:1204-1305` and `nodes/_otr_story_brief_helpers.py:483-504`. Concrete fix: specify and test the exact ledger/meta key for the title; verify: exact procgen title field and fallback when absent.

5. [Still-to-clip lifecycle] `canonicalize_image` is still a stub and its signature does not match the current image-engine protocol. Image engines implement `render_image(request, prepared)` and return pixels/path to `OTR_ImageGenDispatcher`; they do not call canonicalizers. See `nodes/_otr_shared/cloud_media_canonical.py:106-109`, `nodes/_otr_image_engines/registry.py:79-83`, `nodes/otr_image_gen_dispatcher.py:715-735`. Concrete fix: after S1, state whether cloud image canonicalization lives inside the new image adapter before returning a path, or in dispatcher support for PartnerResult.

6. [ideogram_lyric_vid] `cloud_wan_i2v` will not receive the still through the current request shape. The cloud adapter reads top-level `request["init_image"]`, while `build_request()` emits `asset_refs.init_image`. See `nodes/_otr_video_engines/eng_cloud_video.py:197-203`, `nodes/_otr_video_engines/render_driver.py:225-263`, `nodes/_otr_video_engines/render_driver.py:1358-1360`. Concrete fix: make cloud video adapters read `asset_refs.init_image` or change the request builder/schema consistently before depending on `cloud_wan_i2v`.

7. [Still-to-clip lifecycle] “estimated_usd read from rendering_speed selection” is not implementable from the pin as-is. `partner_nodes.yaml` records `rendering_speed: COMBO` but excludes options and has only a notes string for pricing; `invoke_partner_node()` requires the adapter to pass a numeric estimate. See `nodes/_otr_shared/partner_nodes.yaml:126-155`, `nodes/_otr_shared/cloud_media_invoke.py:561-581`. Concrete fix: add a checked-in speed-to-price map or extend the pin to structured pricing; test TURBO/DEFAULT/QUALITY estimates.

8. [Caching] The cache claim is not true unless this adapter wires it. `RequestCacheKey`, `cache_lookup`, and `cache_store` exist, but production code under `nodes/` does not call them outside the cache module itself. See `nodes/_otr_shared/cloud_media_cache.py:66-119`, `nodes/_otr_shared/cloud_media_invoke.py:577-599`. Concrete fix: either wire lookup/store around the Ideogram call or remove “retries are cheap” from the build contract.

SHOULD-FIX:
1. [Proposed approach] The excerpt rule needs a concrete tokenizer/punctuation algorithm. “First clause boundary capped at 8 words” is underspecified for quotes, abbreviations, ellipses, em dashes, and empty first clauses. Concrete fix: add a pure helper with tests for punctuation, whitespace, stage directions, and all-empty input.

2. [Risks] “Record chosen palette/layout per card” has no named ledger field. Existing image rows record `provenance`, hashes, dimensions, and engine metadata, but not layout/palette. See `nodes/otr_image_gen_dispatcher.py:634-640`. Concrete fix: either add `provenance.lyric_card` fields or cut the repeat guard from v1 acceptance.

3. [Open questions] TURBO vs DEFAULT and v1.5 gating are implementation choices, not build-time open questions. Concrete fix: set v1 defaults in the plan, with env overrides if needed.

OPTIONAL / NICE-TO-HAVE:
OCR/legibility smoke is useful, but keep it as an operator eyeball/manual smoke unless a deterministic OCR dependency already exists. verify: whether the repo already has an OCR test dependency.

CUT THESE (over-engineering):
1. [Proposed approach] Cut the new video-role engine if the goal is still lyric cards. The current architecture already separates image minting from video duration via image engines plus `still_pan`/`still_flat`; adding a video adapter duplicates that seam.

2. [ideogram_lyric_vid] Cut v1.5 from the first build. Current `cloud_wan_i2v` has request-shape and dynamic-model issues; ship the still-card lane first, then animate only after the cloud video seam is proven.