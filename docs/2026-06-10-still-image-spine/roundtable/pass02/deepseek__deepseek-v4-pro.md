<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan contains multiple unverified prerequisites (still_kenburns init support, video render node gate input) and missing implementation details (episode_id input, cache key changes, seed scheme, etc.) that must be resolved before building.

MUST-FIX BEFORE BUILD:
1. [ST-5] still_kenburns engine must accept an external init image. Verify it does; if not, implement support before proceeding. Without this the 6/5 look cannot ship.
2. [ST-6] The video render node (OTR_VideoRenderBatch or equivalent) must have a gate input that can be wired to `image_done`. Verify it exists; if missing, add it. Otherwise the image-before-video ordering cannot be enforced.
3. [ST-3] OTRImageGenDispatcher node lacks an `episode_id` input. Add it to INPUT_TYPES and update the workflow JSON to wire it.
4. [ST-3] The dispatcher must save stills to `output/otr/episodes/<ep>/stills/` (episode-local) instead of the current content-addressed global pool. Implement episode-local saving in `dispatch_images`, including copying from the global pool on cache hit.
5. [ST-3] The cache key must gain `kind`, `w`, `h`. Modify `request_cache_key` to include these fields.
6. [ST-3] Seeds must follow the V-7 request-hash scheme (deterministic from request hash), not a fixed seed from image policy. Implement seed derivation in the dispatcher.
7. [ST-3] The dispatcher hardcodes `role="character_video"`. It must resolve the engine per role using the image policy slots (`announcer_image_model`, `music_image_model`, `other_beats_image_model`). Update `dispatch_images` accordingly.
8. [ST-2] The exact landscape dimensions for scene stills are ambiguous ("canvas-derived /32"). Define them concretely (e.g., 1472×832) and ensure the dispatcher receives them.
9. [ST-2][ST-3] Portrait stills must be generated at 832×1216. The image generation request must include `w` and `h` for both portraits and scene stills.
10. [ST-4] Implement `_still_index(ledger)` in `render_driver.py` returning `{beat_id: path}` for `kind=scene_*`.
11. [ST-4] In `build_request_from_shot`, for `image_to_video` and `static_motion` families, set `init_image` from `_still_index` with a LOUD fallback to today’s behavior when absent.
12. [ST-4] Trace rows must gain `init_source` (portrait|scene_still|none) and `init_image` basename. Update `run_episode` to include these fields.
13. [ST-1] Implement `compose_still_prompt` in `_otr_story_brief_helpers.py` with the specified layer order and era-tail profile.
14. [ST-1] Implement `get_open_subject` and refactor `render_driver` to use it for the open subject string.
15. [ST-2] `OTR_MetaBriefImagePromptGen` must derive and emit scene-still objects alongside portrait rows in `image_prompts_json`. Update the node’s `generate` method.
16. [ST-6] Apply the workflow JSON edits to `workflows/otr_scifi_16gb_full.json`: wire `episode_id` to the dispatcher and `image_done` to the video render node’s gate.

SHOULD-FIX:
1. [ST-3] Define the image policy schema for role-specific image models explicitly.
2. [ST-3] Add tests for cache-hit materialization (copying to episode stills).
3. [ST-2] Ensure the image engine respects requested dimensions or add post-processing to enforce 832×1216 / landscape.
4. [ST-2] Verify the no-text clause is appended correctly without exceeding prompt length limits.
5. [ST-4] Add logging for the missing scene-still fallback path.
6. [ST-1] Implement the parity test (driver LTX text prompt vs. open still prompt leading subject).

OPTIONAL / NICE-TO-HAVE:
- The global-pool retirement sweep could be done now instead of deferred, but the plan’s deferral is acceptable.

CUT THESE (over-engineering): none.

[ASSUMPTION] still_kenburns engine exists and can be modified to accept an init image.
[ASSUMPTION] The video render node is `OTR_VideoRenderBatch` (or similar) and its gate input can be added if missing.
[ASSUMPTION] The workflow JSON file is editable and the episode_id string is available as a constant or node output.
[ASSUMPTION] Image engines (Flux) can be configured to output specific dimensions.
[ASSUMPTION] The global pool directory (`output/otr/stills/`) remains accessible for cache-hit copying.