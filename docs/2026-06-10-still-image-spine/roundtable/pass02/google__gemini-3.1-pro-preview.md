<!-- requested_model: google/gemini-3.1-pro-preview | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan outlines the correct architecture but misses critical data plumbing (w/h to the engine, cache materialization, trace stamping) that will break the render or the acceptance checks.

MUST-FIX BEFORE BUILD:
1. [ST-3] `dispatch_images` does not pass `w` and `h` to the `request` dict, and `request_cache_key` does not hash them. If omitted, `gen_fn` will render scene stills at its default portrait resolution, breaking the landscape requirement. Fix: Update `request_cache_key` to include `kind/w/h`, and extract `w` and `h` from `pinfo` to include in the `request` dict.
2. [ST-3] Cache hits do not materialize into the current episode's directory or append a ledger row (the code just `continue`s). Fix: On cache hit in `dispatch_images`, look up the source path from `images_section["images"]`, copy the file to the current episode's `stills/` directory, and append a new row to the `images` list.
3. [ST-4] Trace rows do not capture `init_source` or `init_image`, which breaks the mechanical acceptance check. Fix: In `build_request_from_shot`, stamp `_init_source` and `_init_image` onto the request dict. In `run_episode`, add them to the tuple of keys copied to the trace row.
4. [ST-2] `derive_image_prompts` currently only loops over cast and announcer lines; it does not generate scene still objects. Fix: Update `derive_image_prompts` to call `derive_opening_music_beat` and scan lines for announcer/outro beats, generating their prompts via `compose_still_prompt` and appending them to the output dict with `kind`, `w`, and `h`.
5. [ST-1] `render_driver.py` hardcodes the open subject strings instead of calling the shared helper. Fix: Implement `get_open_subject(role, synthetic)` in `_otr_story_brief_helpers.py` and refactor `render_driver.py`'s `_is_open` block to call it.
6. [ST-3] `OTRImageGenDispatcher` lacks the `episode_id` input required to save stills to the episode-specific directory. Fix: Add `"episode_id": ("STRING", {"default": ""})` to `INPUT_TYPES`, pass it to `dispatch_images`, and construct `output_dir = f"output/otr/episodes/{episode_id}/stills/"`.

SHOULD-FIX:
7. [ST-3] `dispatch_images` hardcodes `role = "character_video"` and `other_beats_image_model`. Fix: Read `role = pinfo.get("role", "character_video")`, and use it to dynamically select the correct engine slot from `image_policy`.
8. [ST-4] `build_request_from_shot` does not select `init_image` by engine family. Fix: Implement `_still_index(ledger)` and update the `init_image` resolution to prefer scene stills for `image_to_video` and `static_motion` families, with a loud fallback to `_portrait_index` if absent.
9. [ST-2] `OTRMetaBriefImagePromptGen` lacks access to canvas dimensions to compute landscape `w` and `h`. Fix: Read `os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")` (matching `render_driver.py`) to derive the landscape dimensions without adding a new node input.

OPTIONAL / NICE-TO-HAVE:
- [ST-6] Wiring `image_done` to `OTR_ShotLock` perfectly sequences the video render node (since video render requires `ledger['video']` which `OTR_ShotLock` creates). No new gate inputs are needed.

[ASSUMPTION]
- [ST-5] verify: `still_kenburns` engine accepts `init_image` from `request.get("asset_refs")`.
- [ST-2] verify: "outro" beats can be reliably identified by role mapping over lines (if the outro is just the last announcer line, the logic needs to explicitly tag it as an outro beat).