<!-- requested_model: google/gemini-3.1-pro-preview | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The image dispatcher hardcodes the character role and lacks episode-folder routing, breaking scene stills and episode-local storage.

MUST-FIX BEFORE BUILD:
1. [S1/Routing] `OTR_ImageGenDispatcher` hardcodes `role = "character_video"` in `dispatch_images`. If scene stills (e.g., `music_visual` or `scene_broll`) are dispatched, `_ireg.assert_usable(engine_id, role)` will fail closed for any image engine that doesn't support `character_video`. Fix: Read the role from the prompt payload: `role = pinfo.get("role", "character_video")`.
2. [S2] `OTRImageGenDispatcher` node does not accept an `episode_id` input, meaning `dispatch_images` falls back to its default `output_dir=None` (which writes to the global pool). Fix: Add `episode_id` to `INPUT_TYPES` and pass `output_dir=f"output/otr/episodes/{episode_id}/stills/"` to `dispatch_images`.
3. [S1] `OTR_MetaBriefImagePromptGen` only generates prompts for `cast` and `announcer_line_char_ids`. It completely ignores scene/open beats. Fix: Update `derive_image_prompts` to iterate over `ledger.get("video", {}).get("shots", [])`, identify open/scene beats, and generate prompts for them. Extract the LTX scene-prompt composition logic from `render_driver.py` into `_otr_story_brief_helpers.py` so both can call it.

SHOULD-FIX:
4. [S5] The era tail is still too heavy. `get_era_tail` takes the top 3 palette terms, and `get_story_brief_lighting` takes ALL lighting and atmosphere terms. Fix: Apply the requested diet in `_otr_story_brief_helpers.py` by slicing `[:2]` on palette, lighting, and atmosphere lists.
5. [S3] `eng_ltx_video.py` declares `required_inputs = ("text_prompt",)` but does not declare `init_image` as an optional input. If the installed wrapper exposes img2vid conditioning, the registry won't know LTX can consume the still. Fix: Add `optional_inputs = ("init_image",)` to `LtxVideoEngine`.

OPTIONAL / NICE-TO-HAVE:
- [S4] The M4->HuMo creative seam: HuMo is `audio_driven_face` and ignores `text_prompt` anyway, but if it falls back to `still_kenburns`, that floor might benefit from the M4 prompt if `init_image` is somehow missing. Passing the M4 prompt through to cast beats is a low-risk resilience win.

CUT THESE (over-engineering):
1. [S1] The `_is_synthetic_open` detection logic in `render_driver.py` relies on hardcoded string suffixes (`_OPENING_MUSIC_SUFFIX`). This is brittle and over-engineered. It is safe to cut and rely purely on the shot's `role` (`announcer_visual` or `music_visual`) to determine if it's an open.