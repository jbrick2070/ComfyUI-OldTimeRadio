VERDICT: yes-with-fixes. Plausible sequencing, but requires resolving the registry configuration, prompt plumbing, and conformance test debt before build.

MUST-FIX BEFORE BUILD:
1. [`ideo_word` build contracts #1/#2 + Open questions]: Splitting into two registered engine NAMES (`ideo_lyric_text` and `ideo_title_mood`) under one node key `cloud_ideogram_v4` will pollute the dropdown selectors. Because `OTR_VideoDirector` dynamically builds the combo options from `_ireg.all_engine_names()` in `nodes/otr_video_director.py:169-178`, both engines will be selectable in all roles. The user could select `ideo_title_mood` (wordless old-radio centerpiece) for character beats or `ideo_lyric_text` (typographic text card) for the music slot, leading to misconfiguration.
   - **Fix**: Register a single engine name `"ideo_word"` in `nodes/_otr_image_engines/eng_cloud_image.py`. Internally, the adapter class for `"ideo_word"` should check `request["role"]` or `request["kind"]` at runtime and delegate to the appropriate implementation class (`CloudIdeogramLyricTextEngine` vs `CloudIdeogramTitleMoodEngine`).
2. [Prompt modes keyed by role / derive_image_prompts]: The prompt composer `derive_image_prompts()` has no access to the selected image engine list (`image_models` from `image_policy_json`), meaning it cannot know if `"ideo_word"` is selected for a role, and therefore cannot conditionally switch the target kind to `lyric_card`.
   - **Fix**: Extract `image_models` in `OTRMetaBriefImagePromptGen.generate()` in `nodes/otr_meta_brief_image_prompt.py:1384` and pass them to `derive_image_prompts()`. In the composer, check if the engine selected for a role is `"ideo_word"` to switch `tgt["kind"]` to `lyric_card` and trigger lyric card generation.
3. [Prompt modes keyed by role (title_mood)]: The ledger/meta key for the episode title is left as "verify-at-build".
   - **Fix**: Ground the key to `meta["episode_title"]` as stamped by the writer in `nodes/OTR_LedgerScriptWriter.py` and validated by tests like `tests/test_video_ledger.py:281`. If the key is missing or empty, fail loud by raising `ValueError` in `title_mood` mode.

SHOULD-FIX:
1. [Conformance test debt]: In `tests/test_cloud_partner_conformance.py:50-59`, `_engine_by_node_key()` returns `out[nk] = eng`, which overwrites duplicate node keys (e.g. `ideo` and `ideo_word` sharing `cloud_ideogram_v4`), meaning only one engine per node key gets its emitted kwargs verified.
   - **Fix**: Change `_engine_by_node_key()` to map `node_key -> list[engine]` and update `test_emitted_kwargs_are_declared` to assert conformance for all engines in the list.
2. [Workflow JSON]: The plan says "ImageDirector combo", but model dropdowns actually live in `OTR_VideoDirector` (see `nodes/otr_video_director.py:221-229`). `OTR_ImageDirector` has no image-model widgets and only reads the video policy (see `nodes/otr_image_director.py:241-292`).
   - **Fix**: Clarify that the verification target for dropdown selectable options is `OTR_VideoDirector`.

OPTIONAL / NICE-TO-HAVE:
- Include the selected mode (`lyric_text` vs `title_mood`) in the image prompts report payload for observability.

CUT THESE (scope / over-engineering):
1. [`ideo_word` build contracts #4]: Cut `cloud_media_cache` (`cache_lookup`/`cache_store`) wiring from the v1 adapter. Because lyric cards are per-beat unique and cannot be pooled across episodes, the global cache will have a near-zero hit rate. The dispatcher's local `cache_index` in `nodes/otr_image_gen_dispatcher.py` is sufficient to prevent regeneration during the same episode build run.

[ASSUMPTION] We assume that `comfyui` runs in an environment where the Ideogram provider credentials (`auth_token_comfy_org` / `api_key_comfy_org`) are injected and valid. If invalid, the node fails closed immediately without any user-facing hint in the UI.
[ASSUMPTION] We assume the widget values in saved workflows (like `workflows/otr_scifi_16gb_full.json`) will not be disrupted by registering a new engine. Because the selector is dynamic, registering `ideo_word` will append it to the dropdown options, but it won't be pre-selected in the saved workflow (defaulting to `flux_gen1`).
