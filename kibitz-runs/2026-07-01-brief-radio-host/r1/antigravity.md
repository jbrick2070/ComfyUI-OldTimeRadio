VERDICT: yes-with-fixes. The core strategy is correct, but there are multiple structural integration gaps in the scene-still derivation, mesh fodder prompt pipeline, and voice-ref/init-image routing for lineless bookend beats.

MUST-FIX BEFORE BUILD:
1. [Design 3.3 / 3.4]: Missing init_image for music_visual beats on HuMo.
   - Defect: [ASSUMPTION] The plan routes both announcer and music bookends (roles `announcer_visual` and `music_visual`) to HuMo when the toggle is active. However, music beats are lineless synthetic bookends and do not have a dedicated cast row/`char_id` (unlike the announcer), so they generate no portrait in the image dispatch phase (verify: `nodes/otr_meta_brief_image_prompt.py:722`). If they are routed to a HuMo-family engine (which requires `init_image`), the dispatch will fail or fallback to a black/empty screen because `init_image` resolves to empty.
   - Fix: In `nodes/_otr_video_engines/render_driver.py` (`build_request_from_shot`), if a `music_visual` shot is routed to a HuMo-family engine, explicitly fallback to resolving `init_image` using the canonical `"announcer"` portrait still (which is the seed-pinned humanoid radio-host face).

2. [Design 3.2 / 3.4]: Hardcoded 1940s Open/Bookend Scene Stills.
   - Defect: The plan asserts that every radio surface will be brief-driven. However, regular scene stills for bookends (roles `announcer_visual` and `music_visual`) are generated via `compose_still_prompt` calling `get_open_subject`, which returns hardcoded 1940s strings (e.g. `nodes/_otr_story_brief_helpers.py:353-358` returns `"vintage radio set warming up on a wooden table..."`). This means the landscape scene stills used by `ltx_audio_in`, `still_pan`, and `still_flat` on bookends will remain hardcoded 1940s, violating the stated Goal.
   - Fix: Update `get_open_subject` (in `nodes/_otr_story_brief_helpers.py`) to accept `meta` and return a brief-driven, faceless radio description (e.g., `"tabletop radio receiver"` finished with the still-trimmed era tail) instead of the hardcoded 1940s strings.

3. [Design 3.2 / 3.4]: Unfinished/Non-Brief-Driven Mesh Fodder Prompts.
   - Defect: The plan states `mesh_stage` will use a "brief-driven 3D radio OBJECT", but `_compose_mesh_fodder_prompt` (`nodes/otr_meta_brief_image_prompt.py:917`) simply combines the raw subject returned by `_mesh_fodder_subject` with `MESH_FODDER_POS_SCAFFOLD`. It never calls `finish_visual_prompt` or `get_era_tail`. Thus, the mesh fodder radio prompt will consist of a generic static receiver and will not inherit any of the brief's era, style, or palette terms.
   - Fix: Update `_compose_mesh_fodder_prompt` to append the still-trimmed era tail (`get_era_tail(meta, profile="still")`) or style terms to the subject string before combining with the negative scaffold.

4. [Design 3.2]: Banned Gear Words Bypassed for Announcer in Prompt Request.
   - Defect: `_build_char_prompt_request` (`nodes/otr_meta_brief_image_prompt.py:454`) unconditionally instructs the LLM to `"Do not mention radios, microphones, studios, or any broadcasting equipment"`. For the synthetic announcer whose physical appearance is defined by `build_radio_host_prompt` (which contains radio elements), this instruction forces a contradiction. This ensures that any LLM-refined announcer prompt will strip the radio elements, fail the consistency gate (`_passes_consistency` at `:769`), and fallback to the template anyway.
   - Fix: In `derive_image_prompts` (`nodes/otr_meta_brief_image_prompt.py:745`), skip the LLM refinement call entirely for the synthetic announcer (`_is_announcer_row` or `_synthetic_announcer`) and directly stamp the template prompt. This avoids wasted LLM API calls and potential gate failures.

SHOULD-FIX:
1. [Design 3.4]: Announcer Mesh Fodder retains a face (suit/tie person).
   - Defect: [ASSUMPTION] `_mesh_fodder_subject` (`nodes/otr_meta_brief_image_prompt.py:583`) returns `"a vintage 1940s radio announcer in a tailored suit and tie"` for `announcer_visual`. This describes a human person (with a face), contradicting the rule that "only humo should have a face" and "mesh_stage -> brief-driven 3D radio OBJECT, NO face".
   - Fix: Update the `announcer_visual` branch in `_mesh_fodder_subject` to return a faceless tabletop radio/console form similar to `music_visual`.
2. [Design 3.3]: Wide/Landscape HuMo Aspect Mismatch.
   - Defect: [ASSUMPTION] If the active video engine is a wide variant of HuMo (e.g. `humo_1.7B_169` with `render_aspect="wide"`), and the toggle is active, it will animate the announcer portrait. However, the announcer portrait still is generated as a `portrait` aspect (480x832/832x1216) by default. Passing a vertical portrait as an `init_image` to a wide engine will cause pillarboxing or mismatching.
   - Fix: In `derive_image_prompts`, ensure the synthetic announcer's aspect follows the aspect of the announcer video engine slot (similar to `still_aspects.get("announcer_visual")` logic in `derive_image_prompts:740`).

OPTIONAL / NICE-TO-HAVE:
1. [Design 3.3]: Explicit environment variable/feature flag: [ASSUMPTION] The operator toggle to enable/disable HuMo bookends should be implemented via a uniform env variable (e.g. `OTR_ENABLE_HUMO_HOSTS = 0/1`) to align with existing levers in `nodes/_otr_config.py`.

CUT THESE (scope / over-engineering):
1. [Design 3.2]: (optional) the ltx_audio_in radio-console motion prompt -> same brief-driven form.
   - Why safe to cut: The plan itself states in Design 3.4 that `ltx_audio_in` should be deferred for now ("maybe ltx audio in but not yet") to keep the animated console as-is. Standardizing the motion prompt text at this stage is unnecessary scope creep and risks breaking the calibrated baseline for `ltx_audio_in`'s motion patterns.
