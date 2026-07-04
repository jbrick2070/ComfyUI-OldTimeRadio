VERDICT: no. C1 and C2 conflate different capability contracts, and C5’s soak path can miss the new role-slot wiring.

MUST-FIX BEFORE BUILD:
1. [C1] `engine_consumes_still()` is the wrong predicate for render-driver scene-still binding. It returns true for `MotionEngineBase.accepts_still=True` (`nodes/_otr_video_engines/motion_common.py:439`) and therefore for HuMo (`nodes/_otr_video_engines/eng_humo.py:83`), whose `init_image` is a portrait/face input, not a scene still. If used at the current scene-still branch (`nodes/_otr_video_engines/render_driver.py:928`), it can wire b-roll stills into audio-driven face engines. Concrete fix: add/use a separate render-driver contract such as `scene_still_init=True`, or keep an explicit allowlist/family predicate for engines that should receive scene stills; explicitly exclude `audio_driven_face` and preserve the separate LTX-I2V branch at `render_driver.py:959`.

2. [C2] The fail-soft rule “if engine declares no `required_inputs`” will be wrong if it treats `required_inputs=()` as absent. `role_compat.engine_fits_role()` treats an explicit empty tuple as a valid capability that fits every known role (`nodes/_otr_shared/role_compat.py:107`), while `AbstractFamily.required_inputs=()` has legacy roles only `background_abstract` and `music_visual` (`nodes/_otr_video_engines/cheap_families.py:165`). Concrete fix: fall back to legacy `roles` only when the attribute is missing or `None`, or the role is unknown; do not fall back for `()`.

3. [C5] The existing soak profile builder still patches the old three-role model and says character is not director-mappable (`scripts/_otr_combo_soak.py:67`). Current wiring has dedicated role slots in `role_slots.py:41` and profile mappings for `character_visual`, `scene_broll_visual`, and `background_abstract_visual` in `config/profiles/widget_mapping.json`. Concrete fix: rebuild the soak profile generator to set all five keys: `announcer_visual`, `music_visual`, `character_visual`, `scene_broll_visual`, and `background_abstract_visual`; fill non-under-test roles with known-compatible baseline engines so fallback to `other_beats_video_model` cannot hide a broken slot.

4. [C3] Adding only `"scene": scene_broll` is ungrounded unless the ledger actually emits `speaker_role="scene"`. Unknown speaker roles currently fall back to `background_abstract` in `nodes/otr_shot_lock.py:55` and `nodes/otr_shot_lock.py:74`; `otr_meta_brief_image_prompt.py:290` imports the same map, so the same miss affects still-prompt generation. Concrete fix: verify the actual ledger token emitted for b-roll; map that token or change the writer/sequencer to emit `"scene"`. Add a test that the verified token produces `role=="scene_broll"` and resolves `scene_broll_video_model`.

SHOULD-FIX:
1. [C4/C2] The plan calls for a shared `descriptor_for_engine`, but descriptors are already duplicated locally in `nodes/otr_video_director.py:140` and likely need to stay consistent with registry/tests. Concrete fix: put the helper in the shared registry layer and make Director, registry tests, and the C4 matrix use the same function.

2. [C5] Specify the content oracle interface before building it: exact clip source, ffprobe/ffmpeg dependency behavior, luma threshold, temporal variance window, and exemptions for intentionally static engines. Without this, the soak can fail on environment/tooling variance rather than wiring defects. verify: exact oracle implementation file.

OPTIONAL / NICE-TO-HAVE:
- Add a workflow audit assertion that every mapped profile key in `config/profiles/widget_mapping.json` lands on an existing `OTR_VideoDirector` widget in `workflows/otr_scifi_16gb_full.json`.

CUT THESE (over-engineering):
1. [C4/C5] Do not duplicate live canonical-workflow coverage inside the CPU registry matrix. Keep C4 as a pure eligibility contract test; let C5 own canonical JSON/profile application/render-path coverage. This is safe because the two tests exercise different interfaces.