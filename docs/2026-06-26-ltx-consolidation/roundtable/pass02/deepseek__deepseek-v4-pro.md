<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — the core routing changes are implementable, but several underspecified / missing details will cause build failures or silent regressions if not clarified.

MUST-FIX BEFORE BUILD

1. **[B2 / AUDIO]** The plan states `_uses_ambient_master_audio` must return `False` for character-face beats and that its signature should be changed to accept the shot or a classifier result. However, the call site in `build_request_from_shot` passes `(engine_id, family)` only. The plan does not specify how to inject the character-face determination into that call.  
   - *Defect:* Ambiguous implementation guidance. A developer may update the function definition but not the call, breaking the audio routing silently.  
   - *Fix:* Provide the exact new signature (e.g., `def _uses_ambient_master_audio(shot, engine_id, family_id)` or accept a pre‑computed `is_character_face` flag) and show the updated call within `build_request_from_shot` (the block that slices the master audio). All callers of this function must be listed.

2. **[B1 / B4]** The `still_route` helper’s “optional-I2V set” is described as `{ltx_video}` without mentioning the `OTR_ENABLE_LTX_I2V` environment flag. The current code enforces the flag inside a dedicated `ltx_video` branch, which would be replaced by `still_route`.  
   - *Defect:* If `still_route` is implemented as written, `ltx_video` will always be treated as optional-still regardless of the flag, making `OTR_ENABLE_LTX_I2V=0` ineffective.  
   - *Fix:* Clarify that `still_route` must check `OTR_ENABLE_LTX_I2V` when the engine is `ltx_video`. The optional set `{ltx_video}` should only be active when the flag is enabled; otherwise, `ltx_video` is not considered a still-consuming engine.

3. **[Part C / render_driver.py]** The constant `_LTX_OPEN_ENGINES` currently includes `ltx_av_talk` and `ltx_av_music`. The plan says to add `ltx_audio_in` but does not explicitly state removal of the old names.  
   - *Defect:* If the old names remain, the BUG-LOCAL-413 health check will never fire for radio opens rendered with the new engine, because the check expects the engine to be in `_LTX_OPEN_ENGINES` and the old strings will be in the frozenset.  
   - *Fix:* Change `_LTX_OPEN_ENGINES` to `frozenset({"ltx_video", "ltx_audio_in"})` and remove both `ltx_av_talk` and `ltx_av_music`.

4. **[Part A / eng_ltx_av.py]** The plan instructs setting `default_roles=("music_visual","announcer_visual")` on `LtxAudioInEngine`. The grounding excerpt shows no `default_roles` attribute on that class (only `roles`).  
   - *Defect:* Without explicit assignment, the engine will not be the default for those roles, causing downstream dropdowns or default selections to fail.  
   - *Fix:* Add `default_roles = ("music_visual