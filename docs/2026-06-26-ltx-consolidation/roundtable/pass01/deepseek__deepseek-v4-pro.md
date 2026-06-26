<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — core consolidation is sound, but the unified still and audio routing fails to handle the dual use of `ltx_audio_in` for character (face) beats versus scene beats, breaking invariants and producing wrong outputs.

MUST-FIX BEFORE BUILD:
1. **[Part B, still routing]** The proposed capability rule (`engine_consumes_still` AND `family != "audio_driven_face"`) unconditionally overrides the portrait with the scene still for `ltx_audio_in` on **every** beat, including character beats. This violates Invariant 5 and will feed a wide scene plate to a face-talking render.  
   **Fix:** After the initial portrait is set, gate the scene-still override on whether the beat is a **talking-head beat**. Determine that by role (`character_video`) or by the presence of `char_id` when the shot role is not announcer/music. For those beats, keep the portrait untouched; for all other beats, apply the unified scene-still rule. This preserves the portrait for HUmo / `ltx_audio_in`-on-character and the scene still for music/announcer/wide engines.

2. **[Part B, audio routing]** The unified engine’s family `audio_conditioned_video` causes `_uses_ambient_master_audio` to return `True` for all beats, including character beats. When per-line audio is missing, a character beat would fall back to an ambient master-mix slice (containing music/other voices) instead of degrading cleanly.  
   **Fix:** In the per-beat audio resolution (around the `_uses_ambient_master_audio` gate), add a condition: for `ltx_audio_in`, if the beat is a character face beat (role == `character_video` or char_id present and not announcer/music), **do not** synthesize a master-mix slice; instead leave `audio_ref` as-is (likely empty) to match the old `audio_driven_face` behavior. This keeps voice beats from receiving the wrong audio.

3. **[Part C, scene-prompt composition]** The plan updates the scene-prompt branch (line 1163) to include `ltx_audio_in`, which would apply brief-based scene composition to **character beats** as well, producing unsuitable scene descriptions for talking heads.  
   **Fix:** Exclude `ltx_audio_in` from the scene-prompt branch when the beat is a character face beat (same condition as above). For those cases, fall back to the existing character fallback prompt / gear scrub / M4 prompt path (as `audio_driven_face` does today). Only apply scene prompts for announcer/music roles.

4. **[Invariant 5 / Part B]** The plan asserts Invariant 5 is preserved, but the unified rule will leak a scene still into character beats if not fixed. Amend the implementation to explicitly guarantee: for any engine that consumes a still, if the beat is a character talking head, the portrait is used; otherwise, the scene still applies for wide engines.

5. **[Part B, OTR_ENABLE_LTX_I2V kill-switch]** The new capability-driven still routing does not incorporate the `OTR_ENABLE_LTX_I2V` env gate that previously governed still conditioning for `ltx_video`. If the operator wants to disable I2V for `ltx_audio_in`, they would have no control.  
   **Fix:** In the unified still rule, check `os.environ.get("OTR_ENABLE_LTX_I2V","1")=="1"` (or a dedicated env) when the engine is `ltx_audio_in`, mirroring the old `ltx_video` branch. Without this, the kill‑switch is incomplete.

SHOULD-FIX:
- **[Part B, capability check]** Use a function `engine_consumes_still` that verifies both `accepts_still` and `init_image in required_inputs` to avoid edge cases where an engine accepts a still for other purposes but should not get the scene still. This is a safer default.
- **[Part C]** Resolve the engine-id question (keep `ltx_audio_in` or rename to `ltx_av`) before solidifying file changes, to avoid two waves of churn.
- **[Part D]** Add tests specifically for `ltx_audio_in` on a character beat (portrait used, no ambient audio) and on a music beat (scene still used, correct audio slice).

OPTIONAL / NICE-TO-HAVE:
- Document the beat‑classifying logic (character vs. scene) in CLAUDE.md for future engine additions.
- Consider a new engine capability flag (e.g., `requires_portrait`) to further generalise still routing independent of name or family.

CUT THESE: none (the plan is already lean).

[ASSUMPTION] The plan assumes `ltx_audio_in` will be used for both scene and character beats; the routing fixes above must be implemented to make that true.  
[ASSUMPTION] The ambient-audio slice fallback for `ltx_audio_in` on music beats is desired (as with old `ltx_av_music`), and the fix only excludes character beats.  
[ASSUMPTION] The `visualizer` engine’s `accepts_still=False` is verified—ensure it is not inadvertently changed.