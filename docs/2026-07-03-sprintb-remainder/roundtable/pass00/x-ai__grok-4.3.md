<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Unresolved video-policy wiring into the composer (generate inputs only carry image_policy_json), post-rip-sfx-broll pool mode, and missing render_driver registrations for the new engine will prevent a working build.

MUST-FIX BEFORE BUILD:
1. [Build steps 2 + Grounded facts] Composer (otr_meta_brief_image_prompt.py:generate) receives only script_json + image_policy_json; plan requires per-role video_models to branch on still_word. Add "video_policy_json" (forceInput) to INPUT_TYPES/params, parse it, and thread into derive_image_prompts/compose_still_prompt call; otherwise the branch is unreachable.
2. [What still_word IS + Open for the panel] Pool mode for char beats was deleted 2026-07-01 (rip-sfx-broll); plan still describes "pooled char -> mood SET". Scope as DEFERRED (named constant only, no objects), ship only word-driven + music-title paths.
3. [Build steps 1 + cheap_families.py:StillFlatFamily] still_word must register exactly like StillFlatFamily (name, family="static_image_gen", roles=..., uses_still=True, _still_motion=False, accepts_still=True) but plan omits updating render_driver.py:ENGINE_FAMILY, _PROFILES, _SCENE_INIT_FAMILIES and _still_index paths. Add the four sites or still_word never receives a minted still.
4. [Grounded facts + Build steps 2] Empty episode_title (music) and missing beat.text (word) must fail LOUD with explicit exceptions before any prompt_hash. Add the checks in the new pure helper; current _clean_llm_prompt path silently degrades to template.
5. [Build steps 2] "pure deterministic helper that turns the line into an image prompt" has no signature or tests specified. Define it (e.g., def word_to_prompt(text: str) -> str) with the five edge cases listed, or the composer change cannot be implemented.

SHOULD-FIX:
1. [Build steps 4] word_razzle stub is mentioned but Phase-gated; remove the name from v1 cheap_families.py to avoid accidental registration.
2. [render_driver.py:build_request_from_shot] still_word (static_image_gen) will hit the still_pan/still_flat/ltx_audio_in scene_still branch only if still_index already contains the minted asset; add explicit comment that ordering depends on image_policy_json["video_models"] having already run.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line unit test that still_word appears in the video combo after registration (cheap_families + registry).

CUT THESE (over-engineering):
1. "Model-agnostic proof: same still_word beat via >=2 image models" -- safe to cut; the image dropdown already decouples the choice and accepts_still mints independently.
2. Full "suite + Bug Bible + push per green chunk" ritual in the plan -- the spec already gates on green; the extra process text adds no code.

[ASSUMPTION] The still_word prompt branch will be a minimal-diff addition inside compose_still_prompt (imported from _otr_story_brief_helpers) rather than a new top-level function in otr_meta_brief_image_prompt.py.