<!-- requested_model: openai/gpt-5.5 | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is directionally plausible but not build-ready because F1/F2/F3 leave prompt sites unhandled, conflict with existing helper contracts/log formats, and do not specify cache/hash/order semantics tightly enough.

MUST-FIX BEFORE BUILD:
1. [F1/G4/Acceptance] Log format and call frequency contradict the grounded helper. `log_story_brief_disposition` emits `[story_brief:<consumer_id>] ...`, but Acceptance expects `[story_brief] consumer=...`; its docstring also says each consumer calls it “exactly ONCE per run,” while F1 would call `finish_visual_prompt` from per-prompt sites. Concrete fix: choose one contract before coding. Either update Acceptance to grep `[story_brief:<consumer_id>]`, or change the helper format and tests. Also specify whether logging is per run, per shot, or per prompt; if F1 is per-prompt, update the helper docstring and expected log volume.

2. [F1] `finish_visual_prompt` fail-soft behavior is underspecified and currently conflicts with the invariant “tails degrade to defaults.” Grounded `get_story_brief_lighting()` returns `""` for absent/failed/empty brief; F1 only says it appends `get_story_brief_lighting`. Concrete fix: define constants in `_otr_story_brief_helpers.py`, e.g. `DEFAULT_ERA_TAIL = "timeless cinematic aesthetic"` and `DEFAULT_STYLE_TAIL = "cinematic, 35mm film look, subtle film grain, volumetric lighting"`, and make `finish_visual_prompt` use the era default when lighting is empty and the style default unless explicitly suppressed.

3. [F1/F2] One shared finisher needs consumer-specific length policy; otherwise LTX can exceed its documented prompt budget. Grounded `get_story_brief_ltx()` says LTX total motion budget is 220-240 chars with only 80-100 chars for the brief fragment. Appending full lighting/atmosphere plus the 35mm tail can blow that. Concrete fix: add a parameter such as `max_chars=None` / `consumer_kind="ltx"` to `finish_visual_prompt`, and for LTX trim the final prompt at a clause/word boundary after preserving required suffixes like “no on-screen text” if present. Do not globally cap portrait/M4 prompts at LTX length.

4. [F2] The proposed scene-open rewrite can drop current LTX-specific constraints. Grounded `render_driver.build_request_from_shot` currently adds “a vintage radio set glowing in the scene,” “slow cinematic camera drift,” and “no on-screen text.” F2 says the prompt becomes `get_story_brief_ltx(meta)` as the core plus F1 finishing, which does not preserve those motion/render constraints. Concrete fix: build scene-open as `brief_ltx_or_setting_core + LTX scene-open clauses + finish_visual_prompt(...)`, not just brief + tails.

5. [F2/G4] Env override semantics conflict with the audit requirement. F2 says `OTR_LTX_RADIO_PROMPT` is used verbatim and not F1-finished, but G4/Acceptance require story-brief disposition logging for scene-open consumers. Concrete fix: when `OTR_LTX_RADIO_PROMPT` is set, do not modify the prompt, but still call `log_story_brief_disposition(meta, consumer_id, log)` or a logging-only variant. Also change the existing warning text in `build_request_from_shot`; today it says “prompt composed from the episode brief” even when the env override path was used.

6. [F3] Hash/cache ordering is missing. Grounded `derive_creative_directives` computes `prompt_hash` from `text_prompt`, and `derive_image_prompts` computes `prompt_hash` after prompt acceptance. If finishing is applied after those hashes, cache keys and reports become stale. Concrete fix: in ShotLock and image prompt generation, run `finish_visual_prompt` before computing/storing `prompt_hash`; for ShotLock also ensure `creative["text_prompt"]` in `ledger['video'].shots[]` is the finished prompt.

7. [F1/F2/F3] Not all grounded visual prompt sites are covered. `_provide_lipsync_base` in `render_driver.py` overwrites the base render prompt with `_LSYNC_BASE_PROMPT` or `OTR_LSYNC_BASE_PROMPT`, a hardcoded “1940s radio actor...” prompt that bypasses the brief and tails. Concrete fix: for non-env override base clips, prefer the already-finished `request["text_prompt"]` when present, or pass meta through the request and finish `_LSYNC_BASE_PROMPT`; keep `OTR_LSYNC_BASE_PROMPT` verbatim if set.

8. [F2] The real episode path still has a generic fallback for any text/image video shot outside the announcer/music LTX branch. Grounded `build_request_from_shot` starts from `build_request`, whose default prompt is `"a 1940s radio studio, warm tungsten light, on air"`, then only replaces it for creative text or announcer/music `ltx_video`. [ASSUMPTION] If policy/env override routes `scene_broll`, `background_abstract`, or any no-creative role to `ltx_video`/`wan_i2v`, those shots remain generic. Concrete fix: add a generic no-creative visual fallback for text-driven/image-driven engines using brief core + F1, not only announcer/music LTX.

9. [G8/F5] The music-mood claim is internally contradictory with the grounding. The audit says `get_story_brief_music_mood()` has zero callers; the helper docstring says `nodes/musicgen_theme.py` imports it at C5g and that a test locks this. Concrete fix: verify actual `git grep get_story_brief_music_mood` and `nodes/musicgen_theme.py` before choosing “wire” vs “deprecate.” If grep is truly zero, update the stale helper docstring and any tests that claim it is wired.

10. [Question 1/Fix design completeness] `meta.visual_plan.scenes[].visual_prompt` and `meta.visual_plan.characters` are writer-stamped surfaces shown in the legacy adapter but not consumed by the new grounded path excerpts. Current `otr_shot_lock.py` reads cast rows and `story_brief_terms`; `otr_meta_brief_image_prompt.py` reads cast rows plus setting terms; `render_driver.py` reads setting terms and `meta.style`. Concrete fix: verify full-source grep for `visual_plan`; if no active new-path consumer exists, either explicitly declare `meta.visual_plan` retired or wire `visual_plan.scenes[].visual_prompt` into scene/broll prompt fallback.

SHOULD-FIX:
1. [F1] “Dedupe fragments already present” is too vague and risky. A naive substring dedupe can wrongly remove “volumetric lighting” because “lighting” appears elsewhere, or duplicate differently cased tails. Concrete fix: implement only case-insensitive exact-fragment dedupe after comma-splitting and whitespace normalization; add tests for existing `STYLE_ANCHOR` containing “film lighting” so it does not suppress “35mm film look” or “volumetric lighting” incorrectly.

2. [F1] “style-preset aware if cheap” conflicts with the legacy grounding comment that style preset is upstream-only and downstream visuals derive from the brief. Concrete fix: cut style-preset awareness for this sprint; use the fixed legacy default style tail unless an explicit operator prompt override is being used verbatim.

3. [F3] The M4 LLM instruction change must preserve the parser contract. Grounded `_build_batch_prompt` asks for JSON objects containing only `beat_id`, `expression`, `motion`, `camera`; `_parse_directives` tolerates `text_prompt` but does not require it. Concrete fix: add one sentence such as “Do not include film-stock or lighting tail terms; they will be appended later,” without changing the required JSON schema.

4. [F3/Invariants] Portrait guard ordering needs an explicit implementation point. The plan says guards run before finishing and finishing must not re-trigger guards; code must place finishing after consistency/person guard and before hash. Concrete fix: add a unit test where LLM output has person evidence, finishing appends tails, and no second person-guard fallback occurs.

5. [F1] Consumer IDs need to be enumerated. Grounded `log_story_brief_disposition` lists canonical IDs (`flux_env`, `flux_portrait`, `ltx`, `humo`, `musicgen`), but F1/F2/F3 refer to scene-open, M4, and portrait sites. Concrete fix: define exact IDs before coding, e.g. `ltx_scene_open`, `shotlock_m4`, `flux_portrait`, and update the helper doc/test expectations.

6. [F2] Existing `render_driver.build_request_from_shot` scene-open role detection parses `group_id` fallback. If F2 changes this block, preserve that fallback or older ledgers with only `grp_<role>` will silently skip scene-open composition.

7. [F1/F3] Import paths differ by module location. `render_driver.py` is under `nodes/video` and must import `.._otr_story_brief_helpers`; `otr_shot_lock.py` and `otr_meta_brief_image_prompt.py` are under `nodes` and must import `._otr_story_brief_helpers`. Concrete fix: specify this in the patch plan and add cold-import tests.

8. [Acceptance] “single 30w production render” is not enough to catch duplicate tails and hash-staleness. Concrete fix: add CPU tests for `finish_visual_prompt`, ShotLock creative hash-after-finish, image prompt hash-after-finish, LTX override logging-without-mutation, and no-creative LTX fallback.

OPTIONAL / NICE-TO-HAVE:
- Add a structured field to finished prompt reports, e.g. `tail_source={"era":"brief|default","style":"default|skipped_override"}`, but do not block the sprint on it.
- Add a one-line report count of prompts finished per consumer to make log volume reviewable.

CUT THESE (over-engineering):
1. [F1] Cut “style-preset aware if cheap.” It is not required to restore the lost legacy default, and the legacy grounding explicitly moved downstream era/style derivation away from style presets.
2. [F1] Cut complex semantic dedupe. Exact normalized fragment dedupe is enough; anything smarter risks deleting useful prompt terms.
3. [F5] Do not delete or deprecate `get_story_brief_music_mood` in this sprint. Grounding contains a direct contradiction about whether it is wired; verification and doc correction are sufficient before the render.