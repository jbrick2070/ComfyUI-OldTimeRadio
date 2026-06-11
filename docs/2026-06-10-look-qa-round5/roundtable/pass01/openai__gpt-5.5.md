<!-- requested_model: openai/gpt-5.5 | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan cannot pass the stated re-render gate as written because the current frozen misattribution is not repaired in the production render path, the visual_plan-to-beat mapping is undefined in shown code, and the LTX cap plan omits the compositor contract needed to preserve duration.

MUST-FIX BEFORE BUILD:
1. [R5-5 / Invariants / Acceptance] The attribution fix is scoped to “future episodes” and/or LOUD logging, but Acceptance requires this already-frozen episode to render with the correct face. Current `render_driver.build_request_from_shot()` gets `char_id` only from the frozen line (`char_id = str(line.get("char_id") or "")`) and ignores any shot-level override. Concrete fix: add a video-only attribution override in ShotLock for self-vocative cases, stamp it on the shot row, and change `build_request_from_shot()` to prefer `shot["char_id"]` / `shot["visual_char_id"]` over `line["char_id"]` when resolving portraits. Do not rewrite frozen audio or line rows.

2. [R5-2] “Map beats->scenes via the visual_plan’s scene index” is not implementable from the shown production shot rows. `otr_shot_lock.extract_beats()` does not extract any scene id/index, and `build_execution_plan()` does not stamp `scene_id`, `scene_index`, or `visual_prompt` onto shots. The legacy `visual_plan.scenes[]` only shows scene rows with `scene_id` / `visual_prompt`, not a beat mapping. Concrete fix: define the actual mapping source and stamp the resolved `scene_id` + `scene_visual_prompt` into each relevant shot at ShotLock; then `render_driver.build_request_from_shot()` can use the shot field directly. If no mapping exists, use a deterministic beat-order spread across scenes, not “scene 0,” because Acceptance requires b000/b001/b005 prompts to differ.

3. [R5-1] Capping inside `eng_ltx_video` alone will produce a shorter clip while the shot/manifest still carries the original `target_frame_count`. The shown `build_clip_manifest()` records both `frame_count` from the clip and `target_frame_count` from the shot, but the document does not specify or prove the compositor will hold-fill the short clip rather than stretch, loop, truncate, or create a timing gap. [ASSUMPTION] compositor behavior is outside the excerpts. Concrete fix: implement or verify the compositor rule explicitly: when `clip.frame_count < target_frame_count`, assemble exactly `target_frame_count` frames by appending hold-last-frame with the existing slow-zoom/Ken Burns treatment, no interpolation/stretch, and keep the audio mux byte-identical.

4. [R5-4] The M4 person-anchor change is underspecified against the current M4 prompt contract. `_build_batch_prompt()` currently asks the LLM to return only `{"beat_id","expression","motion","camera"}`; although `_parse_directives()` can read `text_prompt`, the prompt does not request it. If only the instruction text changes, the final `text_prompt` may still be composed from `appearance, setting, b["text"], expression, motion, camera`, allowing action/stage-direction dominance. Concrete fix: either update the requested JSON schema to include `text_prompt` with an explicit visible named-character subject, or make the deterministic HuMo prompt builder prepend a cast-anchored subject clause such as `<character name/appearance>, visible face-forward mid-shot, speaking`, then append motion/camera. Add a post-compose guard before `prompt_hash`.

5. [R5-6] “Normalize the announcer line char_id to the cast row id at ShotLock” risks mutating frozen ledger line content and still would not help unless the render driver reads the normalized value. Current `build_request_from_shot()` uses line `char_id`; shots do not carry a char id. Concrete fix: do not rewrite line rows. Stamp shot-level `char_id` for video only, change `build_request_from_shot()` to prefer it, and add the LOUD warning when the resolved video char id has no portrait path.

6. [Acceptance / R5-5] “No self-vocative/mis-attributed line ships” is not enforceable if R5-5 is allowed to “at minimum LOUD-log.” A warning-only mode can still produce the same wrong face and fail the eyeball. Concrete fix: for acceptance, self-vocative detection must either auto-repair a video-only char_id override or make the acceptance gate fail. Keep render fail-soft if required, but the re-render gate must not pass with only a warning.

SHOULD-FIX:
1. [D1] The forensic statement “passes `target_frame_count` STRAIGHT to the sampler” is not exact for the shown code. `eng_ltx_video.render_clip()` snaps length to LTX’s `8n+1` rule: `238` becomes `233`, not `238`. The real issue is “no max-frame cap before the 8n+1 snap.” Fix the wording and make the log/assertion report requested frames, capped frames, and snapped LTX length.

2. [R5-1] “default 121 (or 161)” is not build-ready. Pick one default. Given the cited proven-good range is 49-121f, use `121` unless there is new GPU evidence for `161`. Parse `OTR_LTX_MAX_FRAMES` defensively, clamp to LTX-compatible positive values, and log invalid env overrides LOUDLY.

3. [R5-1 / Acceptance] Acceptance says “no >cap LTX asks in the log,” but the engine snaps to `8n+1`. The check should assert the final graph `length` is `<= cap` and `length % 8 == 1`, or define whether the cap is applied before or after snapping.

4. [R5-2] Add robust `meta.visual_plan` shape handling. The legacy file contains defensive coercion for `characters` and `scenes`; the proposed new render-driver path should not assume `meta.visual_plan.scenes` is always a clean list of dicts. Reuse equivalent tolerant extraction or fail back to brief prompt with a LOUD warning.

5. [R5-2 / Acceptance] “fallback to scene 0 / nearest” can still make all three LTX prompts identical, directly violating “b000/b001/b005 not identical.” If scene mapping is absent, add a deterministic differentiator from role/beat/order or cycle through available scene prompts.

6. [R5-3] The bright-radio clause must survive the 240-char finisher. `finish_visual_prompt()` only preserves `no on-screen text` specially; it can trim other flavor text. Put the bright radio clause at the front, log the final prompt, and test that the final capped prompt still contains `radio`, `warmly`, and `dials` or equivalent.

7. [R5-4] A “person token” guard is too weak for HuMo. A prompt containing “person” could still not name the correct cast member. The guard should require either the cast display name or core appearance tokens plus visible-subject framing. Current `_prompt_is_consistent()` only checks appearance/setting tokens and does not guarantee visible face subject.

8. [Acceptance] “luma variance, not a flat field” is not a precise gate. Define the measurement window, threshold, and whether it runs on source LTX clip or final composite. Otherwise this can pass/fail nondeterministically depending on HUD/blend/compression.

9. [R5-5] The self-vocative detector needs a concrete rule. Current plan says `"<OwnName>, ..."` but b004 is described as text beginning with the addressee name while attributed to that same character. Define normalization for case, punctuation, surnames/first names, aliases, and quoted/stage-direction prefixes.

OPTIONAL / NICE-TO-HAVE:
- Add a per-shot prompt manifest containing final prompt, prompt source, scene_id, char_id used for portrait lookup, request seed, requested frames, capped frames, and final rendered frames.
- Add prompt-hash diversity checks for LTX shots so “code ran” gates catch prompt reuse before eyeball.
- Add a mid-beat still extraction artifact for each HuMo shot to make face/portrait misses reviewable without replaying the whole render.

CUT THESE (over-engineering):
1. [R5-1] Cut the alternative “split long synthetic beats at ShotLock into N chained shots” for this round. The stated goal is to stop LTX coherence collapse; a single capped render plus compositor hold-fill is cheaper, touches fewer seams, and preserves the frozen timing.

2. [R5-5] Cut the LLM repair pass as the first-line fix for self-vocatives. It adds model dependency and nondeterminism to a production-critical attribution seam. Use deterministic video-only override rules plus a LOUD acceptance failure when ambiguous.

3. [R5-4] Cut any image-level/person-detection guard for this build if contemplated by “mirror of the FLUX person guard.” A prompt-text guard plus portrait-path existence check is sufficient for the identified HuMo failure and stays CPU-cheap.