<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — the fix design is directionally correct but has multiple concrete defects (sequencing of hash computations, missing era-tail fallback, length budget for LTX, and over-engineered dedup/style-preset) that would cause broken prompts, stale hashes, or silent degradation at build time.

MUST-FIX BEFORE BUILD
1. **[F1/F3] Prompt‑finishing BEFORE hash computation.** In `otr_shot_lock._derive_creative_directives`, the finished `text_prompt` must be passed through `finish_visual_prompt` *before* the line that computes `prompt_hash = _content_hash(text_prompt)`. Same defect in `otr_meta_brief_image_prompt.derive_image_prompts`: finishing must happen after the person guard but before `prompt_hash = _content_hash(prompt)`. The plan’s wording “run through F1” is ambiguous about ordering; without this sequencing, the stored hashes will not match the actually‑rendered prompts, breaking determinism and cache‑key correctness.
   - **Fix:** Insert the finish call in both sites at the exact point described, and update the plan text to mandate this order.

2. **[F1/G2] Missing fallback for era tail.** `finish_visual_prompt` calls `get_story_brief_lighting(meta)`, which returns `""` when the brief is absent/failed. The plan says “tails degrade to defaults” but does not specify the default. The legacy `_DEFAULT_ERA_TAIL = "timeless cinematic aesthetic"` must be applied when the brief’s lighting is empty. Without it, the prompt will lack any visual‑aesthetic prose, violating G2’s stated need.
   - **Fix:** Define `ERA_TAIL_DEFAULT = "timeless cinematic aesthetic"` (or import from legacy) and use it as the fallback in `finish_visual_prompt`.

3. **[F2/LTX budget] No character‑limit enforcement for LTX finishing.** The LTX scene‑open prompt is already built from a 90‑char brief fragment; appending the full era tail + style tail can easily push the total beyond LTX’s 220‑240‑char limit. The design must ensure the finished prompt stays within budget, either by using a truncated era tail, a total‑length cap, or by only applying the style tail.
   - **Fix:** Add a `max_chars` parameter to `finish_visual_prompt` (default None, ignored for non‑LTX) and enforce it for the LTX site, or compose a shorter tail variant specifically for LTX.

SHOULD-FIX
1. **[F1]** Remove the “dedupes fragments already present” logic. The era/style tails are short; simple concatenation rarely causes harmful repetition, and dedup via substring matching is fragile, adds complexity, and risks removing legitimate repeated words. It is safe to cut.
2. **[F1]** Drop the “style‑preset aware if cheap” design element. No style‑preset system exists in the grounding excerpts; the legacy only had a single default. Adding a preset mechanism now is scope creep. Use a constant `STYLE_TAIL_DEFAULT` replicating the legacy string.
3. **[F3]** The plan says “the LLM instructions mention the era/style tail will be appended”. The actual modification to the ShotLock batch prompt and the portrait‑LLM instruction is not detailed. Verify that the new note is added to both instruction templates, and that it explicitly tells the model *not* to inject film‑grain/lighting itself.
4. **[G8]** In addition to the planned verification of `get_story_brief_music_mood`, also confirm that `nodes/_otr_music_prompt.py` (the live audio consumer) imports and uses that helper. If it uses an unrelated protocol, the helper should be marked deprecated but still kept; if it is indeed a dead duplicate, the deprecation comment should state that clearly.
5. **[GAP]** `meta.visual_plan` appears to be wholly orphaned after the `otr_video_plan.py` deletion. The plan never mentions it. Verify whether the writer still stamps it; if so, add a note (or a deprecation) to avoid wasted LLM work, or remove the writing step.

OPTIONAL / NICE-TO-HAVE
- Consider appending a small “radio‑set” visual anchor (e.g., “a vintage radio set in the scene”) to the LTX‑specific finishing tail for announcer/music beats to preserve the explicit studio cue that the current fallback provides, in case the brief prose alone does not reliably produce that element.

CUT THESE (over-engineering)
- The **dedup step** in `finish_visual_prompt`. Safe to cut because duplication of a few tokens in the prompt is harmless and the dedup logic introduces maintenance burden and subtle bugs.
- The **style‑preset‑aware tail selection**. The legacy had no style presets; implementing one now is premature. Use a single constant style tail.

[ASSUMPTION] The grounding does not show `render_driver` import for `_otr_story_brief_helpers`; the plan assumes it is importable at that call site (same process). Verify the import path and confirm no circular dependency arises (the helpers module is pure, so risk is low). [ASSUMPTION] The `finish_visual_prompt` call within `render_driver` will receive the logger `_LOG`; the plan’s sketch of `consumer_id` and `log` is compatible.