<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan’s thesis is coherent, but the build design still relies on prompt obedience in the two places it explicitly says prompt obedience already failed: no-spoiler open and style-shaped body.

MUST-FIX BEFORE BUILD:

1. [2.3 / 5.1 / grounding `_ANNOUNCER_OUTRO_SYSTEM`, `compose_announcer_outro`] News coda goal contradicts the current outro machinery. The current system prompt says “Do NOT state a moral, lesson, or news-summary” and demands a “CONCRETE FINAL IMAGE,” while the redesign wants an explicit real-world teaching coda. Worse, the resolved branch injects `ending_change` and says “State this outcome plainly,” which is the fictional ending, not the news payload. Concrete fix: under `story_scaffold`, split the close contract into: final character beat lands fiction; announcer close is a labeled news coda sourced from `news_close_brief`. Remove/disable the resolved fictional-outcome instruction for the announcer coda, or pass `ending_change` only as “do not restate this as the coda.” Replace the old final-image/no-news-summary prompt and fallbacks with a coda-specific prompt/fallback.

2. [2.1 / 5.2 / grounding `compose_announcer_intro`, `fallback_announcer_intro`] The open is still not deterministically no-spoiler. The plan says “feed structured inputs” and add a no-spoiler prompt, but current fallback echoes `script_brief` verbatim, and the doc admits `script_brief` can contain the outcome. Concrete fix: create a `SafeOpenBrief` containing only era/time_of_day, place, cast names/roles, opening status quo, and contract tone. Do not pass `script_brief` to the open under `story_scaffold`. Build a deterministic fallback only from `SafeOpenBrief`. Add a post-gate using a forbidden phrase/token set derived from climax/outcome/twist/ending_change/news_close_brief fields; reject on overlap, reroll once, then use safe template fallback.

3. [1 / 5.3] KILL-2 repeats the single-prior failure unless “style” is made mechanically enforceable. Rendering `sound_world` / `story_engine` into every prompt is not a hard lever for weak local writers; it is another instruction. “Style register” is also not reliably gateable if it only means vibe/prose tone. Concrete fix: convert each `StoryContract` into deterministic, checkable obligations: required sound-world anchors, scene pressures, conflict mechanics, motif/object vocabulary, and ending-shape constraints. Gate outline/beat/line outputs for those anchors. If a contract has no gateable anchors, do not claim deterministic enforcement for it.

4. [1] “Selected pre-outline” and “from outline + StoryContract” are conflated. KILL-2 says build the contract before `OutlineRequest`; the open later requires time/place/cast/opening-situation “from the outline + contract.” That is fine only if these are two different stages, but the plan does not state the actual data handoff. Concrete fix: specify the pipeline: `StoryContract` before outline; outline emits safe open fields; announcer open consumes only those safe outline fields plus contract. verify: whether cast-lock truly exists before `OutlineRequest`; if not, move contract selection to the earliest stable seed point and defer cast-dependent fields.

5. [2.2 / 4 / 5.4] The plan contradicts itself on climax position. [2.2] says “The last voiced CHARACTER beat carries the dramatic climax,” while [4] says climax position should become spine-driven and not forced last. Concrete fix: stop using `final_character_line` as a proxy for climax. Add explicit `climax_beat_id` / `climax_character_line` / `ending_class` fields. The outro may receive final line for tone, but it must not assume final line equals resolution or climax.

6. [3 / 4] Cutting `consequence` because it is “unreachable under climax-last” conflicts with the stated KILL-3 direction. If climax can move earlier later, consequence/aftermath becomes reachable and important. Concrete fix: either keep consequence enrichment now, or explicitly mark it deferred until KILL-3 rewrites the role model. Do not delete it based on a constraint the design intends to remove.

7. [1 / 5.3] “Make conflict objects premise-specific” is a major subsystem hidden inside KILL-2 with no mechanism. It is not the same as wiring `StoryContract` fields into prompts. Concrete fix: either define a minimal deterministic mapping from contract + news domain to conflict objects, or cut this from the first build and leave the existing pool unchanged.

8. [6] Byte-identical-off is asserted but not designed through the new data path. New fields on `OutlineRequest`, `LineRequest`, meta, fallbacks, and prompt builders can change serialization/order or telemetry even when empty. Concrete fix: define the flag boundary explicitly: no `StoryContract` construction, no new prompt text, no `meta.story_contract`, no changed fallback text, and no request-shape change visible to old paths when `story_scaffold` is off. Add off-flag golden-output tests.

SHOULD-FIX:

1. [5.1] Use a fixed deterministic coda lead-in for the first build, not LLM-varied phrasing. Recommended: `The real story:` or `And now, the real story:` every episode under `story_scaffold`. Teachability matters more than elegance here, and variation can come later as a closed seed-keyed set after the coda behavior is proven.

2. [2.1 / 5.2] Define the cold-open structure exactly. Recommended template target: sentence 1 orients: era/time/place/cast/status quo. Sentence 2 creates intrigue without outcome terms. Example structure, not literal output: “Good evening. In [era], at [place], [cast/roles] begin with [status quo]. Tonight, [safe pressure] moves through the static.” This gives the gate something concrete to validate.

3. [1] The acceptance test “read N episodes” is subjective. Concrete fix: add objective checks: `render_style_grammar` or successor is called; `meta.story_contract` records slug and obligations; generated outline/line prompts contain contract fields under flag; each body beat satisfies at least one contract anchor; delete-it test reverts visible behavior.

4. [2.3] The coda length budget is unresolved. Current outro asks for one or two sentences, 14–34 words. A label plus real fact plus OTR signoff may not fit. Concrete fix: set a coda-specific budget and validator, e.g. one or two sentences, 18–45 words, label required, `news_close_brief` fact required, no fictional `ending_change` restatement.

5. [6] LOUD fallbacks need feature-specific telemetry names. Concrete fix: separate flags such as `announcer_intro_safe_fallback`, `announcer_intro_spoiler_reject`, `story_contract_style_gate_fail`, `news_coda_fallback`, not generic `announcer_intro_fallback`.

6. [1] Random style selection by seed is not inherently compatible with every news item. [ASSUMPTION] Some styles may fight the premise. Concrete fix: keep seed draw, but filter by minimal compatibility constraints, or allow the contract to adapt its anchors to the news domain before injection.

7. [3] The KILL-4 truncation fix needs a narrative invariant, not just “reserve the tail.” Concrete fix: define which enrichment fields are mandatory and preserve them first; truncate the old intent around them. Otherwise the body can still starve if the wrong material is preserved.

OPTIONAL / NICE-TO-HAVE:

- [5.1] After the fixed coda proves itself, allow a closed deterministic phrase set with the same semantic marker: “The real story:”, “What actually happened:”, “In the actual report:”. Do not let the LLM invent the label.
- [1] Add a per-contract “sample fulfilled line” to aid local models, but do not rely on it as enforcement.
- [2.1] Add a spoiler-gate debug artifact listing rejected terms so failures are inspectable.

CUT THESE (scope / over-engineering):

1. [4 / 7] Cut KILL-3 implementation from this build. Keep only interface-proofing if needed. It is explicitly deferred and will complicate `final_character_line`, outro assumptions, and role validation before the announcer/KILL-2 work is stable.

2. [1] Cut “make conflict objects premise-specific” from the first KILL-2 build unless reduced to a tiny deterministic map. It is a separate story-generation subsystem and not required to prove that `StoryContract` reaches and shapes the body.

3. [1] Cut full grammar injection into every `LineRequest` if it bloats prompts. Safer first build: pass compact contract obligations plus beat-specific required anchors. Full prose grammar can stay in macro/phase prompts or telemetry.

4. [2.3] Cut the old resolved-outcome announcer branch under `story_scaffold`. It actively works against the NEWS CODA goal by making the announcer restate the fictional ending.

5. [2.1] Cut all use of raw `script_brief` in the open path under `story_scaffold`, including fallback. It is explicitly contaminated by possible outcome text and cannot be made safe by prompt wording.