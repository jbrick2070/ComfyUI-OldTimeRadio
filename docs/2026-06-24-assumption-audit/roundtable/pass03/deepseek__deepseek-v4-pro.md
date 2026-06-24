<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — multiple sequencing and integration gaps exist that would break the build or silently degrade key features.

MUST-FIX BEFORE BUILD:
1. [Build 2 / D.2] The plan says to move `select_style` to D.2 but does not clarify interaction with the existing two-pass style picker at the same location. If both run independently, the resulting `style_descriptor` (from picker) and `story_contract_slug` (from `select_style`) will be chosen by different logic and can diverge, undermining the later collapse into a single `StoryContract.slug`. Fix: Designate a single canonical selection step in D.2: after the two-pass picker resolves the style descriptor, feed that descriptor (or the article text) into `select_style` to deterministically pick the StoryContract slug. Store the slug and use it to build the contract.
2. [Build 1 / I] The body gate on exchange text (`_ex_text`) lacks a defined failure‑recovery path. The body gate runs after `compose_line` with a reroll; for exchange-composed text, no reroll exists and the plan says only “gate it too”. If the validator rejects an exchange line, the writer will either lose the line or crash. Fix: Implement a fallback: on gate failure for an exchange line, log a warning, keep the original text unmodified (no reroll), and stamp a flag. Ensure audio is never dropped.
3. [Build 4 / I.5] The announcer close fix requires `ending_tag` to be passed to `compose_announcer_outro`. The writer’s I.5 call site must have access to this value, but the plan does not wire it from the selected StoryContract. Currently only `_ending_template` and `_climax_beat_id` are computed in F2; `_ending_tag` is not stored or forwarded. Fix: After selecting the StoryContract slug in D.2, store it prominently (e.g., `_ending_tag`) and thread it to the I.5 `compose_announcer_outro` call as the new `ending_tag` argument.
4. [Build 2 / OutlineRequest] The plan adds `style_grammar`, `sound_world`, `story_engine`, `ending_tag` to `OutlineRequest`, but the grounding shows `_build_phase_user_prompt` and `_build_beat_user_prompt` do not yet render these fields. If the rendering code is not added alongside the dataclass changes, the outline LLM will never see the grammar, negating the injection. Fix: Implement the rendering of these fields in `_build_phase_user_prompt` and `_build_beat_user_prompt` before merging Build 2.

SHOULD-FIX:
1. [Build 2 / K5] Collapsing `style_descriptor` and `_style_slug` into `StoryContract.slug` changes the composer’s STYLE block from a short descriptor string to the verbose `render_style_grammar` output. This may affect prompt length and LLM behaviour. Should: Run a dedicated smoke test comparing line quality with the old descriptor vs. the new grammar block before finalising the collapse.
2. [Build 3 / F2] The role‑keyed enrichment map for non‑irreversible climax roles (revelation, reversal, confession, etc.) will call `fallback_content`, which currently provides only generic `personal_cost` strings. Using the same generic content for all roles may produce inappropriate enrichment (e.g., a “bittersweet_parting” beat getting a “cost them to be the one who decides” clause). Should: Extend `fallback_content` (or add a new map) with role‑specific fallback texts for each climax class before deploying Build 3.
3. [Build 1 / I] The reroll hint for the body gate is built solely from offending crisis tokens. This can produce repetitive hints (e.g., “avoid ‘reactor’”) without suggesting what to use instead. Should: Include the beat’s `conflict_object` in the hint to steer the model toward the grounded alternative.

OPTIONAL / NICE-TO-HAVE:
- [Build 4] The plan suggests retiring the C5 outline‑intent gate after the announcer close fix. Keeping it as a no‑op default is safe and can be done later.

CUT THESE:
- (None)

[ASSUMPTION] The `_otr_reroll.build_reroll_line_request` function exists and will be updated to handle new `LineRequest` fields; not confirmed from grounding.
[ASSUMPTION] The `select_style` deterministic function in `_otr_style_catalog` is callable with `script_brief` or `news_seed` as the `premise` argument; its signature accepts a string and a meta dict, but the exact shape of meta at D.2 may differ; assume it tolerates missing keys gracefully.