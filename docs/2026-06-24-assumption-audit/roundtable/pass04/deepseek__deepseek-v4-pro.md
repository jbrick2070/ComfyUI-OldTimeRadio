<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — plan is nearly converged but contains two build-blocking ambiguities (object-match rule, prompt de-licensing) and one under-specified dependency (OutlineRequest prompt builders) that must be resolved before implementation.

MUST-FIX BEFORE BUILD:
1. [BUILD 1] `validate_composed_grounding` object-match rule is ambiguous: “HEAD NOUN / token overlap” could be interpreted as either head-noun-only or any-token-overlap, leading to incompatible gates. Specify the exact algorithm: e.g., “the line must contain at least one non-stopword token from `conflict_object` that is not in `GENERIC_CRISIS_NOUNS`, OR the last word (head noun) of `conflict_object` appears as a whole word in the line.” Provide a concrete definition.
2. [BUILD 1] Prompt de-licensing instructions are vague: “scope composer 1162 ("mission control ... fine") + 1442 ("Ground this line in the news facts") to compatible styles.” Define “compatible styles” precisely — e.g., “omit those lines when the style grammar is active (i.e., when `LineRequest.style_grammar` is non-empty), because the grammar block already provides style-specific grounding.” Without this, the prompt changes risk regressions or inconsistent behavior.
3. [BUILD 2] Outline injection: the plan says to add `style_grammar/sound_world/story_engine/ending_tag` to `OutlineRequest` and render `style_grammar` in `_build_macro_user_prompt`, `_build_phase_user_prompt`, and `_build_beat_user_prompt`. The grounding does not show these functions; their signatures and current use of `req.style` are unknown. Specify the exact changes to each function (or confirm they exist and can accept the new fields) to avoid a build-blocking dead end.

SHOULD-FIX:
1. [BUILD 2] K9/K10: “select_style ‘best-fit’ wording” and “DOMAIN_PALETTE scored matching” are mentioned without detail. Provide the concrete changes (e.g., rename a function, adjust the keyword map) so the implementor does not guess.
2. [BUILD 3] The role-keyed map for `_enrich_intent` is described only conceptually. Define the mapping explicitly (e.g., a dict from `beat_role` to tail text template) to prevent drift and ensure the specified tests (revelation/reversal/confession/quiet do not receive personal-stake wording) can be verified.

OPTIONAL / NICE-TO-HAVE: None.

CUT THESE: None.

VERIFY-AT-BUILD checklist:
- [BUILD 2] On first run after implementation, dump `resolved["style"]` and `meta["story_contract"]["slug"]` to confirm they are consistent (or intentionally divergent) per the plan.
- [BUILD 2] Verify that `_otr_pitch_room.run_pitch_room`’s `dataclasses.replace` preserves the new `OutlineRequest` fields (`style_grammar`, `sound_world`, `story_engine`, `ending_tag`) by inspecting the replaced request in a test.
- [BUILD 1] Confirm that `grounded_nouns` is passed to `_otr_reroll.build_reroll_line_request` (inspect the reroll path) so that rerolled lines do not lose the grounding palette.
- [BUILD 4] Test that for non-resolving ending tags (`unresolved_final_sound`, `revelation`, `quiet_acceptance`), the announcer close does not state an outcome and uses `fallback_announcer_outro` (not `_resolved_outro_fallback`).

[ASSUMPTION] The prompt-building functions `_build_macro_user_prompt`, `_build_phase_user_prompt`, `_build_beat_user_prompt` exist in `_otr_outline` and currently accept an `OutlineRequest`; the plan assumes they can be extended with the new fields without breaking other callers.