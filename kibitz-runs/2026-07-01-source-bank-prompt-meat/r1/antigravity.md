VERDICT: yes-with-fixes. Positional widget shifts in OTR_LedgerScriptWriter will break saved workflows, and public-domain prompts are dead code since their selectors are cut.

MUST-FIX BEFORE BUILD:
1. [input.md / final.md: Final Cut List] Public-Domain prompt coverage creep. The spec includes prompt changes for public-domain models (e.g., `faithful_radio_adaptation`) even though the final cut list explicitly removes the public-domain workflow selector and `source_text_path` from the transplant. Fix: Remove all public-domain models and adaptation prompts from the first transplant scope.
2. [nodes/OTR_LedgerScriptWriter.py:INPUT_TYPES] Positional widget shift. Inserting `source_bank` and `story_model` in the middle of existing optional/required widgets will shift existing graph indices. Fix: Strictly append the new inputs to the end of the `optional` dictionary in `INPUT_TYPES` to preserve backwards compatibility of existing workflows like `workflows/otr_scifi_16gb_full.json`.
3. [nodes/OTR_LedgerScriptWriter.py:_resolve_inputs] Media Archive RSS call leak. If `custom_premise` is left blank under the `media_archive` source bank, the resolver falls back to fetching science news RSS. Fix: Add an explicit check in `_resolve_inputs()` to raise a loud `RuntimeError` if `source_bank == "media_archive"` and `custom_premise` is empty, unless a local mock/fixture pathway is explicitly configured.
4. [nodes/OTR_LedgerScriptWriter.py:_resolve_inputs] Tonal style vs visual style collision. The visual styles (e.g., `noir`) overlap in name with outline story styles (e.g., `noir interrogation`), creating confusion in prompt building. [ASSUMPTION]: We infer that visual look and outlining genre are distinct concepts. Fix: Rename the visual style input widget to `visual_style_look` and isolate the namespaces in `meta` (i.e. `meta.style` vs `meta.visual_style`).

SHOULD-FIX:
1. [nodes/_otr_style_picker.py:pick_style] Threading system prompt overrides. Must-fix #2 in `final.md` requires overrides for `pick_style()`, but the current signature in `nodes/_otr_style_picker.py#L783-L792` doesn't accept `inventor_system_prompt` or `chooser_system_prompt`. Fix: Add the three optional override parameters as kwargs to `pick_style()` and update the callsite in `nodes/OTR_LedgerScriptWriter.py#L2827`.
2. [nodes/news_interpreter.py#L705] Non-news interpretation hardcoding. The system prompt in `_build_v1_prompt` hardcodes "news article" and "closing news read" instructions. Fix: Parameterize these labels in `build_news_briefs()` to read from the active `StoryPromptProfile`.
3. [nodes/_otr_story_brief_helpers.py#L229-L252] Automated testing for visual look scrubs. The plan lacks unit tests for validating that visual style changes actually scrub forbidden cinematic/radio phrases under styles like `anime` or `cartoon`. Fix: Implement `tests/test_visual_style_leakage.py` asserting that forbidden phrases are absent when non-default visual styles are used.

OPTIONAL / NICE-TO-HAVE:
- Annotate `story_model` choices in `INPUT_TYPES` with source-bank prefixes (e.g., "[Archive] Gentle Thriller") to prevent the operator from selecting mismatched model/source combinations.

CUT THESE (scope / over-engineering):
1. [input.md: Source/Story Models] Public-Domain prompt rules. Safe to cut because the adapter is excluded from the first build, avoiding dead weight and verification overhead.
2. [input.md: Visual Styles] Non-radio/non-documentary visual styles (e.g. `paper_origami`). Safe to stub or cut because deep visual prompt changes are deferred to V3.
