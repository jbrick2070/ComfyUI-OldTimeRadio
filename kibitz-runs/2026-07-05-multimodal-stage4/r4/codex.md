VERDICT: yes-with-fixes. The plan is close, but `cliche_replacements` can still become loaded-but-not-live unless repair threading and replacement validation are made explicit.

MUST-FIX BEFORE BUILD:
1. [§3, §5] `cliche_replacements` consumer is under-specified. Current repair uses `_CLICHE_REPLACEMENTS` directly in `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_line_hygiene.py:721` and iterates the constant at `:732`; `compose_line` calls `repair_cliche_span` at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_line_composer.py:2562` and `:2605`. Fix: state that `StoryRules` carries compiled `cliche_replacements`, `repair_cliche_span(text, *, rules=None, replacements=None)` uses `rules.cliche_replacements`, and every `compose_line` repair/find path passes the same `_story_rules`.
2. [§2, §5] Replacement-template validation is missing. Regex compile does not validate replacement backrefs, and current `repair_cliche_span` catches failures and returns the original text at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_line_hygiene.py:744`. Fix: loader must fail loud on invalid replacement templates/backrefs, not just invalid patterns; add this to the fail-loud matrix.

SHOULD-FIX:
1. [R3 fixes, §3] `compose_line_draft` is required to call `resolve_creative_system_prompt` even when `creative_repo_id is None`, but the router signature is still `repo_id: str` in `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_creative_prompt_router.py:90`. Fix: widen/document `repo_id: str | None` or specify the exact sentinel string passed for `None`.
2. [§5] Add explicit acceptance for cliche repair coverage after extraction. Current coverage expectation exists around `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_story_quality_cliche.py:63`; preserve an equivalent test against `science_news.json`.

OPTIONAL / NICE-TO-HAVE:
None.

CUT THESE:
None. The remaining detail is mostly build-contract material, not excess scope.

VERIFY-AT-BUILD checklist:
1. Verify `nodes/story_rules/science_news.json` exists, has `rules_id == "science_news"`, and every `story_rules/*.json` stem is registered in `nodes/story_packs/banks.json`; missing packs are allowed only for `runnable:false` banks.
2. Verify duplicate-key rejection and control-character lint fire on story-rules JSON, especially decoded backspace from single `\b`.
3. Verify all extracted pattern sets compile with uniform `re.IGNORECASE` and science-lane reasons/spans/compose_flags match the pre-extraction constants.
4. Verify `OTR_LedgerScriptWriter.run()` resolves `story_rules` once after the runnable gate and before the beat loop, and `_w1b_s3_kwargs` passes `stage3_banned_phrases=list(story_rules.banned_phrases)`.
5. Verify `_otr_reroll.py:656` and `_otr_story_spine.py:170/:215` pass `source_bank_id` from ledger/meta.
6. Verify `scripts/story_quality_scan.py` treats `StoryRulesError` as fatal instead of falling into the per-ledger skip at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\story_quality_scan.py:598`.
7. Verify `compose_line_draft` preserves science-lane repo-None `_SYSTEM_PROMPT` identity while non-science repo-None routes through the story-pack resolver.