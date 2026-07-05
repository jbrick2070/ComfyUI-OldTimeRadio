VERDICT: yes-with-fixes — only one contract ambiguity can produce split-brain interpreter output; close it before build.

MUST-FIX BEFORE BUILD:
1. [1a / 1c] `validate_interpreter_result` validates direct attributes and `model_dump()` values independently but does not require them to match. The writer uses both surfaces: `meta["news"] = briefs.model_dump()` and then direct attrs for locals at `nodes/OTR_LedgerScriptWriter.py:3030-3033`; downstream reads dump fields such as `news_close_brief` at `nodes/OTR_LedgerScriptWriter.py:4330`, `nodes/video_engine.py:1787`, and `nodes/_otr_story_spine.py:212`. A duck-typed interpreter could pass validation while giving different direct vs dump values. Concrete fix: require `dump["casting_brief"] == result.casting_brief`, `dump["script_brief"] == result.script_brief`, `dump["key_terms"] == list(result.key_terms)`, and `.attempts` is `int`; add a mismatched-stub contract test.

SHOULD-FIX:
1. [3] “A non-science bank at run-intent still dies at `require_runnable_bank`” is only true for the currently shipped non-science rows, because [1b] intentionally allows a future non-contract bank when `requires_source_contract=false` and `pipeline.executable=true`. Concrete fix: change to “Current shipped non-science banks still die...”
2. [5] Consolidate all prior verify-at-build flags in section 5, not only the three current bullets. The plan mostly has concrete tests elsewhere, but the final checklist should include them explicitly.

OPTIONAL / NICE-TO-HAVE:
None material.

CUT THESE:
1. [2.6] The focused writer AST check for no `_OTRNI.build_news_briefs` is redundant if the behavioral test proves D.2.5 calls `resolve_interpreter` and no `_OTRNI` references remain. Safe to cut if test count or brittleness matters.

VERIFY-AT-BUILD checklist:
- Confirm writer except topology at `nodes/OTR_LedgerScriptWriter.py:3039-3102`: `SourceInterpretError` only catches interpreter execution failures; contract errors propagate.
- Confirm halt stamp/re-raise preserves underlying `NewsInterpreterError` identity via `exc.__cause__`.
- Confirm `pipelines.json` adds required bool `requires_source_contract` to every row and all `_pipe_row()` fixtures inherit it.
- Confirm no tests pin pipeline row keys byte-for-byte.
- Confirm no remaining `_OTRNI` use in `run()` after reroute.
- Confirm `_otr_ledger_freeze.py:232-250` still expects `meta.news.key_terms` as list.
- Confirm `news_close_brief` dump consumers remain covered, including `nodes/_otr_story_spine.py:212`.
- Run post-change `OTR_WorkflowValidator` + JSON round-trip + link/widget audit with expected no diff.