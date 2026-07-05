VERDICT: yes-with-fixes. The main change is implementable, but the plan misses current refine kwargs behavior and several pinned widget-order tests that will fail.

MUST-FIX BEFORE BUILD:
1. [4] `locals()` auto-carry is not safe for refine re-entry. `nodes/OTR_LedgerScriptWriter.py:2538-2545` currently captures non-`run()` kwargs like `os` and `_scaffold`, and `_refine_loop` forwards `_core_kwargs` into `self.run(**_core_kwargs)` at `nodes/OTR_LedgerScriptWriter.py:2299-2305`. Adding `source_bank` does not fix that; the refine-lane test will hit unexpected keyword failures when refine is enabled. Concrete fix: replace the `locals()` capture with an explicit dict of real public `run()` args, including `source_bank`, or filter against `inspect.signature(self.run).parameters` while excluding keyword-only refine internals.

2. [5] The planned run-intent gate is not actually “before ANY side effect” if placed only before the refine gate at `nodes/OTR_LedgerScriptWriter.py:2532`. `_apply_story_scaffold_env(story_scaffold)` mutates process env at `nodes/OTR_LedgerScriptWriter.py:2524-2525` before that point. Concrete fix: call `require_runnable_bank(source_bank)` before `_apply_story_scaffold_env`, before the refine delegation, before budget resets at `nodes/OTR_LedgerScriptWriter.py:2560-2590`, and before `_resolve_inputs()` / RSS fetch at `nodes/OTR_LedgerScriptWriter.py:2592-2622` and `nodes/OTR_LedgerScriptWriter.py:1157`.

3. [6] “composer entry kwarg” is underspecified and easy to under-wire. The resolver call is inside `compose_line_draft()` at `nodes/_otr_line_composer.py:2063-2066`, while the writer calls `compose_line()` at `nodes/OTR_LedgerScriptWriter.py:4581-4588`, `4649-4660`, and `4788-4798`; `compose_line()` forwards to `compose_line_draft()` at `nodes/_otr_line_composer.py:2451-2463`. Concrete fix: add `source_bank_id: str = "science_news"` to both `compose_line()` and `compose_line_draft()`, forward it at `compose_line_draft(...)`, and pass it in every writer `compose_line()` call that is meant to use the selected bank.

4. [3] Updating only `tests/test_workflow_json_guardrails.py` will not produce a green build. Existing tests pin `story_scaffold` as the last optional widget and total widget count 25: `tests/test_story_scaffold_toggle.py:50-53` and `tests/test_openrouter_slot_widgets_s2.py:51-63`. Concrete fix: update those tests so `story_scaffold` remains slot 24 and `source_bank` becomes slot 25 / last optional, with total 26.

5. [1] The registration-failure test is not precise enough because `_otr_story_routing` caches the registry in `_REGISTRY` at `nodes/_otr_story_routing.py:106-107`, and `list_bank_ids()` reads the cached registry at `nodes/_otr_story_routing.py:423-425`. Concrete fix: in the broken-registry test, call `_clear_caches()` or monkeypatch `list_bank_ids()` itself, and assert `OTR_LedgerScriptWriter.INPUT_TYPES()` raises `StoryRoutingError` instead of being swallowed by the existing broad `INPUT_TYPES` try/except pattern at `nodes/OTR_LedgerScriptWriter.py:1716-1722`.

SHOULD-FIX:
1. [6] Add a direct resolver threading test, not only caller-count tests. Existing AST tests count resolver call sites and phase literals in `tests/test_creative_prompt_router.py:172-191` and `tests/test_creative_prompt_routing_wired.py:88-102`; they will not prove `source_bank_id` is passed. Concrete fix: monkeypatch `resolve_creative_system_prompt` or `resolve_story_pack` and assert the selected non-default bank reaches both `nodes/_otr_outline.py:1843-1845` and `nodes/_otr_line_composer.py:2064-2066`.

2. [2] Update API companion fixtures if they are part of the acceptance subset. `tests/test_otr_api_companions.py:148-153` and `tests/test_otr_api_companions.py:178-204` still model `story_scaffold` as the final writer widget at slot 24. verify: whether this file is run in the planned suite subset.

OPTIONAL / NICE-TO-HAVE:
- [1] Use the bank labels from `nodes/story_packs/banks.json` only in tooltip/help text; keep the dropdown values as stable bank ids to avoid positional/schema drift.

CUT THESE (over-engineering):
1. [6] “Caller-count pin test updated only if it counts kwargs” is unnecessary as written. The existing caller-count tests count `ast.Call` nodes, not keyword arguments, so the new bank threading should be covered by a targeted capture test instead.