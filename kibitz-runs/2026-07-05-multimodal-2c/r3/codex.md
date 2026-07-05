VERDICT: no. The plan under-wires the live prompt paths and headless patch surfaces.

MUST-FIX BEFORE BUILD:
1. [6] Outline threading is underspecified and mismatched to the real interface. `generate_outline()` calls `resolve_creative_system_prompt(creative_repo_id, phase="outline")` once at `nodes/_otr_outline.py:1843`, but the live outline has three separate stage system prompts at `nodes/_otr_outline.py:1870`, `nodes/_otr_outline.py:1998`, and `nodes/_otr_outline.py:2103`; the science pack exposes `outline_macro_system`, `outline_phase_system`, and `outline_beat_system` at `nodes/story_packs/science_news/science_news_default.json:8-10`, not a single `outline` seam. Concrete fix: add explicit resolver phases for the three outline seams and thread `source_bank_id` to each stage, or explicitly leave outline constant and remove the false outline-threading claim.

2. [6] `compose_line()` recursive repair paths will silently fall back to the default bank unless `source_bank_id` is threaded through every internal recursive call. The plan only names the writer’s three call sites, but `compose_line()` calls itself at `nodes/_otr_line_composer.py:2507`, `nodes/_otr_line_composer.py:2664`, and `nodes/_otr_line_composer.py:2762`. Concrete fix: add `source_bank_id` to `compose_line()`, pass it to `compose_line_draft()`, and pass it through all three recursive `compose_line()` calls.

3. [6] The live `use_exchange=True` path does not reach `compose_line_draft()`; it builds its own hard-coded system prompt. The plan says “If the exchange path reaches compose_line_draft independently,” but `_otr_compose_exchange` calls `build_exchange_prompt()` and then `generate_fn()` directly at `nodes/_otr_compose_exchange.py:533-546`; its system prompt is hard-coded at `nodes/_otr_compose_exchange.py:384-429`. Workflow node 1 ships `use_exchange` on in `workflows/otr_scifi_16gb_full.json` and is pinned by `tests/test_workflow_json_guardrails.py:681-685`. Concrete fix: either add a source-bank-aware exchange seam and thread `source_bank_id` through `run_exchange_prepass()` / `compose_exchange()` / `build_exchange_prompt()`, or explicitly disable/bypass exchange for any future runnable non-science bank.

4. [1/2] The plan omits the headless creative patch whitelists, so automation cannot set the new widget. `scripts/otr_api.py:753-764` and `nodes/_otr_workflow_apply.py:489-504` whitelist creative widgets and currently do not include `source_bank`; parity is pinned at `tests/test_workflow_apply.py:258-261`. Concrete fix: add `source_bank` to both `CREATIVE_WHITELIST` sets and keep the parity test green.

SHOULD-FIX:
1. [4/5] The plan does not say where `source_bank` is stored in the resolved runtime/config surface. `_resolve_inputs()` returns the canonical writer resolved dict at `nodes/OTR_LedgerScriptWriter.py:1374-1422`, and all later metadata/config stamping reads from `resolved[...]`. Concrete fix: add `source_bank` to `_resolve_inputs()` and `resolved`, even if the first build only uses it for the run gate and prompt resolver; this gives tests and ledger/meta stamping one authoritative value.

2. [3] The test list updates positional fixtures but misses a direct API-patcher test for `source_bank`. `scripts/otr_api.py:413-527` maps widget name to serialized slot and will reject on vector drift. Concrete fix: add a test that `patch_widget_by_name(..., "source_bank", "science_news", ...)` lands at slot 25 in the updated fixture.

OPTIONAL / NICE-TO-HAVE:
- [1] Add one assertion that the `source_bank` COMBO choices are exactly `list_bank_ids()` order and include non-runnable ids, so the UI honesty contract is pinned.

CUT THESE (over-engineering):
1. [3] Cut any new caller-count AST pin for kwargs. The current risk is missing runtime propagation through recursive and exchange paths, not call count; a monkeypatched resolver/threading test is more direct.