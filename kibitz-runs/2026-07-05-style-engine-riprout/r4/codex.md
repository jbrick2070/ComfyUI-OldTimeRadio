VERDICT: yes-with-fixes — the plan is close, but it still has build-blocking ambiguity around contract on/off semantics, slug-vs-label threading, and several concrete test/fallback traces that survived the rip-out.

MUST-FIX BEFORE BUILD:
1. [1 / 1a / 5 / 7] `story_scaffold` OFF semantics conflict with “every episode” gets exactly one `build_story_contract()` call and with the new required `meta.style`. Current code only builds/stamps the contract under `_style_grammar_on` in `nodes/OTR_LedgerScriptWriter.py:3337-3362`, while existing readers expect `meta.style` (`nodes/_otr_story_brief.py:561-565`, `nodes/_otr_ledger_freeze.py:582-592`). Concrete fix: state one rule. Recommended: build/stamp `meta.story_contract` + `meta.style` unconditionally after `script_brief` and `cast_seed` exist; let `_style_grammar_on` only govern prompt/climax injection. If OFF is meant to suppress even metadata, define what `meta.style` becomes.

2. [1 / 5 / 7] “Thread `contract.label`/`.slug`” is ambiguous and will produce incompatible outputs. Current former `resolved["style"]` readers include prompt-facing fields (`lock_cast` style prompt in `nodes/_otr_casting.py:291,345,350`; `OutlineRequest.style` prompt in `nodes/_otr_outline.py:297,602,1158`) and slug/ledger-facing fields (`style_descriptor` in `nodes/OTR_LedgerScriptWriter.py:4334-4355`; `meta.visual_plan.style` / `meta.style` in `nodes/OTR_LedgerScriptWriter.py:5631-5636`; story brief treats style as a controlled slug in `nodes/_otr_story_brief.py:561-565`). Concrete fix: define exact mapping: e.g. prompt-facing `Style:` uses `contract.label`; ledger/canonical fields use `contract.slug`; `style_descriptor` either remains slug or is renamed/rewired deliberately.

3. [0 / 2] “No fallback” is not fully specified for `_otr_style_catalog.py`. Current catalog helpers still explicitly fall back: `ending_template_for()` returns a default for unknown slug (`nodes/_otr_style_catalog.py:613-618`), `render_style_grammar()` returns empty string on unknown slug (`nodes/_otr_style_catalog.py:679-684`), and `build_story_contract()` documents “Never raises on a missing style” (`nodes/_otr_style_catalog.py:754-759`). Concrete fix: either explicitly exempt these impossible defensive paths from “no fallback,” or make selected-slug lookup fail loud and update/delete the negative test in `tests/test_announcer_kill2_c1.py:73-80`.

4. [2 / 7] The test rewrite/delete list is incomplete. `tests/test_style_randomization.py:17-53` imports and tests `_resolve_style_rng_seed`, which section 2 deletes. `tests/test_news_briefs_required.py:31-44` passes `style_custom=` into `_resolve_inputs`, which will fail once `style_custom` is deleted from the signature. Concrete fix: add both files to section 4/7’s required test edits; delete or replace `test_style_randomization.py`, and remove `style_custom` kwargs from the news-brief tests.

SHOULD-FIX:
1. [2] Extend the zero-hit grep list to include `_resolve_style_rng_seed`, `OTR_STYLE_SEED`, and `style_descriptor` where it is specifically old picker trace. Current references include `tests/test_audio_byte_identical.py:48,61`, `tests/test_style_randomization.py:5-17`, and `_otr_line_composer.py` comments tying `style_descriptor` to `_otr_style_picker` at `nodes/_otr_line_composer.py:763-764`.

2. [5] Decide whether `meta.gen_params_initial.style` validation is retained only for legacy ledgers or removed as trace. Current freeze validator still has a dedicated block for it at `nodes/_otr_ledger_freeze.py:594-616`; harmless if absent, but contradictory to “only surviving style record is `meta.story_contract`” unless documented as legacy-tolerant validation.

OPTIONAL / NICE-TO-HAVE:
- Update stale module/doc comments in `nodes/_otr_outline.py:6,297-301` and `nodes/_otr_line_composer.py:763-764` so future builders do not reintroduce a user-supplied/picker style mental model.

CUT THESE:
1. [Preamble] Cut the long r1/r2/r3 history from the build handoff. It is useful audit context, but not needed by a builder and duplicates later section-specific decisions.

2. [2] Cut “Safe-removal grep already run, clean” from the plan body. Make it a verify-at-build item instead; the current repo still contains the old symbols, so the statement will only be true after implementation.

VERIFY-AT-BUILD checklist:
- No explicit earlier `UNVERIFIABLE` labels are present in this r4 input. Verify the following build-time items anyway.
- Run zero-hit grep after edits for `_otr_style_picker`, `pick_style`, `StylePick`, `StyleGenerationFailedError`, `_STYLE_PICKER_SEED_POOL`, `_LLM_STYLE_FALLBACK`, `_resolve_style_rng_seed`, `OTR_STYLE_SEED`, `style_pick`, `STYLE_PALETTE`, `_otr_style_palette`, `style_custom`, `_STYLE_CHOICES`, excluding `__pycache__`.
- Validate `workflows/otr_scifi_16gb_full.json` with `OTR_WorkflowValidator`, JSON round-trip, widget-count vs live `INPUT_TYPES`, wired input-name audit, and link referential integrity.
- Confirm writer node `widgets_values` length is 25 and `story_scaffold/source_bank/visual_style` land at indices 22/23/24.
- AST-parse touched `.py` files.
- Run full regression suite plus Bug Bible.
- Confirm a generated ledger stamps `meta.story_contract`, canonical `meta.style`, and no `meta.style_pick` / `gen_params_initial.style*`.
- Confirm RSS fetch/rerank and `news_interpreter` no longer accept or render style text.
- Confirm C7 determinism: one contract draw per episode, keyed by `cast_seed`, with no second `select_style()` path.