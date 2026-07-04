VERDICT: yes-with-fixes. The r1 plan is implementable if v1 stays a bridge through existing nodes, but it still needs exact file/module targets, schema names, and test gates before coding.

MUST-FIX BEFORE BUILD:
1. [V1 Contract Shape / StoryInputPacket] [CONFIRMED] The plan names `close_brief`, but the existing `NewsBriefs` schema and writer code use `news_close_brief`. A casual rename will break legacy mirrors and tests. Concrete fix: use canonical `close_brief` only inside `StoryInputPacket`, and define a required adapter mapping `close_brief -> meta.news.news_close_brief`.
2. [First Coding Plan Shape / Step 2] [CONFIRMED] The writer already has `IS_CHANGED` returning `time.time()`, so the first bridge does not need new cache invalidation. The moment source fetch/load moves to a standalone source node, though, that node must not blindly copy writer behavior. Concrete fix: document "writer bridge keeps existing always-run behavior; future source nodes use source hash/refresh nonce/file hash."
3. [First Coding Plan Shape / Step 3] [CONFIRMED] Replacing only `Science story` labels is not enough: `_otr_outline._build_user_prompt` also hardcodes "Plan a science-fiction audio drama outline." Concrete fix: add `source_label` and `story_form_label` or `genre_label` to the outline request/prompt adapter, defaulting to the current text for `science_news`.
4. [First Coding Plan Shape / Step 6-7] [CONFIRMED] Adding `OTR_VisualStyleDirector` and optional inputs means code + `__init__.py` registration + workflow JSON wiring in the same chunk. Concrete fix: split pure style helpers from ComfyUI node wiring; do not register/wire the node until `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock` can consume the policy.
5. [First Coding Plan Shape / Step 7] [CONFIRMED] `OTR_MetaBriefImagePromptGen` optional inputs are currently `image_policy_json`, `consistency_gate_warn_only`, `gate_in`; `OTR_ShotLock` optional inputs are `image_done`, `consistency_gate_warn_only`, `gate_in`. Concrete fix: append `visual_style_policy_json` after current optional entries, and update function signatures with a default `"{}"` at the end only.
6. [First Coding Plan Shape / Step 8] [CONFIRMED] Public-domain text must not start as arbitrary network search. Concrete fix: first implementation accepts operator-supplied local text/path/string and records `source_hash`; no Gutenberg API/search code in first sprint.

SHOULD-FIX:
1. [First Coding Plan Shape] [CONFIRMED] Add exact modules and tests:
   - `nodes/_otr_source_packet.py`
   - `nodes/_otr_visual_style_policy.py`
   - maybe `nodes/_otr_story_blueprint.py`
   - `tests/test_source_packet_contract.py`
   - `tests/test_visual_style_policy.py`
   - targeted extensions to `tests/test_news_interpreter_wiring.py`, `tests/test_lfc_c4_news_used_passthrough.py`, `tests/test_brief_prompt_finishing.py`, `tests/test_workflow_json_wiring_invariants.py`.
2. [Accepted Architecture Decisions] [CONFIRMED] `meta.theme` is computed from `meta.news.script_brief` in the writer and then passed into line composition. Fix: when adding `meta.source`, compute theme from the canonical packet, then mirror enough into `meta.news` so the current line composer remains unchanged in v1.
3. [VisualStylePolicy] [CONFIRMED] Current still prompt code appends `IMAGE_GRADE_TAIL` after `finish_visual_prompt`. Fix: style application must happen at the finalization layer, not merely before LLM prompt generation, or cinematic tails will survive in anime/archival modes.
4. [Rejected Or Deferred Suggestions] [CONFIRMED] The coding plan should state that no new LLM/model widgets are added outside `OTR_LedgerScriptWriter`; existing tests enforce no-widget/model-widget constraints.

OPTIONAL / NICE-TO-HAVE:
1. [First Coding Plan Shape] Add a diagnostic `meta.source_migration` block during the first bridge so test fixtures can assert the mirror source.

CUT THESE (over-engineering):
1. [First Coding Plan Shape] Cut `OTR_SourceBankDirector`, `OTR_StorySourceInterpreter`, and `OTR_StoryDirector` nodes from the first coding sprint. Build pure modules and writer bridge first; new source nodes wait until `science_news` and `media_archive` are proven.
2. [VisualStylePolicy] Cut `model_family_hints` enforcement in v1. Keep it trace-only until the first non-default style profile proves what model constraints actually matter.

