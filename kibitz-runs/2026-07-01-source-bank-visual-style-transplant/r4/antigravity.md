VERDICT: yes-with-fixes. The plan is highly mature but has minor omissions regarding model serialization fields, system prompt default values, and helper parameter definitions.

MUST-FIX BEFORE BUILD:
1. [Must-Fix Wiring Decisions, 7] News model serialization: `news_interpreter.NewsBriefs` (the Pydantic model in `nodes/news_interpreter.py`) currently lacks `title`, `headline`, and `link` fields. When `OTR_LedgerScriptWriter` calls `briefs.model_dump()`, Pydantic will omit these fields, preventing them from being stored in the ledger meta or accessed downstream. Fix: Either declare optional `title`, `headline`, and `link` fields in the `NewsBriefs` Pydantic model (in `nodes/news_interpreter.py`), or explicitly add them to the dictionary returned by `briefs.model_dump()` within the writer node.
2. [Must-Fix Wiring Decisions, 8 & 10] Visual style policy stamping: To avoid import loops or disk I/O in prompt-finishing helper modules (e.g., `_otr_story_brief_helpers.py`), the orchestrator/MetaBrief node must stamp the resolved visual style policy dictionary directly into the ledger under `meta.visual_style`, allowing `finish_visual_prompt(meta, ...)` to retrieve the fields directly from `meta` without imports. [ASSUMPTION]

SHOULD-FIX:
1. [Must-Fix Wiring Decisions, 3] OutlineRequest defaults: The default values for the new fields in `OutlineRequest` (e.g., `story_form_label`, `source_material_label`, `source_develop_verb`, etc.) must be explicitly defined to guarantee science behavior preservation for old callers. Fix: Define default values: `story_form_label="science-fiction audio drama"`, `source_material_label="Science story"`, `source_develop_verb="extrapolate dramatically from this story"`, `outline_rules_extra=""`, `forbidden_plot_patterns=()`, and `outline_system_prompt=None`.
2. [Must-Fix Wiring Decisions, 1] `resolve_story_model_id` module location: The plan does not specify which module this helper belongs to or what the type/expected behavior of the `rng` parameter is. Fix: State that this function must be implemented inside `nodes/_otr_story_model_catalog.py`, and clarify that `rng` can be either a `random.Random` instance or an integer seed, defaulting to a deterministic fallback if `rng` is None. [ASSUMPTION]
3. [Must-Fix Wiring Decisions, 2] `compose_source_coda` signature: The signature of `compose_source_coda()` is not defined, leading to implementation ambiguity. Fix: Define that it should be placed in `nodes/_otr_line_composer.py` and accept `(*, creative_fn, source_close_brief, premise, intro_text="", ...)` to mirror `compose_news_coda`.

OPTIONAL / NICE-TO-HAVE:
- Include documentation for each new visual style preset detailing what positive and negative prompt tails are mapped.

CUT THESE:
- None — plan is already lean and staged.

VERIFY-AT-BUILD checklist:
1. Verify that positional widget indices are untouched in `workflows/otr_scifi_16gb_full.json` after adding the new writer inputs.
2. Verify that whitelists in `scripts/otr_api.py` and `nodes/_otr_workflow_apply.py` are updated with the new writer parameters.
3. Verify that the two-slot routing works when `creative_writing_model` and `technical_model` are set to different values.
4. Verify that visual prompt finishing correctly extracts policy tails from `meta.visual_style` and applies them without leaking sci-fi terms into non-sci-fi genres.
