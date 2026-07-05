# Anchor review (Claude, code-grounded) -- 2C source_bank selector wiring -- r1

VERDICT: SHIP WITH FIXES (2 MUST-FIX clarifications, no architectural change)

Grounding performed against the REAL Windows files 2026-07-05:
- nodes/OTR_LedgerScriptWriter.py (INPUT_TYPES ends optional at story_scaffold :2196;
  run() signature :2436-2507, story_scaffold="auto" is the last positional-default
  before the keyword-only refine block; refine `_core` is a locals() filter :2538
  so a new positional-default param auto-threads into refine passes).
- nodes/_otr_creative_prompt_router.py (resolve_creative_system_prompt(repo_id, phase)
  :90; _SCIENCE_BANK_ID transitional binding :84,127).
- nodes/_otr_story_routing.py (require_runnable_bank :443, exported :478).
- Call sites of the resolver: exactly 2 production (_otr_outline.py:1843,
  _otr_line_composer.py:2064), pinned by the caller-count test.
- workflows/otr_scifi_16gb_full.json node id=1: wv_count=25, slot 23="Off",
  slot 24="auto"; story_scaffold has NO inputs[] entry (precedent for widget-only
  append -- codex r2 S4 CONFIRMED at build).
- tests/test_workflow_json_guardrails.py:673-733 (len==25 pin + slot pins).

MUST-FIX
- M1 (CONFIRMED gap in the spec text): the sub-plan says "all 4 existing callers"
  of resolve_creative_system_prompt stay byte-identical, but there are exactly TWO
  production call sites (outline :1843, line_composer :2064) and neither receives
  the writer's widget value today -- both are reached deep inside generate_outline /
  compose paths that only carry creative_repo_id. 2C must define the THREADING PATH
  precisely: writer run(source_bank) -> the outline request / composer request (or
  explicit kwarg through generate_outline(...) and the composer entry) -> the two
  resolver calls pass source_bank_id=<selection>. A default-only param with no
  caller threading would leave the widget DEAD for the science lane and silently
  pinned to science for every lane -- exactly the unwired-widget failure mode
  (2026-06-13 section-4D). Spec must name the two call sites and the carrier
  (request field vs kwarg) before build.
- M2 (CONFIRMED): require_runnable_bank(source_bank) placement -- "before story
  execution" must mean IN run() BEFORE the refine gate at :2532, so a non-runnable
  pick fails once, loudly, before any LLM/model load, and before _refine_loop
  re-entry (which re-enters run() with _refine_active=True and would re-check
  harmlessly). Placement after _resolve_inputs would waste an RSS fetch.

SHOULD-FIX
- S1: INPUT_TYPES currently guards its whole optional build behind a
  try/except-free path except the openrouter probe; the new dropdown pulls from
  the lazy registry INSIDE INPUT_TYPES. Note the INPUT_TYPES comment :1720 says
  "INPUT_TYPES must never raise" for the openrouter probe -- the plan's
  fail-registration-LOUD posture deliberately BREAKS that convention for the
  registry (accepted r1 decision). Add a code comment at the call site saying the
  raise is intentional (no-fallback law) so a future reader doesn't "fix" it.
- S2: the widgets_values slot pin test must ALSO assert the value is a str and a
  registered bank id (cheap cross-check via the routing registry), not just
  == "science_news" -- catches a future re-order.
- S3: _core locals() filter (:2538) auto-carries source_bank -- add a refine-lane
  test that a non-default source_bank survives into a refine pass (assert via the
  resolver call or ledger stamp), since the filter is implicit.

CONFIRMED spec points (no change needed): widget-only append (no inputs[] entry --
matches story_scaffold precedent in the real JSON); slot 25 default "science_news";
guardrail test len 25->26 same commit; story_scaffold stays slot 24 "auto";
non-runnable banks listed in the dropdown (honest error on use);
require_runnable_bank exists and is exported.

UNVERIFIABLE (verify-at-build): exact ComfyUI serialization when the node is
re-saved from the UI vs hand-edited JSON (we hand-edit; the round-trip +
OTR_WorkflowValidator gate covers it).
