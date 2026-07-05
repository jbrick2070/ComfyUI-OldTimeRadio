# r3 JUDGMENT (Cowork Claude, anchor + judge) -- style engine rip-out. r3: WIRING.

Panel this round: Codex (gpt-5.5, high reasoning) -- OK, exceptionally
strong grounded review. Sonnet subagent (spawned via Agent tool, per
operator directive to replace antigravity) -- OK, strong grounded review,
independently found the same core ripple (RSS fetch -> story_orchestrator)
Codex found, plus corroborated the widget-index math against the live
JSON. Antigravity: not used this round (dropped per operator directive
2026-07-05 -- "codex and sonnet are your local panel").

This was the most consequential round yet. Codex found two genuine
build-breaking sequencing bugs neither the plan, my own prior anchors, nor
the Sonnet review caught. Both verified true on direct read.

## Grounded and ACCEPTED (all verified against the real files)

1. **CRITICAL, build-breaking: `lock_cast()` reads the deleted
   `resolved["style"]` BEFORE the contract exists.** Confirmed by direct
   read: `_OTRCAST.lock_cast(..., style=resolved["style"], ...)` at
   `OTR_LedgerScriptWriter.py:3193-3198`, but `build_story_contract()` (the
   sole surviving style source) isn't called until `:3337-3345` -- ~150
   lines LATER in the same function. If `resolved["style"]` is simply
   deleted per the plan's section 1, this is an immediate `KeyError`/
   `NameError` at the very first place style is needed. MUST-FIX: move the
   `build_story_contract()` call to BEFORE `lock_cast()` (after
   `script_brief` and `cast_seed` both exist, ~line 3174), and thread
   `contract.label` (or `.slug`) into `lock_cast` and every other caller
   that currently reads `resolved["style"]` -- do not leave any read of a
   value that no longer exists.

2. **CRITICAL: a circular dependency the plan didn't see.**
   `news_interpreter.build_news_briefs()` (via `_otr_source_payload.py:
   233-259`) takes a `style` parameter and uses it in a prompt line
   (`nodes/news_interpreter.py:719, 731-740`) -- but this call happens
   BEFORE `script_brief` exists, and `build_story_contract()` needs
   `script_brief` as an input. You cannot build the contract early enough
   to feed `news_interpreter`, and you cannot keep feeding
   `news_interpreter` a deleted `resolved["style"]`. MUST-FIX: strip the
   `style` field from `build_news_briefs()`/`news_interpreter.py`'s prompt
   entirely (it's sourcing-stage, pre-contract) rather than trying to
   thread a value that structurally can't exist yet at that point in the
   pipeline.

3. **The RSS/rerank ripple is WIDER than r2 found, confirmed independently
   by BOTH reviewers.** `_fetch_rss_seed_or_die` is not called directly
   inside `OTR_LedgerScriptWriter.py` -- its real caller is
   `_otr_source_payload.py:219-230`'s `_fetch_science_rss(*, bank,
   style_slug, technical_model)`, documented as "the S31 B6 slot-label/id
   agreement invariant." `_resolve_inputs` passes `style_slug=` at
   `OTR_LedgerScriptWriter.py:1404-1408` (Codex) / confirmed live at
   `_otr_source_payload.py:219-230` (Sonnet, independently). Downstream,
   `story_orchestrator.py` uses `style` for LLM rank-prompt text
   (`genre_human = (style or "sci-fi").replace("_", " ")`, line ~1490) at
   FOUR call sites (`:1670-1682`, `:1843-1849`, `:1934-1940`,
   `:1957-1964`). MUST-FIX: this is not a self-contained parameter removal
   -- the fetcher contract, the `_otr_source_payload.py` wrapper, the
   writer call, `story_orchestrator.py`'s ranking/history signatures, AND
   `tests/test_writer_input_resolve.py` (which AST-asserts the 2nd
   positional-arg contract) all change together, in the same edit. Leaving
   `story_orchestrator.py`'s hardcoded `"mission_control_procedural"`
   default in place would be exactly the "no fallback" violation section 0
   bans.

4. **`meta.style` is read by MORE live consumers than the plan assumed.**
   The plan's section 1/5 claimed `meta.story_contract` becomes the ONLY
   surviving style record. Confirmed false: the writer also stamps
   `meta["visual_plan"]["style"]` and `meta["style"]`
   (`OTR_LedgerScriptWriter.py:5631-5636`); `_otr_story_brief.py:565` emits
   `STYLE: {meta.get('style')}`; the freeze validator audits `meta.style`
   (`nodes/_otr_ledger_freeze.py:582-592`). MUST-FIX: either keep a
   canonical `meta.style` field DERIVED from `meta.story_contract.slug`/
   `.label` (cheap, one-line addition, keeps every existing reader
   working), or update every one of these consumers/validators to read
   `meta.story_contract.*` directly. Do not delete the stamp with no
   replacement -- that breaks the freeze validator and the story-brief
   text.

5. **Widget-reindex test scope is much wider than r2's sweep found, in
   BOTH directions (Codex full list, Sonnet independent JSON read match).**
   Confirmed live widget order (`OTR_LedgerScriptWriter.py:1919-2297` +
   the actual `workflows/otr_scifi_16gb_full.json` node-1 array, 27
   values, matching exactly): `[8] style, [9] style_custom, [10]
   creativity, [11] perfect_run_spacesaver, [12] min_p, [13]
   repetition_penalty, [14] max_new_tokens_cap, [15] lemmy_cameo, [16]
   use_exchange, [17] enable_production_stage3_validators, [18]
   news_briefs_required, [19] openrouter_slot_a_model, [20]
   openrouter_slot_b_model, [21] comfy_slot_a_model, [22]
   comfy_slot_b_model, [23] refine_target_grade, [24] story_scaffold,
   [25] source_bank, [26] visual_style`. Post-deletion: length becomes 25;
   `story_scaffold` -> [22], `source_bank` -> [23], `visual_style` -> [24].
   Index-pinned tests needing updates beyond `test_workflow_json_
   guardrails.py`: `tests/test_otr_api_companions.py:34-214,466`,
   `tests/test_source_bank_widget_2c.py:322-323`,
   `tests/test_visual_style_widget_3c.py:172-174`,
   `tests/test_openrouter_slot_widgets_s2.py:62`, plus any API-type
   fixtures assuming old positions.

6. **Deleting `_otr_style_picker.py` has TWO import sites + a telemetry
   stamp, confirmed independently by both reviewers.** Import at
   `OTR_LedgerScriptWriter.py:2797`, call at `:2994-3005`, phase telemetry
   stamp at `:5545-5549`, AND a second, easy-to-miss import inside an
   in-file smoke-test helper at `:6103-6155`. MUST-FIX: delete all of these
   in the SAME code edit as the file deletion -- do not run any
   intermediate validation between the file delete and the writer cleanup,
   or ComfyUI fails to import the node at boot (ImportError on node
   registration).

7. **Sequencing note on the bank-scope doc comment (section 1b):** since
   finding #1 moves the `build_story_contract()` call site earlier (before
   `lock_cast`, not at its current ~3340 location), the doc-only bank-scope
   comment must land at the NEW call site, not the old one -- otherwise
   step 6 attaches a comment to code that's about to move, and it gets
   silently orphaned or misplaced during step 1's rewire. Sequence: do
   step 1 first (move + rewire the call site), THEN add the doc comment at
   its final resting place.

8. **CUT, accepted:** no compatibility shim for old `style`/`style_custom`
   payloads on already-rendered episodes. Live workflow JSON and code are
   a clean break; a migration shim would violate the no-back-compat
   directive. (Historical ledger files on disk stay untouched regardless,
   per section 5 -- that's not a "shim," it's just not touching archived
   output.)

## Rejected / not folded

- None. Every claim from both reviewers checked out on direct grounding
  (spot-verified: the `lock_cast` call site, `_otr_source_payload.py`'s
  `_fetch_science_rss`/`_interpret_news` signatures). Zero misreads.

## Convergence statement

r3 was NOT a clean pass -- it found two build-breaking sequencing bugs
(findings #1, #2) that would have made this plan fail at the first actual
code edit, plus confirmed and widened the RSS/story_orchestrator and
widget-reindex findings from earlier rounds. The plan needs one more
revision pass (moving the contract-build call earlier, stripping style
from `news_interpreter`, adding `story_orchestrator.py` + `_otr_source_
payload.py` + the expanded test list to the deletion/rewire scope) before
r4 (convergence) can plausibly say "no new must-fix." Carry all eight
accepted items into the v5 plan revision, then run r4 against that.
