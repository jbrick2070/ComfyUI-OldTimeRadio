# 2C WIRING PLAN (r2-hardened) -- source_bank selector on OTR_LedgerScriptWriter

Supersedes r1/final.md. Panel = codex (antigravity credit-bugged, dropped);
Claude anchor+judge. All codex r2 claims grounded CONFIRMED against the real files.

## Changes (one commit: code + JSON + tests together)

1. **Widget**: `source_bank` appended AFTER `story_scaffold` at the END of optional.
   Choices = bank IDS from the lazy registry (labels may appear in tooltip only --
   values stay stable ids). Default "science_news". Non-runnable banks listed
   (honest error on use). Registry failure RAISES out of INPUT_TYPES = registration
   fails LOUD; code comment marks the deliberate exception to the :1720
   "INPUT_TYPES must never raise" convention.
   Registration-failure test (codex r2 M5): call `_otr_story_routing._clear_caches()`
   then monkeypatch the registry source (or `list_bank_ids`) to raise; assert
   `OTR_LedgerScriptWriter.INPUT_TYPES()` raises `StoryRoutingError` -- proving it
   is NOT swallowed by the existing openrouter-probe try/except.

2. **Workflow JSON**: node 1 wv slot 25 = "science_news" (25 -> 26 values). NO
   inputs[] entry (story_scaffold precedent). OTR_WorkflowValidator + round-trip +
   link/widget audit; UTF-8 no BOM.

3. **Tests updated SAME COMMIT (codex r2 M4 -- the full pinned-order set, not just
   the guardrails file):**
   - `tests/test_workflow_json_guardrails.py:673-733`: len 26; slot 24 "auto";
     slot 25 == "science_news" AND a registered bank id.
   - `tests/test_story_scaffold_toggle.py:50-53`: `order[-1] == "story_scaffold"`
     becomes `order[-1] == "source_bank"` + story_scaffold at -2 (CONFIRMED pin).
   - `tests/test_openrouter_slot_widgets_s2.py:51-63` widget-order/count pins.
   - `tests/test_otr_api_companions.py:148-153,178-204` writer widget-vector shape.
   - New INPUT_TYPES positional test: source_bank is LAST optional.

4. **run() signature**: `source_bank="science_news"` after `story_scaffold="auto"`,
   before the `*` block.
   **Refine capture root-cause fix (codex r2 M1 -- CONFIRMED LATENT BUG):** the
   `_core = locals()` filter at :2538 currently captures `os` (module) and
   `_scaffold` (str), so `self.run(**_core)` in `_refine_loop` (:2299) is a
   TypeError whenever refine is enabled -- broken since story_scaffold (2026-06-24)
   added those locals above the capture. Fix at the root IN THIS CHUNK: build
   `_core` by filtering `locals()` against
   `inspect.signature(type(self).run).parameters`, excluding the keyword-only
   refine internals + `self` + `refine_target_grade`. Add a refine-lane test:
   refine enabled (passes>=2, monkeypatched grader) completes without TypeError
   AND a non-default source_bank survives re-entry. Log this as a bug entry
   (BUG_LOG_2026-06.md next BUG-LOCAL id) -- pre-existing, found by kibitz r2.

5. **Run-intent gate placement (codex r2 M2, tightened):**
   `require_runnable_bank(source_bank)` is the FIRST statement in run() --
   before `_apply_story_scaffold_env` (env mutation :2525), the refine gate
   (:2532), budget resets (:2560-2590), `_resolve_inputs()`/RSS (:2592, :1157).
   Ordering test: non-runnable pick raises StoryBankNotRunnableError and a
   monkeypatch sentinel proves `_fetch_science_news` AND
   `_apply_story_scaffold_env` were never called.

6. **Threading (codex r2 M3 -- full chain, both seams):**
   - `resolve_creative_system_prompt(repo_id, phase, source_bank_id="science_news")`
     -> `resolve_story_pack(source_bank_id)`; `_SCIENCE_BANK_ID` literal survives
     only as the default.
   - Outline seam: writer -> `generate_outline(...)` gains
     `source_bank_id="science_news"` -> the :1843 resolver call.
   - Composer seam (the easy under-wire): `compose_line()` (:2451 def; forwards
     to `compose_line_draft()` :2063 resolver call) -- BOTH get
     `source_bank_id="science_news"`; `compose_line_draft` passes it to the
     resolver; the writer passes it at ALL THREE call sites (:4581, :4649, :4788).
     If the exchange path (`_otr_compose_exchange`) reaches compose_line_draft
     independently, thread it there too (verify at build; use_exchange=True in the
     shipped bake so this path is LIVE).
   - Threading test (codex r2 S1): monkeypatch `resolve_story_pack`, run with a
     non-default (runnable-stubbed) bank, assert the selected id reaches both the
     outline and composer resolver calls. Caller-count AST pins unaffected (they
     count Call nodes, not kwargs -- codex CUT-1 accepted).

## Acceptance
Suite + Bug Bible + B7 + test_audio_byte_identical green; widget live in the real
JSON slot 25; validator green; gate-first ordering test green; threading test green;
refine TypeError bug fixed + regression test; zero episode change at defaults.

## Judgment log (r2)
Accepted: codex M1 (locals() leak -- CONFIRMED at :2524-2545, os/_scaffold not in
the exclusion tuple; real latent TypeError), M2 (gate truly first -- env mutation
:2525 precedes the old spot), M3 (compose_line/compose_line_draft chain + 3 writer
call sites CONFIRMED :4581/:4649/:4788), M4 (test_story_scaffold_toggle pin
CONFIRMED :50-53; other named files accepted with build-time verify), M5
(registry cache -> _clear_caches in the test), S1 (direct threading test), S2
(api-companions fixtures -- fold into M4 list), OPT (ids as values, labels in
tooltip), CUT-1 (drop the conditional caller-count-kwargs line).
Rejected: none this round.
Verify-at-build: exchange-path threading; exact api-companions/openrouter-s2
assert shapes.
