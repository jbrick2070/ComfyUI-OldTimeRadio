# r4 JUDGMENT (Cowork Claude, anchor + judge) -- style engine rip-out. r4: CONVERGENCE.

Panel this round: Codex (gpt-5.5, high reasoning) -- OK, strong grounded
review, found real gaps r1-r3 missed. Sonnet subagent (spawned via Agent
tool) -- OK, found one additional real gap (a second freeze-validator
block) independently. Antigravity: not used (dropped per operator
directive 2026-07-05).

VERDICT: NOT a clean "yes" -- but CLOSE. Four concrete, narrow fixes
needed, all mechanical (test list completeness + two clarifying rules),
zero new architectural rethink required. After folding these in, the plan
is code-ready.

## Grounded and ACCEPTED (verified against the real files)

1. **MUST-FIX (Sonnet, r4): a second freeze-validator block reads
   `meta.gen_params_initial.style` separately from the `meta.style` check.**
   Confirmed by direct read: `nodes/_otr_ledger_freeze.py:594-616`
   ("S25 / MG-6 (BUG-LOCAL-216)") validates that field's snake_case shape.
   Not build-breaking as coded (`gp_initial.get("style")` -> `None` once the
   stamp is gone -> the `isinstance(...) and gp_style` guard short-circuits,
   silent no-op) -- but it is dead code pointed at a field that no longer
   exists, which the project's no-dead-code directive forbids. FOLDED IN:
   delete this block in the same edit as the `gen_params_initial` stamp
   deletion; find its pinning test at build time (grep `_otr_ledger_freeze`
   tests for `gen_params_initial`) and update it there -- no single test
   file could be identified by name-pattern search from here.

2. **MUST-FIX (Codex): the test-deletion/rewrite list was incomplete on two
   real, live-verified files.** Confirmed by direct read:
   - `tests/test_style_randomization.py` imports
     `_resolve_style_rng_seed` directly from `OTR_LedgerScriptWriter`
     (line 17) and asserts on its OS-entropy/`OTR_STYLE_SEED` contract
     (BUG-LOCAL-270 regression guard). Section 2 already marks
     `_resolve_style_rng_seed`/`picker_rng` for deletion "if nothing else
     calls them" -- this file IS that caller. FOLDED IN: this whole test
     file is deleted along with the picker RNG plumbing it exists solely
     to pin (it is a positive pin on the OLD system being retired, not a
     "negative test of the new one" -- deleting it is consistent with
     section 0's own rule, not a violation of it).
   - `tests/test_news_briefs_required.py:34,43` passes `style_custom=`
     as a kwarg into a resolver call. Confirmed live. FOLDED IN: added to
     the test-rewrite list; strip the `style_custom=` kwarg from both call
     sites once the resolver branch is deleted.

3. **MUST-FIX (Codex), narrowed after verification: prompt-facing vs
   ledger-facing threading of `contract.label` vs `contract.slug` was
   ambiguous.** Confirmed real by direct read: `nodes/_otr_casting.py`
   builds a human-readable casting prompt line `f"Style: {style_str}"`
   (~line 350) -- a PROMPT-facing consumer, wants prose (`contract.label`).
   `nodes/_otr_story_brief.py:565` and the freeze validator's slug-shape
   check treat `meta.style` as a controlled, well-formed snake_case
   SLUG -- a LEDGER-facing consumer, wants `contract.slug`. FOLDED IN:
   explicit rule added to the plan -- every prompt-facing string (casting
   prompt, outline `Style:` prompt) uses `contract.label`; every
   ledger/meta-facing field (`meta.style`, `meta.visual_plan.style`,
   `style_descriptor`) uses `contract.slug`. Do not mix the two.

4. **MUST-FIX (Codex), narrowed after verification: `story_scaffold=off`
   metadata semantics needed to be stated explicitly, not left implicit.**
   Confirmed by direct read (`OTR_LedgerScriptWriter.py:3330-3362` +
   `tests/test_announcer_kill2_c1.py`'s `TestWriterOffFlagLedgerMeta`):
   this is EXISTING, ALREADY-SHIPPED, ALREADY-TESTED behavior, not a new
   ambiguity the rip-out introduces -- `_style_grammar_on=False` means
   `contract` stays `None` by design ("OFF => contract stays None => no
   meta.story_contract => byte-identical", per the KILL-2 comment), and
   the freeze validator already treats a missing `meta.style` as a
   WARNING, not an error (`_otr_ledger_freeze.py:582-586`). FOLDED IN: the
   plan now states this explicitly instead of implying `meta.style`
   always exists -- when scaffold is off, `meta.style` is simply not
   stamped (tolerated, pre-existing, warning-only), consistent with
   today's behavior. No code change needed here beyond what section 1/2
   already do; this is a documentation-precision fix only.

5. **REJECTED (Codex's MUST-FIX #3, on catalog-helper "no fallback"):**
   Codex flagged `ending_template_for()`, `render_style_grammar()`, and
   `build_story_contract()`'s documented "never raises on a missing
   style" as violations of section 0's "no fallback" rule, and suggested
   deleting `tests/test_announcer_kill2_c1.py`'s
   `test_missing_style_never_raises`. Checked directly: this test
   exercises `StoryContract`'s own constructor tolerance to an already-
   empty slug -- it is testing the CATALOG MODULE's own documented,
   intentional defensive contract (`get_style` returns `None` for an
   unmatched slug and the pure helpers degrade gracefully), not the
   retired dual-selector system this rip-out is killing. Section 0's "no
   fallback" rule targets the STYLE-SELECTION mechanism (competing
   pickers silently reverting into each other) -- it does not require
   `_otr_style_catalog.py`'s own internal defensive helpers to start
   raising on malformed input. FOLDED IN AS A CLARIFICATION ONLY: the
   plan now states this scope boundary explicitly so a future builder
   does not misapply section 0 to this module's own helper functions.
   The test and the helpers are UNCHANGED.

6. **CUT / accepted (Codex): trim the plan's audit-trail preamble for the
   eventual build handoff, and demote "safe-removal grep already run,
   clean" to a verify-at-build item** (the symbols still exist in the repo
   right now -- the statement is only true after the edit lands, not
   before). Both cosmetic, folded into the final doc pass.

## Rejected / not folded (with reason)

- Codex's suggestion to delete `test_missing_style_never_raises` — see
  item 5 above; this test pins a real, intentional, unrelated catalog
  contract and stays.
- No other claims from either reviewer were rejected -- both Codex and
  the Sonnet subagent had a clean grounding record this round, consistent
  with r1-r3.

## Convergence statement

r4 found four real, narrow, purely mechanical gaps (one dead
freeze-validator block, two missing test files, one label/slug threading
rule, one documentation-precision item on already-shipped OFF semantics)
and correctly rejected one over-broad suggestion (rewriting catalog-helper
robustness). None of these require new architecture, new parameters, or
revisiting any of the r1-r3 locked decisions. **This plan is now CODE-READY.**
