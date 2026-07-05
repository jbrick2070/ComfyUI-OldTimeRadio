# r1 JUDGMENT (Cowork Claude, anchor + judge) -- style engine rip-out. r1: ARC / COHERENCE.

Panel this round: Codex (gpt-5.5, reasoning=high) -- OK, full grounded review.
Antigravity (gemini-3.5-pro) -- NO OUTPUT after ~25 min (antigravity.log
stayed 0 bytes, no antigravity.md written); process terminated rather than
left to hang indefinitely. Treated as FAILED-THIS-ROUND, not folded. Retry
antigravity on r2 before trusting it dropped for the rest of the arc.

Codex's review changed the plan more than mine did -- three of its five
MUST-FIXes are new facts my anchor review did not have, and I verified all
of them against the real files before folding them in.

## Grounded and ACCEPTED (all verified against the real files just now)

1. **A third UI control was missing from the plan entirely: `story_scaffold`**
   (`OTR_LedgerScriptWriter.py:2244-2260`, `["auto","on","off"]`, appended
   2026-06-24 as "the single user-facing control over the whole bundled
   scaffold"). `_apply_story_scaffold_env` (line 1710) mutates
   `OTR_ENABLE_STYLE_GRAMMAR` directly from this widget. CONFIRMED by
   direct read. This resolves section 6's "open question" -- it is not an
   abstract env-var design fork, it is an ALREADY-SHIPPED widget whose
   `off` setting explicitly means "no style catalog, no climax-shape
   grammar... the writer's own take" (verbatim tooltip). That is a real,
   named, user-facing SECOND MODE, not an operator escape hatch -- exactly
   the kind of duality the "no fallback, one engine" directive targets.
   REVISED SECTION 6: this widget must be an explicit, first-class decision
   in the plan, not a footnote -- either (a) keep it as an intentional
   creative option ("scaffold off" is a legitimate distinct story mode, not
   a fallback, and survives because it's symmetric/documented, not a silent
   revert) or (b) delete it along with the rest of the old duality. Cannot
   ship without picking one.

2. **The MusicGen "no fallback = author 90 cue triples" cost line item is
   WRONG and should be CUT.** Verified: `nodes/_otr_music_prompt.py:76-99`
   `compose_music_prompt()` builds every cue prompt from `meta` brief
   fields (`story_brief_terms`, `music_mood_terms`, keyword-mined
   `script_brief`) -- zero reference to `_otr_style_palette.STYLE_PALETTE`.
   Grep of `nodes/*.py` for `STYLE_PALETTE`/`_otr_style_palette` returns
   ONLY `_otr_style_palette.py` itself -- no runtime consumer. The
   BUG-LOCAL-216 architecture (style slug -> palette cue lookup) has been
   superseded by the brief-driven composer at some point after that fix
   landed, and `_otr_style_palette.py` is now dead weight relative to
   runtime, kept alive only by its own drift test. REVISED SECTION 3: cut
   the 90-cue-authoring requirement entirely; instead decide whether
   `_otr_style_palette.py` + `test_style_palette_drift.py` should be
   deleted outright as part of "no trace of the retired system" (they are
   the exact class of orphaned code that directive targets), OR kept only
   if some non-runtime consumer still needs it (verify-at-build: check for
   any dynamic/string-based lookup before deleting). This removes what was
   the plan's single largest cost item.

3. **The deletion sweep in section 2 was too narrow.** Grep confirms
   `_otr_style_picker`/`pick_style` are referenced in SEVEN test files, not
   the two my anchor review and the original plan accounted for:
   `test_otr_style_picker.py`, `test_pick_style_routing.py`,
   `test_helper_paired_signatures.py`, `test_audio_byte_identical.py`,
   `test_story_pack_stage1.py`, `test_writer_paired_wiring.py`,
   `test_meta_slot_transitions.py`. Section 2's grep-sweep checklist must
   explicitly include a test-suite pass, not just a `nodes/` pass.

4. **`style_custom` is a live, unresolved bypass of "one engine."**
   Confirmed at `_resolve_inputs` (`OTR_LedgerScriptWriter.py:1345-1353`):
   free-text `style_custom` wins outright and never touches
   `build_story_contract()` (called independently at lines 3340-3344). My
   anchor review flagged this as an assumption; Codex sharpened it to a
   MUST-FIX with exact lines. REVISED SECTION 1: the plan must state
   explicitly whether `style_custom` free text (a) also becomes a
   `StoryContract` override (custom label, catalog-default grammar) or
   (b) is retired from this cleanbreak as a second bypass. Silence on this
   is not acceptable for a "one engine" plan.

5. **Ledger field overload confirmed.** `meta["gen_params_initial"]` stamps
   `style`, `style_combo`, `style_custom`, `style_source` as siblings
   (`OTR_LedgerScriptWriter.py:5505-5508`, read directly). The freeze
   validator's snake_case check on `gen_params_initial.style` (flagged by
   Codex at `_otr_ledger_freeze.py:595-614`, consistent with
   `tests/test_style_palette_drift.py`'s malformed-slug tests seen
   earlier) means the plan must pick ONE canonical field for the slug and
   not let a spaced/hyphenated catalog label collide with the
   snake_case-only invariant. Fold into section 5 as a MUST-FIX, not an
   optional rename.

## Also accepted from Claude's own anchor review (unchanged after grounding)

- Sec 6 resolution reasoning (my anchor's reading (a): a toggle that
  degrades to a simpler single-engine state is not a forbidden fallback)
  still holds, but now applies concretely to `story_scaffold`, per #1 above
  -- superseded in specificity, not in substance.
- `_resolve_style_rng_seed`/`picker_rng` (BUG-LOCAL-270) needs a
  before-deletion grep for other callers -- unchanged, still open.
- `_otr_ledger_consistency.py`'s existing `contract.slug` matrix row is
  unaffected by this rip -- unchanged, still holds.

## Rejected / not folded

- Nothing from Codex was misread or hallucinated -- every MUST-FIX cited a
  real file:line and checked out on direct read. Full marks this round.
- Antigravity: no claims to ground (no output produced).

## Convergence statement

r1 is NOT converged -- this round overturned the plan's costliest line item
(the MusicGen palette) and surfaced a real, previously-invisible third UI
control (`story_scaffold`) that changes section 6 from a hypothetical to a
concrete, already-shipped decision point. Carry all five accepted items
into r2 (coding plan). Retry antigravity on r2; if it fails twice, proceed
r3/r4 on codex + Claude anchor alone and say so plainly in the final
delivery rather than presenting a two-thirds panel as three-thirds.
