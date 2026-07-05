ROUND 1 -- CLAUDE ANCHOR REVIEW (code-grounded, written before seeing the panel)

Grounded against: nodes/OTR_LedgerScriptWriter.py (lines 255-324, 780-820,
1270-1400, 2819, 2960-3010, 3300-3420, 3560-3640, 5930-5945, 6060-6075),
nodes/_otr_style_catalog.py (full file), nodes/_otr_style_picker.py
(header/contract), nodes/_otr_config.py (style_grammar_enabled),
tests/test_style_palette_drift.py, tests/test_workflow_json_guardrails.py,
tests/test_otr_api_companions.py, nodes/_otr_ledger_consistency.py
(story_contract matrix row), BUG_LOG.md (BUG-LOCAL-216/240/270).

- VERDICT: yes-with-fixes. The core diagnosis (one engine should drive both
  tone label and climax shape) is correct and grounded in the real code, but
  the plan currently under-specifies the single highest-cost item (MusicGen
  cue authoring) and leaves one architectural fork (section 6) genuinely
  open rather than resolved -- both must close before this is build-ready.

MUST-FIX BEFORE BUILD:
1. [Sec 0 / Sec 6] "No fallback" is asserted as absolute but Section 6 then
   presents it as a two-reading open question and defers to the panel. A
   plan cannot simultaneously declare a hard constraint and leave that same
   constraint's scope undecided. Fix: pick one reading before build starts.
   Recommendation grounded in the code: `_otr_config.STYLE_GRAMMAR_DEFAULT`
   already defaults True and existed as a de-risking lever for a *dual*-
   system state (grammar-on vs grammar-off-with-old-default-climax). Once
   there is only one engine, "off" no longer reverts to a different
   selection mechanism -- it just disables grammar injection and leaves
   `_climax_role` at `BEAT_ROLE_IRREVERSIBLE_CHOICE` (confirmed at
   OTR_LedgerScriptWriter.py:3574). That is reading (a): keep the lever as a
   feature toggle, not a fallback. It survives "no fallback" because there
   is nothing left to fall BACK to -- it degrades forward to a simpler
   single-engine state, not sideways to a second engine.
2. [Sec 3] The plan flags "author ~90 cue triples" as the likely largest
   line item but does not size it, staff it, or gate the sprint's start on
   it explicitly enough. Per Sec 7 step 1 it IS first in sequence, but Sec 0
   ("no fallback") and Sec 7 both need one unambiguous rule: the code
   changes in Sec 1/2 MUST NOT merge until all 100 palette entries exist and
   `test_style_palette_drift.py` passes against the full catalog set. As
   written, an implementer could plausibly land Sec 1/2 first and treat Sec
   3 as a fast-follow "because it's just content" -- which is exactly the
   smuggled-back fallback the plan's own Sec 9 Q3 worries about. State this
   as a hard merge-gate, not a sequencing preference.
3. [Sec 5] `_otr_ledger_consistency.py`'s `_CONSISTENCY_MATRIX` already has a
   `MatrixRow("style", "contract.slug", "ledger.meta.story_contract.slug")`
   entry (line 68) -- confirmed live in the file. The plan's Sec 5 says to
   "audit... before deleting" `meta.style_pick`, but doesn't confirm this
   consistency-matrix row is UNAFFECTED by the rip (it reads `contract.slug`
   /`meta.story_contract`, not `meta.style_pick`, so it should survive
   untouched) -- the plan should say so explicitly rather than leaving it as
   an open audit item, since grounding already answers it.

SHOULD-FIX:
1. [Sec 1] The plan proposes the combo enumerate all 100 catalog labels.
   ComfyUI combo widgets render as a literal dropdown list; 100 entries in
   one flat list is a real UX regression from the current 10-item list (the
   very complaint that started this thread was "why is the dropdown so
   small," but a 100-row unsearchable flat combo is arguably a worse
   surface, not a better one, even though the screenshot's search-filter
   combo UI can filter/type-to-search). Verify: does ComfyUI's combo widget
   here render with the searchable filter shown in the original screenshot
   (it does -- the screenshot shows a search box + scrollable list), which
   would make 100 entries tolerable. Flag as verify-at-build, not a blocker.
2. [Sec 2] `_resolve_style_rng_seed`/`picker_rng` (used for the deleted
   picker's seed-flavor sampling, BUG-LOCAL-270's fix target) should be
   grepped for any OTHER caller before deletion -- the plan already says
   this ("if nothing else calls them") but should also name the specific
   regression risk: BUG-LOCAL-270 fixed a real production bug in this exact
   RNG helper five weeks ago; deleting it without confirming zero other
   callers risks silently resurrecting a seed-pinning bug if some other
   consumer was leaning on it.

OPTIONAL / NICE-TO-HAVE:
- Consider whether `style_custom` should also accept a catalog slug
  shorthand (e.g. typing "noir_interrogation_chamber") that resolves through
  `get_style()`'s normalization, so power users don't have to scroll a
  100-row combo. Not required by the operator's directive, just a UX idea.

CUT THESE (scope / over-engineering):
1. Nothing in the plan reads as scope creep relative to the operator's
   explicit "one engine" directive -- the plan's scope (widget, resolve_
   inputs, dead-code deletion, palette, tests, workflow JSON) is the
   minimum closure set for the stated goal. No cuts recommended at the arc
   level.

[ASSUMPTION] The plan assumes the 100-catalog's existing `select_style()`
determinism (sha256(cast_seed)-keyed) is adequate as the ONLY draw mechanism
going forward, including for the user's explicit hard-pick case (picking a
specific catalog entry bypasses the hash draw entirely and just uses that
slug's precomputed grammar) -- this is consistent with today's `get_style()`
lookup path and requires no new design, but the plan should say so plainly
rather than leaving it implicit in Sec 1's prose.
