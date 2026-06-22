ROUND 4 -- CLAUDE ANCHOR REVIEW (convergence / residual defects; grounded)

VERDICT: yes-with-fixes. pass03 is build-ready; no NEW must-fix blocker. The residuals below are
verify-at-build / should-fix that will bite the implementor if unstated -- none reopens the architecture.

MUST-FIX BEFORE BUILD: none new. (The R1-R3 must-fixes are folded; the auto-repair is cut; channels/sites
are corrected.)

SHOULD-FIX (residual, will bite at the keyboard):

1. [sec 2 Tier 2 / Tier 3 quote consistency] The shared double-quote segmentation helper is sound, but
   Tier 2 runs on the RAW draft (model's quotes intact) while Tier 3 runs on the FROZEN text AFTER
   `scrub_ledger` step 3a `strip_line_formatting` (753) has already run (3b `_strip_stage_directions` at
   755). If 3a normalizes/removes the double quotes the floor relies on, the floor's segmentation sees
   different boundaries than the helper was validated against. FIX: state that the quote-anchored bare
   scrub must read the quote structure BEFORE any quote-mutating normalization (run it first inside
   `_strip_stage_directions`, or confirm 3a preserves `"`), and add a fixture that feeds 3a-output into
   the floor.

2. [sec 6 / audio lane b] The DEFECT-1 floor changes frozen text for a leaky line -> if the EXISTING
   byte-identical golden (indextts2) contains any line the new floor now strips, the golden shifts
   silently. FIX: make it an explicit DEFECT-1 gate -- run the new floor over the golden fixture's ledger
   and assert ZERO strips; if it strips, that is the operator-gated recapture trigger, surfaced LOUD (not
   discovered by a red byte-identical test).

3. [sec 2 Tier 2 placement] Moving `detect_stage_business_for_reroll` INTO `compose_line_draft` (1689-1928)
   lands it inside the EXISTING repair ladder (max_attempts=2 + the speaker-self/roster-leak retries). FIX:
   the new detection must COORDINATE with the existing one-reroll guard (`_stage_dir_repair_attempted`) and
   the attempt budget -- do NOT add a third nested LLM retry. Specify: detect on the draft, fold the hint
   into the SAME retry the ladder already performs (one extra attempt max), not a new recursion.

4. [sec 4 generation lever reach] `_build_beat_user_prompt` has only a 1-beat adjacency window
   (previous_beat_intent); Manfred's reversal spans b003->b017. A 1-beat prompt cannot enforce full-arc
   stance consistency alone. FIX: the lever must also REFERENCE the antagonist's pinned want
   (`DramaticState.character_b_wants`, which exists with `_wants_must_oppose`) so each beat is checked
   against the global want, not just the previous beat. Frame the lever HONESTLY as best-effort nudge +
   the critic stance axis as the measurement backstop -- not a guarantee of arc coherence.

OPTIONAL / NICE-TO-HAVE:
- Run the no-op check against a REAL archived frontier (opus) episode ledger from the soak, not only the
  synthetic clean fixture, so "no-op on a good script" is proven on real strong output.
- Tier-1 strengthening should be sampled on a frontier model for dialogue-quality regression (an
  over-aggressive ALL-CAPS constraint can flatten good dialogue); keep the wording change minimal.

CUT THESE: none. pass03 already cut the unbuildable auto-repair + DEFECT 4 gate; nothing left to trim.

[ASSUMPTION] `strip_line_formatting` (3a) quote behavior is the one residual I cannot fully confirm from
the grounding -- listed as SHOULD-FIX #1 verify-at-build rather than asserted.

CONVERGENCE: reached. The four-round arc closed -- arc/altitude (R1), codeable algorithm (R2), wiring
reality (R3), residual scan (R4). Recommend locking pass04 as FINAL and handing the coder kickoff.
