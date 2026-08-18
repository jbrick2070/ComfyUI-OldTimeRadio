# r4 judgment -- item F convergence

**Driver:** Claude (Cowork), panelist and sole judge. **Date:** 2026-08-17.
**Reviewed against the FINISHED, PUSHED diff** (`87dee50d`), not a plan.

**Lanes:** `Gemini 3.1 Pro (High)` (`2026-08-17-item-F-r4-pro31/r4/`) and
`Gemini 3.7 Flash (High)` (`2026-08-17-item-F-r4-flash37/r4/`), both with
`KIBITZ_AGY_PRINT_TIMEOUT=15m`, both first try.

## CONVERGED -- zero must-fix from either lane

* **Pro:** *"yes. The plan has converged; the build-breaker was fixed properly
  via lane-gating."* No must-fix, no should-fix, no optional.
* **Flash:** *"build-ready as-is: yes ... the media_archive collision is properly
  gated at the source kind."* No must-fix.

Both independently confirmed the r3 build-breaker fix is correct at the source
kind rather than patched at the call sites, and both independently named the
LIVE LEG as the one thing static work cannot settle -- which is the driver's own
standing position, now arrived at from three directions.

## THE ONE SHOULD-FIX, ADOPTED

Flash: `ADAPTATION_SOURCE_KINDS` was exported for cross-module use but omitted
from `__all__`. Correct and taken -- the module HAS an `__all__`, so a public
constant missing from it is a real inconsistency rather than a style note.
Added.

## THE ONE OPTIONAL, DEFERRED WITH A REASON

Flash suggests parametrizing the composed-frame test over a `public_domain`
title as well as the three Shakespeare ones. Sound in principle and **not taken
here, because the fixture would be dishonest**: that test builds its foreign-term
corpus from the SHAKESPEARE manifest and its vendored scene texts. A
public_domain title has no equivalent manifest in that file, so the parametrized
case would assert only the trivial half (the WORK line renders) while appearing
to assert the cross-work half. **A test that looks like coverage it does not have
is worse than the gap.** The public_domain seam IS covered -- by
`test_closing_seams_bank_routing` for the seam text and by the bank-parametrized
contract tests for the composer. Recorded here so the gap is deliberate and
findable rather than forgotten.

## WHAT THE ARC ACTUALLY COST AND CAUGHT

Four rounds, both agy lanes throughout (after the r1 collision was recovered),
plus a Fable narrative gate and a Sonnet 5 QA pass. **$0 -- all local.** What it
caught that the driver had wrong:

1. **r1: Shape C was a category error** -- the driver tabled a module as a
   candidate fix because its docstring contained the word "Verona". It is a
   verbatim dialogue slicer that touches no announcer symbol.
2. **r2: an `UnboundLocalError` on four of six banks** -- the driver's own diff
   plan read an identity module that was imported inside a `provenance_normalize`
   branch. Would have killed the majority of episodes and passed any
   adaptation-only test.
3. **r3: the media_archive build-breaker** -- the driver asserted every
   non-adaptation lane yields an empty title; 56 of 98 live ledgers disagree.
   Would have announced "a scene from Now See Hear!" on 57% of a live lane.
   **Sonnet CLEARED this lane; Pro CAUGHT it; the corpus decided.**
4. **Fable: killed an adopted row** that would have rejected the correct setting
   on 5 of 14 scenes, and found the whole-play-promise defect nobody had.

**Three of those four were errors in the DRIVER'S OWN work, and two were
build-breakers.** That is the case for the arc on a design item, stated with
receipts rather than as a principle.

## DISPOSITION

**Item F code is CONVERGED and pushed.** The arc is r1-r4 with two agy lanes, a
Fable gate and a Sonnet QA pass; **Codex never participated** (quota-held to
2026-08-19), so if the operator's definition of a full arc requires the Codex
lane, this is a four-round campaign one reviewer short and must be described that
way rather than as a full arc.

**The only thing still owed is the live leg**, which is running as this is
written. No unit result may be reported as "the wrong-play frame is fixed".
