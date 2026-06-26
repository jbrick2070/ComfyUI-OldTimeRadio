# R2 judgment (Claude, sole judge) + campaign convergence -- leaking-words

Panel: gpt-5.5 / gemini-3.1-pro / deepseek-v4-pro, all "no" (coding plan
underspecified). R2 spend ~$0.27; campaign total ~$0.42.

## ACCEPTED (grounded, folded into pass02_plan.md)
- **Cut Layer 3 (LLM cleaner) from v1 -- UNANIMOUS** (gpt CUT-1, gemini CUT-1,
  deepseek implicit). Grounded: `compose_line` returns raw stripped text that
  mangles JSON; a real cleaner would need `_otr_structured_call`. Deferred.
- **Concrete API + transient data model** (gpt #2/#9, deepseek #4/#5). Schema is
  frozen, so `EntityPolicy`/`VerificationResult` are transient dataclasses with
  span-bearing defects. Folded.
- **News-bleed fixed AT `build_allowed_roster`** (all three + anchor) -- CONFIRMED
  the key_terms->allowed_roster merge (line ~368-370) is what allowlists "Trump".
  Folded: add `banned_terms` param + split real-person entities out of key_terms.
- **Caps-vocative: full-name PHRASE matcher, DROP (not title-case), wired after
  cast_strip before detect_phantom_names; negative fixture = first-name vocative**
  (gpt #6/#7/#8, gemini SHOULD-1, deepseek #2). Folded.
- **Malformed = DOUBLE quotes only via segment_double_quotes** (gpt #5, gemini
  SHOULD-2). Folded (apostrophes must not trip it).
- **Shared `_leak_repair_attempted` budget** (gpt #3) -- grounded multiple existing
  budgets; folded a single shared guard.
- **Mandatory-vs-dark contradiction** (gpt #1) -- resolved: mandatory under
  STRICT/CI + post-promotion; dark only for the validation window.
- **Layer-1 prompt: "...unless listed under NAMED ENTITIES"** (gemini #3) -- folded
  (bare "no real names" contradicts the NASA/CERN key_terms injection).
- **Stage-direction extraction via segment_double_quotes with explicit ordering**
  (gpt #4, deepseek #1) -- folded; the leading-quote requirement guards
  `"Running," she said,...`.

## REJECTED / not folded
- gpt SHOULD-1 "extend to Breathless,/Still gasping,/Gasping for air,": kept v1
  NARROW (single capitalised participle) + a test asserting only that class fires;
  widening now re-opens the false-positive surface the operator is tired of. Note
  it as a future fixture if a new variant ships.
- deepseek "define both lanes": folded as a one-liner (local + frontier writer
  paths) rather than a structural change.

## VERIFY-AT-BUILD (coder tickets -- need live code grounding, not another pass)
1. writer order compose->scrub->freeze(`scrub_ledger`)->TTS (verifier upstream of
   audio). 2. whether `compose_line`'s strip pipeline already calls the scrubs
   (wire at writer level if not). 3. the phantom/roster gate reject action.

## CONVERGENCE CALL -- STOP at R2 (do not grind R3/R4)
R1 converged the architecture; R2 converged the build-ready coding plan. R3's
standing focus is workflow-JSON / node / widget WIRING + re-baseline -- but this
change is CONTENT-ONLY: no new node, no widget, no `widgets_values` shift, no
`otr_scifi_16gb_full.json` edit (Layer 3, the only thing that might have needed
plumbing, is CUT). So the R3 risk surface is ABSENT. The only residual is internal
Python hook-point precision, which is enumerated above as coder verify-at-build and
requires LIVE grounding (the coder window's job per the planner/coder split), not a
4th OpenRouter opinion. Re-looping would be "grinding passes to hear looks-good in
more accents" -- explicitly against the standing roundtable rule. Campaign closed
at R2; ~$0.42 total. Deliver pass02_plan.md as the build spec + a coder kickoff.
