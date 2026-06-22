# R1 JUDGMENT (arc/creative) -- accepted / rejected / verify-at-build

Panel: GPT-5.5-20260423, Gemini-3.1-pro-preview-20260219, DeepSeek-v4-pro-20260423. Spend ~$0.0828.

## ACCEPTED (folded into pass01_plan)
- **DEFECT 4 gate CUT** -- unanimous (GPT cut#1, Gemini cut#1, DeepSeek cut, anchor cut#1). Telemetry-only, out of story-lift acceptance.
- **DEFECT 2 is arc-level, not line reroll** -- unanimous (GPT mf#5 w/ "arc repaired" acceptance, Gemini mf#1 "line reroll will thrash", DeepSeek mf#2, anchor mf#2). Detection=critic axis; repair above the line.
- **DEFECT 1 tiered** -- converged (GPT mf#3 tiers, Gemini mf#2 floor=quote-boundary-only, DeepSeek mf#1, anchor mf#1 generation-primary). Generation -> reroll -> conservative deterministic floor; exclude undelimited (b017) from floor; negative fixtures required.
- **DEFECT 3 = COERCE not crash** -- Gemini mf#3 (grounded correction of my anchor's "consistency assert"): a raising assert halts the pipeline on a weak model. Coerce at write points + role_mismatch guard; CI assert only; meta audit of role changes (GPT mf#6/sf#4, DeepSeek sf#1).
- **Measurable lift target** (GPT mf#1), **no-op strong-model fixture** (GPT mf#2, DeepSeek mf#4, anchor sf#3), **three audio acceptance lanes** (GPT mf#4, DeepSeek mf#3), **caught->repaired->absent-from-frozen + reroll-exhaustion behavior** (GPT mf#8), **no-bypass BASELINE re-smoke FIRST** (GPT sf#6, DeepSeek sf#3), **one-gate-path principle** (GPT sf#1), **JSON hash no-drift** (GPT sf#5), **well-formed stripped line** (anchor sf#2), **negative fixtures** (GPT sf#2). All folded.

## REJECTED / CORRECTED (judge, grounded)
- **Gemini mf#2 "strip all text outside matched quotes" (deterministic floor primitive) -- TOO BLUNT.** Grounding the real corpus: b015 = `Well, Manfred, ... expected." tightens her scarf ... "I do hope ...` -- the spoken text "Well, Manfred ... expected." sits OUTSIDE the quotes (malformed quoting). Blind-stripping extra-quote text would delete legitimate dialogue. CORRECTION (in pass01 sec 2 Tier 3): the floor must CLASSIFY the outside-quote span as 3rd-person physical action, not strip all extra-quote text; b015 + b017 go to reroll, not the floor.
- **GPT + DeepSeek VERDICT "no" -- not treated as a defect.** pass00 is a problem statement that explicitly lists open questions; "not build-ready" is its expected maturity, and the campaign's job (R2-R4) is to close those. Their MUST-FIXes are the path to build-ready and were folded; the verdict itself is noted, not actioned as a code defect.
- **GPT mf#5 "line-scoped stance_coherence critic axis routed through scoped reroll" -- partially rejected.** The DETECTION-as-critic-axis is accepted, but routing the REPAIR through the line-scoped reroll contradicts the unanimous "arc-level not line-level" finding (Gemini mf#1). Repair routed above the line instead (pass01 sec 3); mechanism finalized R2.

## VERIFY-AT-BUILD (downgraded from assertions)
- Exact origin of the b011 announcer stamp (outline beat vs role_mismatch repair) -- trace via the meta audit. [GPT sf#4, anchor ASSUMPTION]
- DEFECT 2 repair on a weak model: can mistral execute a coherent through-line on an episode rerun, and is `needs_full_rerun` deterministic + affordable? [Gemini ASSUMPTION, DeepSeek ASSUMPTION] -- decide R2/R3.
- The no-bypass baseline re-smoke must actually reproduce the four defects under normal halt (the source smoke was bypass-on). [GPT sf#6 ASSUMPTION]
- cast names c02=Manfred / c03=Mali / c04=skeptic inferred from text; confirm vs ledger `cast`. [anchor ASSUMPTION]

## CONVERGENCE CALL
R1 CONVERGED on SCOPE + ALTITUDE: DEFECT 4 gate cut; DEFECT 1 tiered (generation-first, conservative
floor); DEFECT 2 arc-level (detect=critic axis, repair above line); DEFECT 3 coerce-not-crash. The
remaining splits are mechanism-level (DEFECT 1 detection primitive; DEFECT 2 repair path) -- exactly
the R2 (coding plan) remit. Advance to R2.
