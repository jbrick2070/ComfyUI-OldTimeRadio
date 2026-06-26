# R1 judgment (Claude, sole judge) -- story-ledger-integrity (DRIFT focus)

Panel: gpt-5.5 (no), gemini-3.1-pro (yes-with-fixes), deepseek-v4-pro (no). R1
spend ~$0.15. Operator steer mid-round: FOCUS ON DRIFT, Claude is the
code-grounded judge -> synthesis weights drift fixes, demotes whole-story-craft.

## ACCEPTED (grounded true, folded into pass01_plan.md)
- **Critic fail-open (all 3 + anchor) -- CONFIRMED** (`clean()` ~189-197; exhaust
  ~445-455). Folded: `ArcVerdict="unverified"` + `meta.story_critic_status` stamp;
  freeze maps unverified -> non-clean. + the A3-floor downgrade (Gemini, ~567-590).
- **Deterministic cross-stage parity test (all 3 + anchor) -- the core.** Folded the
  source-of-truth MATRIX (gpt #3) + the reflect-the-pydantic-models parity test
  (gemini #3) `test_ledger_canon_parity.py`. PURE, offline, CI.
- **Critic is NOT whole-story (gpt #4) -- CONFIRMED** (`_critic_character_lines`
  line ~394 filters `speaker_role=="character"`). Folded: read-only context for all
  story-bearing lines, reroll targets stay character-only.
- **Doctor-edit drift (gemini #1) -- grounded plausible** (doctor runs before
  critic). Folded: pass outline `beat_intent` into the critic prompt.
- **Freeze warn taxonomy (gpt #2/#6, deepseek #5).** Folded: structural blocks /
  accuracy-warn ships-non-clean / cosmetic clean-with-warns; stop calling a shipped
  arc failure "structural."
- **CI drift guards (all 3).** Folded: OTR_WorkflowValidator as a STANDING CI test
  (BUG-LOCAL-097) + schema-version migration/compat with vintage-ledger fixtures.

## CUT (unanimous, folded)
- **Multi-LLM voting for binary gates (gpt CUT-1, deepseek CUT-E).** The honest
  answer to the operator's "multiple LLMs on binary decisions": guards must be
  DETERMINISTIC; LLMs advisory only. Folded as a governing principle.
- **`StanceIssue` telemetry-only (all 3) -- CONFIRMED** ("TELEMETRY ONLY / dead-end"
  in code). Folded: delete it.
- **Over-built positive-evidence LLM engine (deepseek CUT-A).** Folded: the
  deterministic status stamp + parity test suffices.

## VERIFY-AT-BUILD (coder; grounding, not another panel)
1. contract+outline+CastLock in scope at the freeze moment + their schemas
   importable into a CI test (deepseek/gemini [ASSUMPTION]). 2. exact
   `OTR_WorkflowValidator` name + whether a suite test already calls it. 3. whether
   adding "unverified" to `ArcVerdict` ripples any exhaustive match.

## CONVERGENCE
R1 converged the drift architecture (deterministic guards; critic status not
"strong"; the parity test as the core; CI drift guards; warn taxonomy; cuts). The
operator wants to get through the little stuff, so: ONE more round (R2 coding plan)
to lock the parity-test design + concrete signatures + the [ASSUMPTION] checks,
then converge (this is test + status-stamp + enum only -- the R3 workflow-JSON
wiring risk is ABSENT, like leaking-words).
