# R1 judgment (Claude, judge)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro, Grok-4.3 (~$0.056). Convergence: STRONG on two structural
defects in v0; both ACCEPTED and folded into pass01_plan.md.

ACCEPTED (folded):
- L1 "regenerate beat intent on exceed" = a disguised reroll gate (GPT/Gemini/Grok, unanimous). Recast L1 as a
  Python-filled required `conflict_object`/`conflict_type` slot + deterministic substitution; no retry.
- L2 "required slots" are labels unless mechanically filled (Gemini/DeepSeek/GPT). Recast with injected content
  + deterministic fallback templates + carry the beat_role/conflict tag into the composer.
- L1+L2 must ship TOGETHER (DeepSeek: vocab-only change leaves the structure; structure-only leaves the words).
- Split L5 -> L5a (critic-abort + telemetry fix, safe, early, enables measurement) and L5b (gemma default,
  gated on a bake-off, not leading) (GPT/Gemini/DeepSeek/Grok).
- CUT L6 best-of-N from v0 (all 4) -- kept on record as operator-asked, deferred with rationale.
- L3 redesigned as delimiter+regex (not strict JSON; weak models break schemas -- Gemini), sequenced after
  L1/L2 (DeepSeek), with named flag + meta placement (GPT/Grok).
- L4 minimal; mojibake -> verify-only (GPT/DeepSeek).
- Acceptance metric = cross-episode sameness, with L5a (telemetry) as a prerequisite (GPT).
- Ship code excerpts as grounding for R2/R3 (GPT #7, Grok #1: panel couldn't verify code claims from the plan
  alone -- correct process gap).

VERIFY-AT-BUILD (downgraded from assertions):
- `allowed_things` carries real conflict objects vs static nouns (GATES L1) -- checking now.
- gemma `too_many_edits` root cause (formatting instability?) before L5b.
- unknown `meta`/`compose_flags` keys are ignored by all downstream consumers.

REJECTED / not adopted: none outright -- the panel's pushback was all valid. (GPT wanted "commit+push" detail
cut from the design doc -- kept as a one-liner in build order, harmless.)

Convergence call: NOT converged (expected at R1). The arc is sound; R2 must pin the implementability of the
deterministic slot-fill (L1/L2) and the composer delimiter (L3).
