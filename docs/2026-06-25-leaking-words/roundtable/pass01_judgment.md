# R1 judgment (Claude, sole judge) -- leaking-words

Panel: gpt-5.5 (no), gemini-3.1-pro (no), deepseek-v4-pro (no). All three +
my anchor agreed the doc was a MENU and must commit -> done (pass01_plan.md has a
CHOSEN ARCHITECTURE). Spend this round ~$0.15 ($0.058 first pass + $0.092 the
GPT+DeepSeek reasoning-fix re-run).

## ACCEPTED (grounded true, folded in)
- **Root-cause correction (deepseek #2, gpt #4) -- CONFIRMED against code.**
  `_leading_stage_strip` guards on `body[0].islower()` (line 271) and never uses
  `_NARRATION_VERBS`; the `Gasping,` miss is the CAPITALISED-lead guard, not a
  whitelist gap. My anchor + the problem doc were wrong; corrected. Fix re-pointed
  to a narrow capitalised-participle+quote rule.
- **Mandatory deterministic verifier as THE correctness layer; A + D demoted
  (gpt #1/#7, gemini #1, deepseek #3).** CONFIRMED A/D can't be the baseline under
  the offline/deterministic/agnostic invariants. Folded: Layer 2 mandatory
  deterministic; A = optional typed-repair; D = product tier.
- **Narrow structural extract-or-fail, NOT broad `-ing` scrubbing (gpt #4/CUT-2).**
  Folded -- my anchor's "relax the detector" was sharpened to shape-specific rules
  + the negative-fixture FP guard.
- **Typed-repair contract for A (gpt #2).** Folded verbatim (JSON clean_text/
  removed_spans/reason_codes/confidence; reject over-diff; skip clean lines).
- **News needs a per-episode entity POLICY + news-abstraction (gpt #3/#10).**
  Folded as the Layer-2 news-bleed mechanism; the cheap proper-noun guard from my
  anchor is the floor, the abstraction policy is what makes it real.
- **Malformed internal-quote class currently ships (gpt #6) -- CONFIRMED.**
  `sanitize_transcript_text` balances only a single edge wrapper (line 886-ish);
  added fail-closed-to-recompose for internal odd quotes.
- **Acceptance corpus + negative fixtures (gpt #6/eval, deepseek optional).**
  Folded as the acceptance gate.
- **Audio-affecting => ship dark (gpt #5/#7) -- CONFIRMED** via `_otr_config.py`
  95/107. Folded as the placement/rollout discipline.

## VERIFY-AT-BUILD (downgraded; not blocking R1)
- `scrub_self_vocative` coverage + whether `scrub_ledger` calls it (gpt #5). R3.
- `build_allowed_roster` must not whitelist news/key terms (gpt #10). R2/R3.
- The exact freeze->TTS ordering so the verifier sits upstream of audio (gpt #7).

## REJECTED / not folded
- **Gemini "cut A entirely + cut D, move news to C+D" -- PARTIAL REJECT.** Cutting
  A as the BASELINE is right (done); cutting it entirely is too strong -- a scoped,
  opt-in, fail-open typed-repair does not violate the invariants (it reuses
  existing plumbing, default-off, offline floor still runs). Moving news-bleed to
  "C+D" is wrong: prompt+frontier do not DETERMINISTICALLY stop a local leak; news
  needs the Layer-2 policy. Kept A scoped; kept news in Layer 2.
- **gpt "constrained generation where transport supports it" (build seq step 2).**
  Rejected -- same reason B is cut; tiering by transport reintroduces the
  portability trap for marginal gain.

## CONVERGENCE
R1 converged on the architecture (correctness = deterministic Layer 2; A/D
optional above; B cut). The only material new thing R1 surfaced was the
root-cause correction, now folded. Advance to R2 (coding plan: concrete functions,
the news-abstraction policy shape, sequencing vs the freeze cascade).
