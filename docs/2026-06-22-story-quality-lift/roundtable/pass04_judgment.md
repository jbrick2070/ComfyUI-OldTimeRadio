# R4 JUDGMENT (convergence) -- CONVERGED + total spend

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.1097. **All three R4 verdicts = "yes-with-fixes"
-> CONVERGED** (no "no", no new architecture). R4 fixes are spec-locks, all folded into pass04_plan_FINAL.

## ACCEPTED (folded into pass04_plan_FINAL)
- **Pronoun-guard clarity** (GPT mf#2, Gemini mf#1) -- VERIFIED `_PRONOUN_ROOTS` is 1st/2nd-person ONLY, so
  "clutches her"/"taps his" already strip; spec made explicit (3rd-person permitted) + b010/b012 asserts.
- **Well-formedness vs closing quote** (GPT mf#1) -- "last SPOKEN char before the optional closing `"` in
  `_TERMINAL_PUNCT`" (else a strict final-char check aborts b005 and fails acceptance).
- **Quote helper normalizes curly->straight before counting + returns normalized text** (DeepSeek mf#1) --
  else curly-only counting makes b005/b010/b012 odd-quote -> unscrubbed. Shared by Tier 2 + Tier 3.
- **Tier-2 control flow locked** (GPT mf#3, DeepSeek sf#3) -- disable the old compose_line block, single
  guard into compose_line_draft, hint appended in `_BARE_STAGE_HINT` format, test: <=1 reroll per line.
- **StanceIssue schema locked + stance is telemetry-only** (GPT mf#4/#5/#6/cut#1, Gemini sf#1/#2, DeepSeek
  opt) -- `target: str` free-form, `missing_turn_beat: str`; do NOT add "stance" to FailedDimension in v1;
  do NOT convert to a RerollTarget; no-reroll test added.
- **Pre-freeze sweep promoted to MANDATORY** (GPT mf#7, DeepSeek mf#2, Gemini mf#2) -- final step of the
  cascade mutation phase after cast_lock, before freeze hash; `cast_ids = ledger.cast.keys() - sentinels`.
- **Explicit closed `_NARRATION_VERBS` extension** (GPT cut#3, DeepSeek opt) -- no "obvious neighbors".
- **DEFECT-1 byte-identical golden no-op gate** (anchor) ; **acceptance honesty re v1 leak limitation**
  (GPT sf#3); **DEFECT 4 explicitly out-of-scope** (GPT sf#4); **beat-lever references character_b_wants**
  (anchor). All folded. VAB checklist rewritten as concrete steps (GPT, Gemini, DeepSeek all converged on
  the same 4-7 items).

## REJECTED / CORRECTED
- No panel claim rejected as a code misread this round -- R4 claims matched the grounding. The "fix-introduced
  regression" worries (pronoun guard) were SPEC-CLARITY issues, not actual code bugs (the guard is
  1st/2nd-person only); clarified rather than changed.

## VERIFY-AT-BUILD (carried, concrete) -- see pass04_plan_FINAL sec 7
sweep line-position vs cast_lock; compose_flags no strict validation; StanceIssue pydantic round-trip;
OTR_TEST_MODE gate; quote helper straight/curly + idempotence; strip_line_formatting quote behavior;
stance-no-reroll.

## CONVERGENCE + SPEND
CONVERGED at R4 -- locking `pass04_plan_FINAL.md` as the build-ready coder kickoff. The 4-round arc closed:
R1 arc/altitude -> R2 codeable algorithm -> R3 wiring reality (cut the unbuildable auto-repair) -> R4
spec-lock. **Total OpenRouter spend R1-R4 (counted passes): ~$0.4235** (R1 $0.0828 + R2 $0.1257 + R3
$0.1053 + R4 $0.1097); plus one discarded ungrounded R2 run (~$0.05-0.08, killed + relaunched grounded) ->
campaign total ~$0.48-0.50. Raw reviews: `docs/2026-06-22-story-quality-lift/roundtable/pass0N/`.
