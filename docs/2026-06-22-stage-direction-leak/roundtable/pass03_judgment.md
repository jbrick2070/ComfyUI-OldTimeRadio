# pass03 judgment (WIRING) -- Claude = judge. CONVERGED.

Panel: my grounded critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend $0.19.

## ACCEPTED (grounded) -- the wiring decisions
- **Destructive strip is FREEZE-ONLY** (`_strip_stage_directions`), NOT in
  `clean_spoken_character_line`. Grounded: spine Stage 3.7 hygiene only bumps
  `meta["delivery_hygiene_report"]` and does NOT emit a ScrubFinding, so stripping
  there loses the `CODE_STAGE_DIRECTION` finding. This also removes the
  parity-across-call-sites + double-fire complexity. (GPT, decisive)
- **The DETERMINISTIC detector drives reroll, not the LLM critic** -- the failure
  mode is a weak model the critic won't catch. Detector runs in `compose_line` on
  the raw candidate before hygiene; hint CONCATENATED with any existing critic hint
  (`f"{existing}; {stage_hint}"`). (me/GPT/Gemini/DeepSeek)
- **CONTRADICTION FIXED (GPT):** my pass02 corpus listed "looks at Pinky and Brain
  We need a plan." as KEEP, but the rule would STRIP it. Resolved by ABORTING on a
  conjunction in the object chain (conservative keep) -- a single preposition-object
  skip is still allowed.
- Empty-string IndexError guard (guard 0) after a prior delimited strip. (Gemini)
- Object-skip set must include ARTICLES {the,a,an} + possessive adjectives
  {his,her,their,...} + more prepositions {in,inside,behind,past,out,...} -- "looks
  at the Map We...", "looks in Box We...". (Gemini)
- Pronoun guard matches contraction ROOTS (we've/you'll/i'm). (GPT)
- Preserve the `Tuple[str,bool]` return; `out` = post-delimited text; emit
  `CODE_STAGE_DIRECTION` with the stripped prefix; restamp `word_count`. (GPT/DeepSeek/Gemini)
- `_otr_line_hygiene` stays PURE -- cut the `_propose_*` helper; the freeze caller
  compares old!=new to log/emit. (DeepSeek CUT)
- `detect_leading_stage_business -> (bool, reason)`; reason is the hint string. (DeepSeek)
- Build toggle = a source-level `BARE_STAGE_FLOOR_ACTIVE` module constant (NOT
  JSON), set by the precision-gate outcome; split always-on detector/proposer tests
  from mutation tests. (GPT)
- Scan script `scripts/stage_direction_scan.py`: frozen ledgers in, JSONL out with
  stable fields; "false positive" = a `would_mutate` strip on a line that should be
  kept, judged by inspection. (GPT/DeepSeek)
- Exhaustion: accept last draft; the freeze floor cleans the cases it CAN; residual
  all-lowercase/>6-word leaks are an accepted, logged risk (NOT "cleans whatever
  shipped" -- that was overstated). (GPT)
- 3681 identified by stable code context, not the brittle line number. (GPT/DeepSeek)

## REJECTED / corrected
- Gemini SHOULD-FIX: narrow guard (c) to subjective pronouns {i,we} only. REJECTED
  for the DESTRUCTIVE floor -- the broad pronoun abort is the SAFER (precision-over-
  recall) choice; "points at me We..." simply falls to reroll. (judge)
- Gemini pass02 "add and/or to the skip list to STRIP compound objects": REJECTED in
  favor of GPT's conjunction-ABORT (keep) -- conservative for a destructive op.

## CONVERGENCE
Pass03 produced only build-level precision items (exact constants, guard edges,
contract wording) -- no new architecture. The 3-pass campaign (architecture ->
coding -> wiring) has CONVERGED. `pass03_plan.md` is the sprint-ready build plan;
promoted to `docs/2026-06-22-stage-direction-leak/SPRINT_PLAN.md`. Build is
operator-gated.
