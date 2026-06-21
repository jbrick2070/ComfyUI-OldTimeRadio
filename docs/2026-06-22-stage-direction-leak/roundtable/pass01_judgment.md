# pass01 judgment (ARCHITECTURE) -- Claude = judge

Panel: my grounded critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend $0.11.

## ACCEPTED (grounded CONFIRMED)
- Verb-list-led destructive strip is UNSAFE; false-positive counterexamples real
  ("looks can be deceiving, John.", "pauses are evidence, Brain", "look, Pinky",
  "glances at Pinky We..."). -> structural disambiguation + narrow scope. (all)
- Reroll MUST detect on the RAW draft before any hygiene scrub, or the signal is
  destroyed and the gate bypassed. -> L3 primary, raw-draft detection. (GPT/Gemini/DeepSeek)
- Freeze normalizer is the only bypass-proof choke point; both call sites share
  ONE helper. (me/GPT/DeepSeek)
- `scrub_leading_stage_direction` must NOT take `speaker_name` -- grounded:
  `_strip_stage_directions(text)` has no speaker. (GPT/DeepSeek; corrects pass00)
- Audio invariant reframe: clean inputs byte-identical; contaminated lines change
  intentionally; fixture is clean indextts2 -> scrub is a no-op. Add the no-op test. (GPT)
- `< 2 words` guard must allow terminal-punctuated short utterances ("sighs No."). (GPT)
- Boundary must skip capitalized OBJECTS after prepositions/possessives; a capital
  after ", " is a vocative not a boundary. (GPT/Gemini)
- L4 prompt = defense-in-depth, tests pass with prompt ignored; add a POSITIVE
  constraint. (GPT/Gemini/DeepSeek)
- 3681 music patch: grounded INSIDE the NON_VOICED branch -> `(beat.sfx_cue or "")`
  keeps sfx render-contract, drops intent. (me, corrects GPT's "global" worry)
- Document the all-lowercase limitation; idempotency test; replay b004/b006/b007;
  coverage test via scrub_ledger only. (DeepSeek/GPT)
- Add a non-mutating measurement script to validate precision BEFORE the
  destructive floor ships. (GPT/DeepSeek)

## JUDGED SPLIT (the one real disagreement)
Gemini: CUT the destructive scrub entirely (undelimited text can't be safely
regex'd) -> reroll+prompt only. GPT/DeepSeek: keep a NARROW structural destructive
floor. RULING: keep the narrow guarded floor (freeze is the last guarantee; a weak
model can exhaust reroll), BUT (a) it only fires on the high-confidence structural
pattern, (b) when uncertain it does NOT strip, (c) a PRECISION GATE (the scan
script over the real corpus must show ~zero false positives) must pass before it
ships broadly -- else fall back to Gemini's reroll-only. This bounds Gemini's risk
while keeping GPT/DeepSeek's guarantee.

## REJECTED / DOWNGRADED
- DeepSeek "no commas in the lowercase clause" disambiguator: REJECTED -- b006
  "pauses, sets pen down" HAS a comma; the comma rule would miss it. Replaced by
  the second-token copula/modal test + the ", "-precedes-capital vocative rule.
- Broad `-s/-ing/-ed` morphology for the DESTRUCTIVE scrub: cut (false positives);
  fine for the BROAD reroll detector only. (GPT/DeepSeek)

## OPEN -> next passes
- pass02 (coding): the exact structural algorithm, token boundary, guard regexes,
  the full counterexample test corpus, idempotency.
- pass03 (wiring): the precise reroll seam (who sets reroll_hint in the spine; R2
  located it at the critic-flag -> compose_line_draft path; confirm), the
  _strip_stage_directions import/call, the 3681 edit, the scan script.
