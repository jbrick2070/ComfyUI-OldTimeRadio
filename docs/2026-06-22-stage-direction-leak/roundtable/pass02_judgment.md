# pass02 judgment (CODING) -- Claude = judge

Panel: my grounded critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend $0.21.

## ACCEPTED (grounded)
- Strip "confidence" = a CONJUNCTION of guards (a)-(g); fail any -> return original
  (DeepSeek). Formalized in the plan.
- Boundary must have NO terminal punctuation (.!?) before it -- kills "looks like
  rain. We should go." (GPT). Commas allowed (b006).
- Abort if the leading span contains a 1st/2nd-person pronoun OR a dialogue-starter
  -- kills "maybe we should ask John..." AND "look, Pinky,..." (the dialogue-starter
  "look" guard makes the fragile comma-vocative rule UNNECESSARY). (Gemini; my judge call)
- Object-skip list must include conjunctions (and/or) + many more prepositions
  (by/from/over/under/through/into/onto/...) -- "looks at Pinky and Brain We..."
  (Gemini/DeepSeek).
- Preserve `_strip_stage_directions` `Tuple[str,bool]` contract: delimited THEN
  bare, return `(bare, delimited_changed or bare!=out)` -- a naive replace crashes
  the unpack (Gemini grounded the call site). (GPT/Gemini)
- Keep `_otr_line_hygiene` PURE: no logging inside; an internal proposer returns
  the hit flag, callers log. (GPT)
- Canonical sequence applied in BOTH call sites; PARITY test on the shared HELPER
  (not the whole pipelines -- ledger_scrub also normalizes quotes/dashes, so a
  full-pipeline equality would falsely fail). (DeepSeek + GPT)
- Optional leading quote handled before the lowercase test. (GPT/Gemini)
- Short-utterance exception list, case-insensitive + punctuation-stripped lookup;
  extensible. (GPT/DeepSeek)
- Resolve the precision-gate vs L2-cleans contradiction with an explicit BUILD
  ORDER: detect/propose -> scan -> enable destructive only if ~zero false
  positives, else detect-only. (GPT)
- Telemetry via the existing `CODE_STAGE_DIRECTION`/`ScrubFinding`, NOT new `meta`
  fields. (GPT CUT)
- Reroll EXHAUSTION = accept last draft; L1 freeze floor is the deterministic
  backstop. (GPT)
- MAX_STAGE_PREFIX_WORDS=6 exact; longer stage dirs not caught -> reroll backstop
  (documented). (GPT/DeepSeek)

## REJECTED / corrected
- The "capital-after-comma = vocative" rule (pass01): DROPPED -- GPT showed it is
  unreliable both ways ("sighs, Wait!" vs "look, Pinky,"). The dialogue-starter-in-
  lead guard (c) handles "look, Pinky" instead; the destructive floor simply does
  NOT fire on the residual ambiguous cases (reroll covers them).
- DeepSeek "no commas in lead": already rejected pass01 (b006 has a comma).

## STILL OPEN -> pass03 (wiring)
- Exact reroll seam: where the RAW candidate line text is visible, and how the new
  `reroll_hint` coexists with the existing critic flag (R2 located reroll_hint at
  `_otr_line_composer` 673-679/1256-1267/1700-1706 + the critic-flag path -- VERIFY).
- `_strip_stage_directions` call-site edit + `CODE_STAGE_DIRECTION` finding.
- 3681 edit (grounded; confirm exact line post any drift).
- The scan script over `scrub_ledger` / frozen ledgers; JSONL output.
- The build-order gate as the sprint's chunk sequencing.
