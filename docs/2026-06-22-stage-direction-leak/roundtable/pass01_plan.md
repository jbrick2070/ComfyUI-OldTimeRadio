# Bare stage-direction leak -- fix plan (ARCHITECTURE hardened, pass01)

> REVIEW FOCUS NEXT PASS (pass02): **CODING.** Exact helper signatures, the
> structural disambiguation algorithm + its guards, the precise regex/token
> boundary rules, the test corpus (incl. every false-positive counterexample),
> and idempotency. Ground against `_otr_line_hygiene.py` + `_otr_ledger_scrub.py`.

## The problem (confirmed)
The writer LLM (mistral-nemo, max-chaos) emits BARE, undelimited stage directions
as the leading clause of character dialogue; they land in the FROZEN ledger
`line["text"]` -> Bark speaks them + SDH captions show them. Real leaks
(`signal_lost_spinning_contamination_...`): b004 `"twirls his pen nervously Look,
Pinky..."`, b006 `"pauses, sets pen down Alright, Pinky..."`, b007 `"clenches jaw
You mean..."`. Every existing scrubber (`OTR_LedgerScriptWriter.py:3856`,
`_otr_ledger_scrub._strip_stage_directions`, `_otr_line_hygiene.clean_spoken_character_line`,
`_clean_text_for_bark`) only matches DELIMITED shapes `()[]**` -> all miss it.

## THE KEY ARCHITECTURE DECISION (panel-converged + judged)
A verb-list-led DESTRUCTIVE strip is UNSAFE. The panel produced decisive
false-positive counterexamples that a naive "lowercase + action verb + first
later capital" rule would MANGLE:
- `"looks can be deceiving, John."` (GPT/Gemini) -- valid dialogue, "looks" is a
  noun subject; naive strip -> `"John."`
- `"pauses are evidence, Brain, not proof."` (GPT) -- "pauses" is a plural noun.
- `"look, Pinky, we've been through this"` (DeepSeek) -- imperative dialogue.
- `"glances at Pinky We need a plan."` (GPT/Gemini) -- proper-noun OBJECT of the
  action; naive "first capital" boundary -> `"Pinky We need a plan."`

=> **Primary fix = DETECT-then-REROLL on the RAW draft. Secondary = a NARROW,
heavily-guarded destructive floor at FREEZE, gated on a precision check.** Both
panels agree reroll-first is safe (a false positive costs one retry, not data
loss); the destructive floor exists only because freeze is the LAST guarantee
before audio and a weak model can exhaust the reroll budget.

## INVARIANTS (unchanged; a fix that breaks one is rejected)
Ledger `{cast,lines,meta}` schema `l3-2026-05-14` FROZEN (text edits only, new
data in free-form `meta`); `test_audio_byte_identical` GREEN (see Invariant Note);
deterministic + idempotent (C7-safe); model-agnostic (a clean opus line passes
untouched); reuse the EXISTING Sprint-5C `reroll_hint` loop; NO workflow-JSON
change; UTF-8 no BOM.

**Invariant note (GPT, accepted):** "audio byte-identical" means CLEAN inputs stay
byte-identical. A contaminated line's text changing pre-freeze WILL change its
bark bytes -- that is intended. The byte-identical fixture is the indextts2 path
with CLEAN lines, so the scrub is a NO-OP on it. A required test asserts the new
scrub leaves the byte-identical golden lines unchanged.

## BUILD PLAN -- 4 layers (reroll-primary, freeze-floor secondary)

### L3 (PRIMARY) -- detect-then-reroll on the RAW draft
- A detector `detect_leading_stage_business(text) -> (bool, reason)` runs on the
  RAW LLM draft BEFORE any hygiene scrub mutates it (all 3 panels: scrubbing first
  destroys the signal and bypasses reroll). The detector may be BROAD (a false
  positive only costs one recompose).
- On a hit, set `reroll_hint = reason` -> the EXISTING `compose_line_draft`
  reroll loop (Sprint-5C, the critic-flag path R2 already located) recomposes.
  No new reroll infra; the existing 3-attempt budget bounds it.

### L1 (FLOOR) -- narrow, guarded destructive scrub (shared helper)
- `scrub_leading_stage_direction(text) -> str` in `_otr_line_hygiene` (NO
  `speaker_name` arg -- `_strip_stage_directions(text)` has no speaker; panel
  CUT). Folded into `clean_spoken_character_line` AND called by the freeze
  normalizer (L2). STRUCTURAL, not a verb whitelist (panel-converged):
  - Fire ONLY when the line STARTS lowercase (real radio dialogue is
    capitalized -- the core anomaly, near-zero false positive on capitalized text).
  - Identify the leading-action span vs a real lowercase sentence by the SECOND
    token: if it is a copula/modal/auxiliary (`is are was were be been can could
    will would should has have had do does did`), the first word is a SUBJECT ->
    NOT a stage direction (kills "looks can...", "pauses are...").
  - The dialogue boundary = the first Capitalized token that starts a NEW sentence
    -- SKIP capitalized tokens that are objects of the action (preceded by `at to
    toward with of for` or a possessive) so "glances at Pinky We..." cuts at "We".
    A capital preceded by `", "` is a vocative, NOT a boundary (keeps "look,
    Pinky,...").
  - Allow an optional leading quote (`"` or curly) before the capitalized start.
  - Bound the leading span to <= ~6 words.
  - Keep ORIGINAL if the strip would leave < 2 words, UNLESS the remainder is a
    terminal-punctuated short utterance (`yes no wait stop never fine right okay`)
    -- so "sighs No." -> "No." (GPT).
  - When the structural guards cannot make a CONFIDENT call -> DO NOT strip (leave
    it for L3/L4; a rare ship beats a mangled line -- Gemini's risk, bounded).
  - Documented limitation (DeepSeek): an ALL-lowercase stage direction ("twirls
    his pen. look, pinky") is NOT caught deterministically; L3 reroll is the
    backstop. Inline comment required.

### L2 -- freeze is the bypass-proof choke point
`_otr_ledger_scrub._strip_stage_directions` imports and calls the SAME
`scrub_leading_stage_direction` helper (not a copy) so EVERY spoken field is
cleaned at freeze regardless of writer path. `clean_spoken_character_line` calls
the same helper. A test asserts both paths produce identical output on b004/b006/
b007 AND that a contaminated line entering ONLY via `scrub_ledger` is cleaned.

### L4 -- prompt (defense-in-depth, NOT a guarantee) + the S1 music patch
- Composer prompt: negative + a POSITIVE constraint (Gemini): "Write ONLY the
  spoken words. Start the dialogue directly with the first spoken word. NEVER
  prefix dialogue with an action like 'twirls his pen' or 'clenches jaw'." Tests
  must pass with the prompt IGNORED (DeepSeek/GPT).
- `OTR_LedgerScriptWriter.py:3681` (already INSIDE the `NON_VOICED_ROLES` branch
  -- grounded, so voiced rows are untouched): `(beat.sfx_cue or beat.intent or "")`
  -> `(beat.sfx_cue or "")`. music_inter (no cue) -> `""`; a genuine `sfx_cue`
  stays (it is a render-contract `desc` consumed by the sequencer -- keep, per the
  S1 analysis). Mirrors the S1 `init_lines_from_outline` fix that this loop was
  overwriting.

## PRECISION GATE (before the destructive L1 ships broadly)
A measurement script (`scripts/stage_direction_scan.py`, GPT+DeepSeek) scans
frozen ledgers and reports candidate detections + what L1 WOULD strip, WITHOUT
mutating. Run it over the soak corpus; require ~zero false positives before
enabling the destructive floor. If precision is not achievable, fall back to
Gemini's reroll-only stance (L3+L4) and keep L1 as detect-only.

## TELEMETRY (free-form meta only -- no schema change)
Log a WARNING each time L1 strips, and (optional) a `meta` counter, so production
firing rate is observable for tuning.
