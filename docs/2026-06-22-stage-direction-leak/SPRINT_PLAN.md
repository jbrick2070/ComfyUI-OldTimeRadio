# Bare stage-direction leak -- SPRINT-READY PLAN (3-pass roundtable converged)

Campaign: pass01 architecture + pass02 coding + pass03 wiring, each = my grounded
critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude judge/synthesizer.
Total panel spend ~$0.51. CONVERGED (final pass produced only build-level
precision items, no new architecture).

## PROBLEM (confirmed from a real frozen ledger)
The writer LLM (mistral-nemo, max-chaos) emits BARE undelimited stage directions
as the leading clause of character dialogue -> frozen ledger `line["text"]` ->
Bark SPEAKS them + SDH captions DISPLAY them. Real leaks in
`signal_lost_spinning_contamination_...`: b004 `"twirls his pen nervously Look,
Pinky..."`, b006 `"pauses, sets pen down Alright, Pinky..."`, b007 `"clenches jaw
You mean..."`. Every existing scrub (`OTR_LedgerScriptWriter.py:3856`,
`_otr_ledger_scrub._strip_stage_directions`, `_otr_line_hygiene.clean_spoken_character_line`,
`_clean_text_for_bark`) is DELIMITED-only `()[]**` -> all miss it.

## ARCHITECTURE (panel-converged)
A naive verb-list-led DESTRUCTIVE strip is UNSAFE (false-positive counterexamples
below). The design is a fallback ladder:
- **PRIMARY = detect-then-REROLL on the RAW draft** (a false positive only costs
  one recompose). A DETERMINISTIC detector drives it -- NOT the LLM critic (the
  failure mode is a weak model, which the critic won't reliably catch).
- **FLOOR = a narrow, fully-guarded DESTRUCTIVE strip, FREEZE-ONLY** (in
  `_otr_ledger_scrub._strip_stage_directions`). NOT added to
  `clean_spoken_character_line` -- doing so would strip before `scrub_ledger` and
  lose the `CODE_STAGE_DIRECTION` finding (grounded: spine Stage 3.7 hygiene only
  bumps `meta["delivery_hygiene_report"]`, it does not emit a ScrubFinding).
- **PROMPT = defense-in-depth** (not relied on for correctness).
- Gated by a **precision check** before the destructive floor activates.

## INVARIANTS
Ledger schema FROZEN (text edits only; telemetry via the existing
`CODE_STAGE_DIRECTION`/`ScrubFinding`, NOT new `meta` fields); `test_audio_byte_identical`
GREEN (clean indextts2 golden lines are unaffected -> the scrub is a no-op on them;
asserted); deterministic + idempotent; model-agnostic (a clean opus line passes
untouched); reuse the Sprint-5C reroll loop; `_otr_line_hygiene` stays PURE (no
logging inside; the freeze caller compares old!=new and logs); restamp `word_count`
whenever text changes; NO workflow-JSON change; UTF-8 no BOM.

## THE DESTRUCTIVE SCRUB -- formal spec (strip iff ALL guards pass, else return input)
`scrub_leading_stage_direction(text) -> str` in `_otr_line_hygiene` (pure, no
`speaker_name`; `_strip_stage_directions` has no speaker). Implementation is
TOKEN-based with a regex only to locate the boundary.
- **(guard 0)** `if not text or not text.strip(' \"\'""''): return text` (empty after a
  prior delimited strip -> no IndexError -- Gemini).
- **(a)** after removing one optional leading quote, `text[0]` is lowercase (radio
  dialogue is capitalized -- the core anomaly).
- **(b)** NO terminal `.!?` anywhere in the leading span before the boundary
  (kills "looks like rain. We should go."). Commas allowed (b006).
- **(c)** the leading span contains NO 1st/2nd-person pronoun
  {i,we,you,me,us,my,your,our} (match contraction ROOTS: we've/you'll/i'm) AND NO
  dialogue-starter {yes,no,well,oh,maybe,please,now,listen,look,hey,okay,fine,sure}.
  (Kills "maybe we should ask John..." and "look, Pinky,..." -- the starter "look"
  guard removes any need for a fragile comma-vocative rule. Conservative: a rare
  objective-pronoun line like "points at me We..." simply bypasses L1 -> reroll.)
- **(d)** the SECOND token is not in `_COPULA_MODAL`
  {is,are,was,were,be,been,being,am,can,could,will,would,shall,should,may,might,
  must,has,have,had,do,does,did} (kills "looks can...", "pauses are...").
- **(e)** boundary = the FIRST capitalized token (optionally one leading quote)
  after the lowercase prefix that is NOT an action OBJECT. SKIP a capitalized token
  whose immediately-previous token is a preposition `_OBJ_PREP`
  {at,to,toward,towards,with,of,for,by,from,over,under,through,into,onto,upon,on,
  in,inside,behind,past,out,about,around,against}, OR an article {the,a,an}, OR a
  possessive adjective {his,her,their,my,your,our,its}, OR a possessive `'s`. If a
  capitalized object was skipped because its previous token is a CONJUNCTION
  {and,or}, ABORT (return input) -- compound-object chains are too risky to strip
  (resolves "looks at Pinky and Brain We need a plan." -> KEEP). A single
  preposition-object skip is allowed ("glances at Pinky We need a plan." -> STRIP).
- **(f)** the lead is <= `MAX_STAGE_PREFIX_WORDS = 6` tokens before the boundary.
- **(g)** the remainder (from the boundary) is >= 2 words, OR is a
  terminal-punctuated short utterance in `_SHORT_UTT`
  {yes,no,wait,stop,never,fine,right,okay,ok,go} (case-insensitive, trailing
  punctuation stripped before lookup; "sighs No." -> "No."; "sighs No" with no
  punctuation -> KEEP).
- If a boundary is not found -> return input. Restamp word_count at the caller.
- **Idempotent:** the result starts capitalized -> guard (a) makes a 2nd pass a
  no-op. Pinned by a test.
- **Documented limitation:** all-lowercase stage directions and prefixes > 6 words
  are NOT caught here -> L3 reroll is the primary defense; the freeze floor is a
  best-effort deterministic subset.

`_COPULA_MODAL`, `_OBJ_PREP`, etc. are SEPARATE constants from the existing
`_NARRATION_VERBS` (different job), documented.

## THE BROAD DETECTOR (reroll-only)
`detect_leading_stage_business(text) -> tuple[bool, str]` -- returns (hit, hint).
Broader than the scrub (it MAY also flag all-lowercase / >6-word leads), but still
guarded against the clear KEEP cases (dialogue-starters, copula-second-token,
clean capitalized lines) so it does not reroll obviously-clean dialogue. `hint` =
`"bare_stage_direction: write only the spoken words, no leading action description"`.

## BUILD ORDER (chunks; each: full suite + Bug Bible -> commit + push to v2.0-alpha)
### Chunk 1 -- pure helpers + the full test corpus (no wiring)
Add `scrub_leading_stage_direction` + `detect_leading_stage_business` +
`_COPULA_MODAL`/`_OBJ_PREP`/`_DIALOGUE_STARTER`/`_SHORT_UTT`/`MAX_STAGE_PREFIX_WORDS`
+ `BARE_STAGE_FLOOR_ACTIVE` (module constant; default decided after the gate) to
`_otr_line_hygiene`. TEST CORPUS:
- STRIP -> dialogue verbatim: b004, b006, b007; "glances at Pinky We need a plan."
  -> "We need a plan."; "sighs No." -> "No.".
- KEEP UNCHANGED: "looks can be deceiving, John."; "pauses are evidence, Brain, not
  proof."; "look, Pinky, we've been through this"; "maybe we should ask John and
  Mary."; "looks like rain. We should go."; "looks at Pinky and Brain We need a
  plan." (conjunction-object abort); "looks at the Map We should go." (article
  skip); any Capitalized line; a clean opus line; the byte-identical golden lines;
  "sighs No" (no terminal punct).
- IDEMPOTENT for all above. Detector tests pin the same KEEP set (no reroll on
  clean dialogue).

### Chunk 2 -- scan script + the PRECISION GATE
`scripts/stage_direction_scan.py`: reads frozen `otr/episodes/**/*_ledger.json`,
emits JSONL with `{source_id,line_id,speaker_role,raw_text,detect_hit,propose_hit,
proposed_text,guard_reasons,would_mutate}`; mutates NOTHING. Run over the soak
corpus; manually inspect; REQUIRE ~zero false positives among `would_mutate=true`.
GATE OUTCOME sets `BARE_STAGE_FLOOR_ACTIVE`: if pass -> True (floor active); if
fail -> False (detect-only; L2 tests then assert reporting, not mutation).

### Chunk 3 -- L3 reroll wiring (PRIMARY)
In `_otr_line_composer.compose_line` (the entry the spine's `_recompose` already
calls with `reroll_hint=`), run `detect_leading_stage_business` on the RAW
candidate text BEFORE any hygiene scrub; on a hit, CONCATENATE the hint with any
existing critic hint (`f"{existing}; {stage_hint}"` else `stage_hint`) and feed the
existing reroll loop (verify the loop's max-attempts constant). Exhaustion: accept
the last draft (the freeze floor is the deterministic backstop for the cases it
can catch; residual all-lowercase/long leaks are an accepted, logged risk).

### Chunk 4 -- L2 freeze floor (gated on Chunk 2)
`_otr_ledger_scrub._strip_stage_directions`: KEEP the delimited logic; let `out` be
the post-delimited text; `bare = scrub_leading_stage_direction(out)` when
`BARE_STAGE_FLOOR_ACTIVE`; `return (bare, delimited_changed or bare != out)`
(preserve the `Tuple[str,bool]` contract -- a naive replace crashes the
`cleaned, stripped = ...` unpack). `scrub_ledger` emits the existing
`CODE_STAGE_DIRECTION` finding (with the stripped prefix in `detail`) and restamps
`word_count` on a bare strip. Tests: contaminated line entering ONLY via
`scrub_ledger` is cleaned; non-speech rows untouched; the byte-identical golden
lines are a NO-OP. Import the helper (dual import; both modules are stdlib-only
leaves -> no cycle -- grounded).

### Chunk 5 -- L4 prompt + the S1 music patch
- Composer prompt: add negative + POSITIVE constraint ("Write ONLY the spoken
  words. Start directly with the first spoken word. NEVER prefix dialogue with an
  action like 'twirls his pen' or 'clenches jaw'."). No build-blocking
  prompt-compliance test.
- `OTR_LedgerScriptWriter.py`, inside the `NON_VOICED_ROLES` branch (grounded --
  the `cleaned = (beat.sfx_cue or beat.intent or "").strip()` that feeds the
  `[SFX: ...]` render-contract `desc` + `update_line_text`): change to
  `(beat.sfx_cue or "").strip()`. Tests: a `music_inter` beat (no cue) -> ledger
  `text == ""` and the transcript omits it; a real `sfx_cue` is preserved. Mirrors
  the S1 `init_lines_from_outline` fix this loop was overwriting.

## FINAL QA
Re-soak (any writer; bark-forced is fine) -> run `stage_direction_scan.py` over the
new ledger -> ZERO leaks; `test_audio_byte_identical` green; suite + Bug Bible per
chunk; commit AND push per green chunk to v2.0-alpha; prod/main GATED.
