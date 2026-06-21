# Bare stage-direction leak -- fix plan (ARCH + CODING hardened, pass02)

> REVIEW FOCUS NEXT PASS (pass03): **WIRING.** The exact reroll seam (where the
> RAW candidate line text is available + how the new hint coexists with the
> critic flag), the `_strip_stage_directions` call-site edit (preserve the
> `Tuple[str,bool]` contract + `CODE_STAGE_DIRECTION` finding), the 3681 edit, the
> scan script, and the build/precision-gate ORDER. Ground vs `_otr_line_composer.py`
> (compose_line_draft + Sprint-5C reroll) + `OTR_LedgerScriptWriter.py` (3681 + the
> raw-draft seam) + `_otr_ledger_scrub.scrub_ledger`.

## Problem + decision (unchanged from pass01)
Bare undelimited stage directions ("twirls his pen nervously Look, Pinky...") leak
into the FROZEN ledger text -> TTS + captions. Every existing scrub is delimited-
only. FIX = **detect-then-REROLL on the raw draft (primary) + a NARROW, fully-
guarded destructive FLOOR at freeze (secondary), gated on a precision check.**
A verb-list-led naive strip is UNSAFE (panel counterexamples below).

## INVARIANTS
Ledger schema FROZEN (text-only edits; telemetry via existing `ScrubFinding`, NOT
new `meta` fields -- panel CUT); `test_audio_byte_identical` GREEN (clean inputs
byte-identical; the indextts2 golden lines are clean -> the scrub is a NO-OP on
them, asserted by a test); deterministic + idempotent; model-agnostic; reuse the
existing Sprint-5C reroll loop; `_otr_line_hygiene` stays a PURE module (no logging
inside -- callers log); NO workflow-JSON change; UTF-8 no BOM.

## L1 -- the destructive floor: FORMAL spec (strip ONLY if ALL guards pass)
Two helpers in `_otr_line_hygiene`, both pure, never raise, return input on any
empty/uncertain result:
- `detect_leading_stage_business(text) -> tuple[bool,str]` -- BROAD (reroll-only).
- `scrub_leading_stage_direction(text) -> str` -- NARROW (destructive). NO
  `speaker_name` arg (the freeze normalizer has no speaker). Internally a
  `_propose_leading_stage_strip(text) -> (hit:bool, stripped:str)` so callers can
  log the hit while the public fn stays a pure `str->str`.

**Strip iff ALL of (else return text unchanged) -- the "confidence" definition:**
- (a) after removing an optional single leading quote (`"` or curly), `text[0]`
  is lowercase. (Capitalized line -> not our case; also fixes the quoted-line gap.)
- (b) there is NO terminal punctuation `. ! ?` anywhere in the leading span before
  the boundary (kills "looks like rain. We should go." -- GPT). Commas are allowed
  (b006 "pauses, sets pen down" has one).
- (c) the leading span contains NO first/second-person pronoun
  {i,you,we,us,me,my,your,our} AND NO dialogue-starter
  {yes,no,well,oh,maybe,please,now,listen,look,hey,okay,fine,sure} (Gemini: kills
  "maybe we should ask John...", and also "look, Pinky,..." because "look" is a
  starter -> we never need the fragile comma-vocative rule).
- (d) the SECOND token is not a copula/modal/auxiliary `_COPULA_MODAL` =
  {is,are,was,were,be,been,being,am,can,could,will,would,shall,should,may,might,
  must,has,have,had,do,does,did} (kills "looks can...", "pauses are...").
- (e) a boundary is found = the first token that begins with `[A-Z]` (optionally
  one leading quote) and STARTS A NEW SENTENCE, where a capitalized token is SKIPPED
  (treated as an action object, not the boundary) when its previous token is a
  preposition `_OBJ_PREP` = {at,to,toward,towards,with,of,for,by,from,over,under,
  through,into,onto,upon,on,about,around,against} OR a conjunction {and,or} OR ends
  in a possessive (`'s` / curly). (kills "glances at Pinky We..." and
  "looks at Pinky and Brain We..." -- GPT/Gemini.)
- (f) the leading span is <= `MAX_STAGE_PREFIX_WORDS = 6` tokens. (Longer stage
  directions are NOT caught by L1 -> L3 reroll is the backstop; documented.)
- (g) the remainder (from the boundary) is >= 2 words, OR is a terminal-punctuated
  short utterance in `_SHORT_UTT` = {yes,no,wait,stop,never,fine,right,okay,ok,go}
  (case-insensitive, trailing punctuation stripped before lookup) -> "sighs No." ->
  "No.". Else return text.
Operate on the RAW string with a regex for boundary detection (NOT only
`str.split()`), so the punctuation-before-capital tests are visible.
**Idempotent by construction:** the result starts capitalized -> guard (a) makes a
second pass a no-op. Pinned by an idempotency test.
**Limitation (documented inline):** an ALL-lowercase stage direction
("twirls his pen. look, pinky") is NOT caught deterministically; L3 reroll is the
backstop.

## L2 -- freeze choke point (preserve the existing contract)
`_otr_ledger_scrub._strip_stage_directions(text) -> Tuple[str,bool]`: KEEP the
existing delimited logic, THEN `bare = scrub_leading_stage_direction(out)`; return
`(bare, delimited_changed or bare != out)`. (Gemini: a naive replacement crashes
the `cleaned, stripped = ...` unpack at the call site.) Import the shared helper
(dual import; both modules are stdlib-only leaves -> no cycle). `scrub_ledger`
emits the existing `CODE_STAGE_DIRECTION` finding for a bare strip too (GPT).
**Canonical sequence in BOTH call sites (DeepSeek):** delimited-strip THEN
bare-strip. `clean_spoken_character_line` becomes
`scrub_self_vocative(scrub_leading_stage_direction(scrub_parentheticals(text)), name)`.
The PARITY test asserts the shared HELPER produces identical output on b004/b006/
b007 from both entry points -- NOT that the two full pipelines match (ledger_scrub
also normalizes quotes/dashes/whitespace, so a whole-pipeline equality would fail).

## L3 -- detect-then-reroll on the RAW draft (PRIMARY)
`detect_leading_stage_business` runs on the EXTRACTED candidate line text (not the
whole model response) BEFORE any hygiene scrub mutates it. On a hit -> set
`reroll_hint` and reuse the Sprint-5C `compose_line_draft` loop (no new infra).
EXHAUSTION behavior (GPT): after the 3-attempt budget, ACCEPT the last draft -- the
L1 freeze floor is the deterministic backstop that cleans whatever shipped.
(Wiring pass pins the exact seam + how the new hint coexists with the critic flag.)

## L4 -- prompt (defense-in-depth) + S1 music patch
- Composer prompt: negative + POSITIVE constraint ("Write ONLY the spoken words.
  Start directly with the first spoken word. NEVER prefix dialogue with an action
  like 'twirls his pen' or 'clenches jaw'."). NO build-blocking prompt-compliance
  test (panel CUT); all gating tests feed fixed strings with the prompt ignored.
- `OTR_LedgerScriptWriter.py:3681` (grounded INSIDE the `NON_VOICED_ROLES` branch):
  `(beat.sfx_cue or beat.intent or "")` -> `(beat.sfx_cue or "")`. music_inter (no
  cue) -> ""; a genuine `sfx_cue` stays (render-contract `desc`). Mirrors S1.

## PRECISION GATE + BUILD ORDER (resolves the pass01 sequencing contradiction)
1. Build the helpers + `detect` + `scripts/stage_direction_scan.py` in
   DETECT/PROPOSE mode (reports candidate detections + the proposed strip as JSONL;
   mutates NOTHING).
2. Run the scan over the soak corpus. REQUIRE ~zero false positives on the proposed
   destructive strips.
3. Only then enable the destructive floor (L1 active in `_strip_stage_directions` +
   `clean_spoken_character_line`). If the gate fails, ship DETECT-ONLY (L3 reroll +
   L4 prompt) and keep L1 as the proposer -- the L2 tests then assert reporting, not
   mutation. Do NOT leave both outcomes required.

## TEST CORPUS (every counterexample pinned)
STRIP -> dialogue verbatim: b004/b006/b007; "glances at Pinky We need a plan." ->
"We need a plan."; "sighs No." -> "No.".
KEEP UNCHANGED: "looks can be deceiving, John."; "pauses are evidence, Brain, not
proof."; "look, Pinky, we've been through this"; "maybe we should ask John and
Mary."; "looks like rain. We should go."; "looks at Pinky and Brain We need a
plan." (object boundary); any Capitalized line; a clean opus line; the
byte-identical golden lines.
IDEMPOTENT: scrub(scrub(x)) == scrub(x) for ALL above.
PARITY: the shared helper yields identical output via both call sites on b004/6/7.
COVERAGE: a contaminated line entering ONLY via `scrub_ledger` is cleaned; a
non-speech row is NOT altered by the bare scrub.
