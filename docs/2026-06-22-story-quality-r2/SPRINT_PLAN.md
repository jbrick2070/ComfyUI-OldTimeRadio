# Story Quality R2 -- SPRINT-READY BUILD PLAN (seams located; build order)

Campaign: pass01 (structural roundtable) + pass01b (Claude creative pass) + pass02 (coding
roundtable) + seam-location. 8 levers (3 structural + 5 creative); "best story" the north star.
HARD: ledger {cast,lines,meta} schema l3-2026-05-14 FIXED -> new fields go in FREE-FORM `meta`, NEVER
new Pydantic fields; audio frozen; MODEL-AGNOSTIC (every gate is one opus passes -> lifts the weak
end, never rewrites opus); craft-ONLY (no word/beat/budget change); reuse the EXISTING targeted-reroll
loop, no new reroll infra. Each chunk: suite + Bug Bible -> commit + push.

## LOCATED SEAMS (verified)
- Line composer: `_otr_line_composer.compose_line_draft(...)` -- builds the voiced-line prompt; ALREADY
  reads `cast_row.speech_signature` (line ~885-905) and ALREADY supports a Sprint-5C `reroll_hint`
  targeted-reroll (lines 673-679, 1256-1267, 1700-1706). The story-critic already flags lines ->
  reroll_hint. NEW gates feed THIS path.
- Spoken-role set: `_otr_ledger_scrub._SPOKEN_ROLES` (the `is_spoken_role` helper) + `row["text"]`
  materialization -- the S1 suppression point.
- Beat intents: `_otr_outline.generate_outline` Stage 3 `_build_beat_user_prompt` / `_BeatFleshout`
  (action-verb + escalation belong HERE, not the line composer).
- Outline beat seeds: `_otr_outline._assemble_outline` (music_inter + announcer-close intents).
- Cast rows: `_otr_casting.py` carries `speech_signature` (F5 EXISTS -> C3 = strengthen, no schema change).
- Wants: `_otr_dramatic_state.derive_dramatic_state_from_meta` + `_DEFAULT_A/B_WANTS` (the non-default
  classifier lives here).

## BUILD ORDER (each its own green chunk)

### Chunk 1 -- S1: music/non-dialogue beats never render as spoken/caption text
- Confirm `music_inter` (+ music_open/music_close/sfx) are NOT in `_otr_ledger_scrub._SPOKEN_ROLES`;
  at the text-materialization point set `text=""` (or skip) for non-spoken roles while keeping the
  beat/timing/music row + `dialogue_slot_id=None`. Change the `_assemble_outline` music_inter intent
  to neutral ("Bridge to the next phase with music only."). TEST: no transcript/caption contains
  "Musical interlude bridging"; music_inter row count + voiced slot ids unchanged.

### Chunk 2 -- S2: announcer close = final image, not thesis
- `_assemble_outline` close intent -> "Close on a concrete final image showing what changed (use the
  central object if set); no moral, thesis, or news-summary tag." Add a SHARED module constant of
  banned-thesis regexes (case-insensitive, straight+curly apostrophe): `Tonight['’]s revelation`,
  `the lesson is`, `reminding us`, `proving \w+ right`, `\w+ is now shared`, `this shows`. Scan the
  composed close; on a hit reroll via the announcer composer (NOT the character path). TEST: the 3
  grounded close failures reroll.

### Chunk 3 -- S3: cliche + stage-business reject gate (hygiene = FLAGS ONLY)
- `_otr_line_hygiene`: add `flag_cliche(text)` + `flag_stage_business(text)` (small grounded lists:
  "you're playing with fire"/"this changes everything"/"we're not leaving anything to chance";
  "I'll go check"/"I'll double-check"/"I'll lock down"/"I've got this, no need") returning
  (flagged, reason). In the spine's existing reroll seam, a flagged voiced line sets `reroll_hint`
  = the reason -> the EXISTING compose_line_draft reroll loop recomposes it (no new infra; no cap --
  the 3-attempt budget bounds it). TEST: flagged lines reroll; clean lines pass.

### Chunk 4 -- C0: non-default wants + action-verb in OUTLINE Stage 3
- `_otr_dramatic_state`: add `wants_are_default(state) -> bool` (match vs `_DEFAULT_A/B_WANTS`
  templates). `_otr_outline._build_beat_user_prompt`: require each beat intent to be an ACTION VERB
  UNDER PRESSURE (reveal/refuse/demand/bargain/accuse/conceal/choose) + (when non-default) ground in
  the opposed wants; add a Stage-3 post_validator. TEST: a default-wants state is detected; the beat
  prompt carries the verb constraint.

### Chunk 5 -- C1: specificity anchors
- One cheap setup call on the resident writer slot (or deterministic extraction from the news brief):
  3-5 concrete anchors (place/object/number/named bystander) -> `meta["specificity_anchors"]`.
  Inject into `compose_line_draft`'s prompt ("use these concrete anchors"). Gate: a character line on
  a non-opener/closer/music beat with NO anchor + NO proper noun (case-folded vs anchors; EXCLUDE
  cast names + sentence-initial caps) -> reroll_hint. TEST: anchors stamped; generic line flagged.

### Chunk 6 -- C2: central story-object
- Derive `meta["central_object"]` at setup (cheap call / from the brief). Act-1 beat objective
  introduces it; mid complicates; the S2 close references it. (Ordering: C2 derive BEFORE the close.)
  TEST: central_object stamped + referenced in the close prompt.

### Chunk 7 -- C3: voice distinctness
- `_otr_casting`: derive CONTRASTING `speech_signature`s (clipped vs verbose, plain vs ornate) so two
  characters never share a register. Promote the existing signature clause in `compose_line_draft` to
  a HARD per-line constraint. TEST: contrasting signatures at lock; the clause is mandatory.

### Chunk 8 -- C4 + C5: escalation + subtext (lightest)
- C4: `_build_beat_user_prompt` -- each phase's beat objective must RAISE the concrete stake over the
  prior phase. C5: on TURN/climax beats only, `compose_line_draft` adds "imply the pressure, don't
  name it"; a gate flags on-the-nose emotion ("I'm scared"/"this is dangerous") -> reroll_hint on
  those beats only. TEST: escalation clause present; on-the-nose flagged on the turn beat.

## WIRING (confirmed)
NO workflow-JSON / node / widget change -- all content inside OTR_LedgerScriptWriter + its modules
(_otr_outline / _otr_line_composer / _otr_line_hygiene / _otr_dramatic_state / _otr_casting /
_otr_ledger_scrub) + the news brief. New meta keys ride the free-form `meta`. New setup calls reuse
the resident writer slot (V-11, no model widget).

## FINAL QA
Extend `scripts/story_quality_scan.py` with the 4 structural counts (music-placeholder / meta-close /
cliche / stage-business) + craft signals (proper-noun density / central-object recurrence / per-act
stake escalation / voice distinctness). Re-soak: 2-3 weak-local + 1 frontier leg (visualizer, cheap);
read the scripts. GATE: weak-end metrics DROP; the frontier/opus leg does NOT regress (still passes
every gate -> untouched).

## Anti-regression
Every gate is one the opus sample passes (proper nouns, a central object, escalation, distinct
voices, image-ending) -> opus is never rerolled; only the weak end is lifted.
