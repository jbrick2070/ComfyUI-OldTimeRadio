# Bare stage-direction leak -- fix plan to harden (OTR story ledger)

> REVIEW FOCUS THIS PASS: **ARCHITECTURE / APPROACH.** Is the detection signal
> sound? Is the layering (deterministic scrub net vs reroll gate vs prompt)
> right? What is the false-positive risk and how do we bound it? Where should the
> canonical scrub live so it cannot be bypassed by a writer code path?

## The problem (confirmed from a real frozen ledger)
The writer LLM (mistral-nemo at "maximum chaos" creativity) emitted **bare,
undelimited stage directions** as the leading clause of character dialogue. They
landed in the FROZEN ledger `line["text"]`, so Bark SPOKE them and the burned SDH
captions DISPLAYED them. Real examples from
`signal_lost_spinning_contamination_20260621_155545` (6-line episode, 3 of the 6
character lines affected):

- b004: `"twirls his pen nervously Look, Pinky, we've been through this..."`
- b006: `"pauses, sets pen down Alright, Pinky. We've had a breach. It's my fault."`
- b007: `"clenches jaw You mean like the one you assured me couldn't happen?"`

Shape: a leading **lowercase** action clause (verb-led: twirls / pauses /
clenches), glued directly to the capitalized real dialogue. No parentheses, no
brackets, no asterisks.

## Why every existing scrubber misses it (all are DELIMITED-only)
OTR already has four stage-direction scrubbers; each only matches delimited
shapes `(...)`, `[...]`, `*...*`:

1. `OTR_LedgerScriptWriter.py:3856` (pre-freeze "I.6" dialogue scrub):
   `_sd_re = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")` -> parens/brackets only.
2. `_otr_ledger_scrub._strip_stage_directions` (the freeze-time NORMALIZER, runs
   on every spoken field via `scrub_ledger`): `_STAGE_DIRECTION_RES` = `[...]`,
   `*...*`, and cue-like `(...)` only.
3. `_otr_line_hygiene.clean_spoken_character_line` = `scrub_parentheticals` +
   `scrub_self_vocative` (parens + own-name vocative only).
4. `_clean_text_for_bark` (TTS prep): drops `(...)`, `*...*`, `[...]` only.

There is NO detector anywhere for an UNDELIMITED leading stage direction.
`_otr_line_hygiene.detect_narration_self_address` does not help: it only fires on
a `he/she/they` or speaker-name lead, and its `_NARRATION_VERBS` set excludes
twirls / pauses / clenches. The composer prompt says "No stage directions"
(`_otr_line_composer.py:953`) but the model ignored it under max-chaos.

## Secondary defect found while tracing this (in scope to fix together)
The music_inter row's text is RE-STAMPED with the beat intent in the writer
composition loop, defeating an earlier suppression fix:
`OTR_LedgerScriptWriter.py:3681` -> `cleaned = (beat.sfx_cue or beat.intent or "").strip()`
then `update_line_text(beat.beat_id, cleaned)` (line 3706). So non-voiced rows
re-acquire the intent text (e.g. `"Bridge to the next phase with music only."`)
in the ledger + the `[SFX: ...]` script transcript. (Captions already exclude
non-speech roles, so this is transcript/ledger pollution, not a caption bug.)

## HARD INVARIANTS (a fix that breaks one is rejected)
- Ledger `{cast,lines,meta}` schema `l3-2026-05-14` is FROZEN -- content edits to
  `line["text"]` only; NO new Pydantic fields (new data rides free-form `meta`).
- Audio spine FROZEN: `test_audio_byte_identical` (indextts2 path) stays green;
  any text edit must happen PRE-FREEZE / pre-audio so it cannot change the mux.
- Deterministic + idempotent (C7 byte-identity safe); pure-stdlib scrub helpers.
- Model-AGNOSTIC: the gate must lift the weak writer end without rewriting a
  clean (e.g. opus) line. NO workflow-JSON / node / widget change. UTF-8 no BOM.
- Reuse the EXISTING Sprint-5C `reroll_hint` loop in
  `_otr_line_composer.compose_line_draft` -- no new reroll infrastructure.

## PROPOSED FIX (the starting plan to harden) -- 4 layers
### L1 -- deterministic bare-stage-direction scrub (the guaranteed floor)
Add a pure helper `scrub_leading_stage_direction(text, speaker_name) -> str` to
`_otr_line_hygiene` and fold it into `clean_spoken_character_line`. Detection
heuristic (low false-positive):
- The line STARTS lowercase (real radio dialogue is capitalized), AND
- the leading clause's first word is an action/stage verb (an expanded set:
  twirls, pauses, clenches, sets, leans, sighs, nods, shrugs, glances, paces,
  gestures, taps, rubs, crosses, folds, narrows, exhales, ... incl. -s/-ing/-ed),
  AND
- there is a capitalized continuation (the first token that begins with a capital
  and starts a real sentence). Strip the leading lowercase run up to that point;
  keep the dialogue. Keep the ORIGINAL text if stripping leaves < 2 words.

### L2 -- make the freeze normalizer the single bypass-proof choke point
Wire L1 into `_otr_ledger_scrub._strip_stage_directions` so EVERY spoken field is
cleaned at freeze regardless of which writer path produced it (the writer's I.6
scrub is then a best-effort early pass, but freeze is the guarantee). One canonical
helper, two call sites (line_hygiene for the composer/spine, ledger_scrub for the
freeze). Reject silent divergence (the two must share the helper).

### L3 -- reroll gate (the craft fix; this IS R2 sprint item S3)
In the spine's existing reroll seam, a line flagged as stage-business sets
`reroll_hint = the reason` -> the EXISTING `compose_line_draft` reroll loop
recomposes it (no new infra; the 3-attempt budget bounds it). Regenerating beats
scrubbing on quality; L1/L2 are the backstop when the budget is exhausted.

### L4 -- prompt hardening + the S1 music patch
- Composer prompt: add an explicit negative example -- "Write ONLY the spoken
  words. NEVER prefix dialogue with an action like 'twirls his pen' or 'clenches
  jaw'."
- `OTR_LedgerScriptWriter.py:3681`: drop the `or beat.intent` fallback for
  non-voiced rows so music/sfx text stops re-acquiring the intent.

## OPEN QUESTIONS FOR THE PANEL
1. Is "starts lowercase + leading action verb" the right signal, or is there a
   stronger/safer one (e.g. POS-free heuristics that also catch a Capitalized
   stage direction like "Twirls his pen. You mean...")? How do we avoid stripping
   legitimate dialogue?
2. Should the canonical scrub live in line_hygiene (shared) and be called from
   both the composer AND the freeze normalizer, or only at freeze? Bypass risk?
3. Scrub-vs-reroll ordering: scrub first then reroll only if still bad, or reroll
   first? Which preserves craft without risking an infinite loop?
4. Anything that makes this NOT model-agnostic or NOT idempotent?
