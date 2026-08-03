# The writer's scaffolding problem: pick the most robust repair method

**Decision doc for a consensus panel. 2026-08-03, HEAD `1f329d30`.**
Operator constraint, verbatim intent: *no hardcoded pattern-chasing of strings
like `**SCENE 5**` -- an LLM should be asked to look for non-dialogue items and
remove them or do what is appropriate with them.*

## The failure, from last night's live run

Two of seventeen sweep legs died in the writer, killing the leg before any
video engine loaded:

* `ltx_audio_in`: the model decorated its script with markdown structure --
  `**SCENE 5**`, `**SCENE 6**`, `**MUSIC**`, `**CODA**` -- and the parser read
  each as a speaker label: `UNKNOWN_SPEAKER: **SCENE 5 (line 24)` and so on.
  Four repair attempts, all failed, ladder exhausted.
* `viz_mxc_cpu`: the model invented a character not in the locked cast
  (`UNKNOWN_SPEAKER: DR. MOURKIOTI`, three lines). Same exhaustion.

18% of legs were lost to writer/cast failures; the video layer lost none.

## What already exists (grounded, with line numbers)

1. **A deterministic decoration stripper** --
   `_otr_fable2_markup._canonicalize_transport_line` (`:52-109`), added
   2026-08-01 after this exact class killed three of six episodes. It strips
   BALANCED emphasis in three shapes: `**LABEL:** payload`, `**LABEL**: payload`,
   `**TOKEN**`. Last night's script used a FOURTH shape -- the wrapper spanning
   label and payload (`**SCENE 5: ...**` / heading-style lines) -- so the strip
   missed and the speaker catch-all fired. The operator's point stands: shapes
   are an open set; enumerating them is a losing game.
2. **An LLM repair rung** -- the markup ladder (`_otr_scifi_fable2.py:1664-1744`)
   already re-prompts the model with its defects after every failed parse:
   *"Repair only the malformed FORMAT defects below... keep the same story,
   cast, events, and wording."* It ran four times last night and failed four
   times.
3. **Why it failed: the diagnostic lies.** The defect text says
   `UNKNOWN_SPEAKER: **SCENE 5` -- which tells the model "this CHARACTER is not
   in the cast." The true problem is "this line is NOT DIALOGUE." The repair
   model is being sent to fix the wrong thing, four times, and the one thing it
   is never told is the one thing that would fix it.
4. **Precedent for defect-class-specific repair notes** --
   `_standalone_stage_direction_repair_note(defect_rows)` is already appended to
   the repair prompt (`:1736`), so the mechanism for "when defect class X is
   present, add targeted guidance" exists and is tested.
5. **Telemetry for a guard** -- accepted attempts record
   `character_word_count` and per-scene counts in `PassAttemptTrace`
   (`:1702-1709`).

## Candidate methods

**A. A default-on LLM "janitor" pre-pass.** Before every parse, an LLM sweep
classifies each line (dialogue / scaffolding / direction) and normalizes.
Honest cost: one extra LLM call on EVERY script attempt, including the ~majority
that are already clean; a janitor at temperature is itself a new failure
surface; and it can mangle text the parser would have accepted. Rejected as the
primary: pays the cost always, needed rarely.

**B. Re-aim the EXISTING repair rung (recommended core).** When
`UNKNOWN_SPEAKER` defects are present, the repair prompt gains a dual-framing
note: *"A line flagged UNKNOWN_SPEAKER is one of two things. If it is dialogue,
attribute it to the correct member of this exact cast: [roster]. If it is not
dialogue at all -- a scene heading, act break, music cue, chapter mark, or any
other scaffolding -- remove it or fold it into a stage direction. The output
format has no headings beyond the transport lines shown in the example."*
The model that wrote the scaffolding decides what it meant -- no pattern list,
generalizes to `## Coda`, `--- ACT 2 ---`, `[MUSIC STING]`, and whatever comes
next. ALSO repairs the invented-character class (`DR. MOURKIOTI`): "attribute
it to the correct cast member" is exactly the right instruction there.
Zero new passes, zero happy-path cost; engages from attempt 2.

**C. Quarantine + micro-adjudication.** Parser quarantines would-be
UNKNOWN_SPEAKER lines instead of failing, finishes the parse, then ONE tiny LLM
call adjudicates only the quarantined lines (`DIALOGUE_BY <name> | DROP |
DIRECTION`), verdicts applied deterministically, re-validate. Surgical -- the
LLM cannot touch non-quarantined text by construction. Cost: new machinery
(quarantine list, adjudication protocol, re-entry), a second LLM call shape,
and out-of-context misclassification risk. Held as the fallback if B proves
insufficient live, not built now.

**D. Prompt-side prevention.** One line in the system/format example: plain
text only, no markdown emphasis, no headings beyond the transport shown.
Reduces incidence, cannot close the class. Ship alongside B; never the fix.

**E. Complete the balanced-wrapper transport rule.** Extend
`_canonicalize_transport_line` with the fourth balanced shape (wrapper around
the WHOLE line). This is NOT per-string chasing -- it completes the existing
rule "balanced emphasis around transport is transport", closing every `**...**`
variant permanently. Deterministic, reported, roster-independent. Ship
alongside B as the free catch; unbalanced and non-emphasis scaffolding still
flows to B.

## The one hard problem: a repair model that DELETES DIALOGUE fails silently

Tell a model "remove non-dialogue" and its failure mode is removing dialogue --
and that failure is INVISIBLE, because the parse then SUCCEEDS with fewer
lines. Under this repo's ledger rules a silent hole is strictly worse than a
loud dead leg.

**Guard (deterministic, no LLM):** the ladder keeps the previous attempt's raw
text. An accepted repair is compared against it: if the repaired script retains
less than ~75% of the prior attempt's total word mass, the acceptance is
REFUSED, recorded as a `REPAIR_OVERREACH` attempt outcome, and the ladder
continues to its next rung. Word-mass ratio, not a word-count target -- this
does not chase word count (operator rule); it refuses only gross deletion.
Panel is asked to pressure-test the threshold and the metric.

## Recommendation under review

**B + guard as the core; D and E as free hardening; A rejected; C held as
fallback.** Wiring points: dual-framing note joins the existing
defect-class-note mechanism at `_otr_scifi_fable2.py:1731-1738` (cast roster is
in scope -- `cast_names` at `:1700`); the guard wraps the accept branch at
`:1701-1719`; E lands in `_otr_fable2_markup.py:52-109`; D in the format
example.

Live verification: re-run the two dead legs (`ltx_audio_in`, `viz_mxc_cpu`).
Both writer failures are admissible to `PROD_BUG_LOG.md` (live headless-run
evidence). Unit fixtures reproduce last night's shapes; the PBUG entries cite
the live logs, not the fixtures.

Out of scope here: the `OTR_CastLock` freeze cascade (`wan_ti2v`) and the
`OUTPUT_TRUNCATED` 16384-token slot misconfiguration -- separate causes,
separate work.

## Questions for the panel

1. Is B-plus-guard actually the most robust shape, or does C's structural
   bound on LLM reach beat prompt-level framing badly enough to justify its
   machinery NOW rather than as a fallback?
2. Is raw-text word-mass >= 75% the right guard metric, or should it compare
   parsed dialogue words against a deterministic count of plausible
   `NAME: payload` lines in the failed attempt? Name a better threshold if the
   answer is a number.
3. Does the dual-framing note risk teaching the model to delete lines it
   merely cannot attribute -- and if so, what wording pins "when in doubt,
   attribute to the narrator/announcer rather than delete"?
4. E claims to be rule-completion rather than pattern-chasing. True or
   rationalization?
5. What is the cheapest live proof that the guard itself works -- i.e. that a
   deletion-heavy repair is actually refused?
