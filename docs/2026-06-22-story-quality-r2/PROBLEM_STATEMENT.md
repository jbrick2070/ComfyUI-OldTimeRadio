# Story quality -- ROUND 2 (grounded in the 2026-06-22 13h nightly soak output)

## Goal (operator, hard)
A genuinely BETTER STORY -- NOT word count (explicitly dropped). Improve CRAFT. Ledger
`{cast,lines,meta}` schema (l3-2026-05-14) stays FIXED / content-only; the audio spine is frozen;
the change may ADD LLM calls or UPDATE PROMPTS. It MUST be MODEL-AGNOSTIC -- it has to lift the WEAK
end (small local models) too, not just the frontier ones. Story-engine v1 (F1-F8) is already
shipped; this is round 2, grounded in the REAL output it produced over the last 13 hours.

## The corpus (14 real stories, opus -> local; all 18-line)
Strong frontier (opus "The Taste of Nickel"): genuinely good -- distinct voices, an escalating
environmental cover-up, a dated-ledger motif, a real ending. The craft scaffolding WORKS on a strong
model. The problems below are what's LEFT, and they get worse toward the weak end.

## Grounded craft issues (real quotes + code source)

### A. Music-interlude beats leak a PLACEHOLDER as a spoken/caption line (every episode)
`_otr_outline.py:1507-1519` mints a `music_inter` beat (speaker=NARRATOR, role=music_inter) whose
`intent` is the literal string **"Musical interlude bridging {phase_name} into the next phase."** --
and it renders VERBATIM as a transcript/caption line in EVERY episode:
- opus: line 9 "Musical interlude bridging setup into the next phase.", line 16 "...complication...".
- leg_0007: lines 20-21 BOTH placeholders dumped consecutively at the end.
- local retry: lines 20-21 same.
A music interlude should be MUSIC (no spoken/caption text), or carry a real, in-world cue -- never
this stage-direction placeholder. This is the single most visible, universal craft wart. Likely the
cheapest high-impact fix.

### B. Tell-don't-show META / "revelation" lines break immersion
The announcer close (and some inter lines) SUMMARIZE the news/theme instead of dramatizing it:
- leg_0007 line 19: "Tonight's revelation: The ancient plague safely secured for study, but no
  longer a threat."
- local retry line 19: "Tesla's throne is now shared, as Chinese batteries make their mark."
- opus line 21 (close): "...proving Robinson right too late, and reminding us to guard the families
  who live beside the gas." (a moral-of-the-story tag).
These read as a newspaper caption, not radio drama. The close should land an IMAGE or a final beat of
character, not a thesis statement.

### C. Weak / local models collapse to cliche + meandering stage-business
The opposed-wants + specificity scaffolding carries OPUS but NOT the weak end:
- Recurring cliche across weak stories: "You're playing with fire" (leg_0007 AND local retry),
  "We're not leaving anything to chance", "This changes everything".
- Meandering filler with no dramatic spine (local "The Loose Screw"): "I'll go check the perimeter",
  "I'll double-check the windows", "You know what? I'll just lock down the lab", "Mindy, I've got
  this. No need for haste." -- motion without conflict; the opposed wants never surface as a real
  clash.
The scaffolding sets up the drama but does not FORCE a weak model to honor it line by line.

## Questions for the panel (converge on the round-2 fix set, ledger-intact)
- Q1 [A]: Where is the cleanest fix for the music_inter placeholder -- make the beat carry NO voiced
  text (silent music only, caption suppressed), or replace the intent with a real in-world cue? What
  keeps the ledger contract + the audio/timing intact? (music_inter beats still need their timing
  slot for the master mix.)
- Q2 [B]: How to make the announcer CLOSE (and any meta line) dramatize rather than summarize -- a
  prompt change to the announcer composer? a constraint banning "Tonight's revelation/the lesson is"
  thesis phrasing? a final-image requirement?
- Q3 [C, the big one]: What lifts the WEAK end? Options to weigh: a per-line "make THIS line serve
  the opposed want + cut stage-business" pass (an added cheap LLM call), a cliche-scrub/ban-list,
  a stronger per-beat objective injected into the line prompt, or a critic-driven targeted reroll of
  the flattest lines. Which is highest-leverage AND model-agnostic AND ledger-safe?
- Q4: What should we NOT touch -- where is story-engine v1 already good enough (the strong-model
  output is genuinely good) so round 2 stays minimal and does not regress opus?
- Q5: Anti-goal check: none of this should chase word count or beat count; confirm the proposed
  changes are craft-only.
