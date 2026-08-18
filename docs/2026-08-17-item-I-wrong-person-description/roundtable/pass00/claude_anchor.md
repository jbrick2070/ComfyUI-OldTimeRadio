# Claude anchor review -- R1, item I (the driver reviewing its own plan)

**I WROTE THE PLAN UNDER REVIEW.** So this anchor is a self-critique, and its
value is naming the weak joints before the panel finds them -- and, more
importantly, marking which claims are CONFIRMED at the files versus assumed, so
a panel claim contradicting a CONFIRMED one can be discarded on evidence rather
than on authority.

## VERDICT: the diagnosis is sound and CONFIRMED; the REMEDY section is the weak half and the SIZING is weaker still.

## CONFIRMED at the files (a panel claim contradicting these needs real evidence)

1. **The pitch must name.** `_otr_original_radio` -- `SelectCastEntry` and
   `CastSketchEntry` both carry `name: str`, and the validator returns
   `"empty cast name"` / `f"pitch {i}: empty cast name"`. **CONFIRMED.**
   Consequence: "make the pitch stop naming" is not a one-line option.
2. **The prompt shows both authorities with no precedence.**
   `_otr_casting.build_description_prompt`: `story_text = brief` when
   `casting_brief` is non-empty, emitted as `f"Story: {story_text}"`, and the
   assigned name as `f"Name: {name}"`. No sentence anywhere states which wins.
   **CONFIRMED.**
3. **The free-text slot exists and invites a name.** The same function's
   contract literal: `'Format: "<age decade>, <story-linked role>. Face: ...'`.
   **CONFIRMED.**
4. **The brief carries the pitch's names in real data.** The reported episode's
   `meta.news.casting_brief` reads *"We need a seasoned yet vulnerable LUCILLE
   PENNY..."* while `cast` rows are `RICK STEINER` / `NIA PHILBIN`.
   **CONFIRMED** by reading the shipped ledger.
5. **The prior-cast theory is dead.** `LUCILLE PENNY` appears as a cast row in
   **zero** of 1,710 ledgers, so `_format_prior_entry` cannot be the path.
   **CONFIRMED** by corpus scan.
6. **The news lane is not the culprit.** `news_interpreter` asks `casting_brief`
   for *"who belongs in the story: occupations, dynamics, stakes"* -- names are
   never requested. **CONFIRMED.**

## MUST-FIX in my own plan

1. **THE SIZING IS NOT DECISION-GRADE, and the item should not be built on it.**
   I report two floors (28 rows / 20 ledgers; 18 rows / 14 ledgers) with
   different hit sets and an explicitly uncomputed union. Nobody should approve
   a build on "somewhere north of 28". **Worse, I never checked whether the rate
   is FALLING** -- the gender-assignment regime changed within the corpus window
   (item G established that), and if this defect is already decaying the correct
   answer may be a detector plus a watch, not a build. *Fix: compute the union
   and bucket by episode date before choosing a shape.*
2. **OPTION A RESTS ON A MAPPING I HAVE NOT VALIDATED, and the one data point I
   have suggests it is BROKEN.** `meta.continuity.facts` pairs
   `"Lucille (Nia Philbin)"` and `"Harold 'Hal' Bright (Rick Steiner)"` -- while
   the DESCRIPTIONS are crossed the other way (RICK STEINER holds LUCILLE's
   prose). So either the mapping is wrong, or the descriptions are, or they were
   produced by two independent and disagreeing reconciliations. **I tabled A as
   a live option without establishing which.** That is the single biggest hole
   in the plan. *Fix: settle it before A is costed at all.*
3. **I MAY BE MERGING TWO DEFECTS.** "Row carries another NAME in its identity
   slot" and "row carries another person's whole face/voice prose" are treated
   as one thing throughout. The RICK STEINER row shows both at once, which is
   exactly the case that cannot distinguish them. *Fix: find a row exhibiting
   only ONE, or stop claiming they are the same defect.*

## SHOULD-FIX in my own plan

1. **The consumer list is asserted, not enumerated.** I claim the blast radius is
   `visual_plan.characters[NAME].portrait_prompt`. I have not enumerated every
   consumer of `character_description` (voice-fit, TTS, captions, credits, shot
   direction all plausibly read it), so the cost of a repair is unpriced.
2. **Option E collides with THE LAW and I under-state it.** "Detect and repair"
   is fine; "detect and reroll" spends a live model call on an accepted episode
   and any refusal path is forbidden outright. The distinction deserves to be
   explicit rather than a parenthetical.
3. **Option B's loss is asserted as small without evidence.** Stripping
   `LUCILLE PENNY` to "a woman" may cost the brief its specificity. I read one
   brief, not a sample.

## UNVERIFIABLE from the files (verify-at-build)

* Whether the model would actually stop using the brief's name if precedence
  were stated (option D) -- only a render answers that, and item F just proved
  on this same repo that an existing "invent none" instruction did not hold.
* Whether any repair can be applied retroactively to frozen/published episodes,
  or whether this is forward-only.

## What I most want broken

That my prior (B+E) is right. It is the option I reached first, I have not
priced C honestly, and "strip the names and check afterwards" is suspiciously
convenient for a driver who wants a small diff.
