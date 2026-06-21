# Story quality R2 -- Claude's own CREATIVE pass (what makes the story BETTER, in general)

Grounded by close-reading the 13h corpus: WHY the opus story ("The Taste of Nickel") is genuinely
good and the weak ones aren't -- then the general levers to lift EVERY story, model-agnostic. This
is the creative layer ON TOP OF the pass01 structural fixes (music/non-dialogue bleed, announcer
close, cliche/stage-business gate). The scaffolding already gives structure; these make it ALIVE.

## What the strong story does that the weak ones don't (diagnosis)
- **SPECIFICITY, not category.** Opus: "Ferry Lane", "the pumphouse", "the green book", "the Hadley
  boy, he's twelve", "the Vassek widow". Weak: "the system", "the perimeter", "the lab", "the
  windows". Proper nouns + physical objects make it REAL; generic nouns make it forgettable.
- **A CENTRAL OBJECT / motif the story orbits.** Opus: the COUNT / the green ledger book -- it is
  evidence, weapon, headstone, and the final image, all at once. Weak stories have no object; they
  drift. One concrete object that recurs and CHANGES MEANING is the spine of a good short.
- **ESCALATION in concrete terms.** Opus climbs: one cough -> forty dead -> they took the jars ->
  the lab's locked with someone else's key -> two bottles of saline left. Weak stories stay FLAT --
  the same "secure the area / check the windows" pressure the whole way.
- **DISTINCT VOICES.** Opus's Doctor (clipped, defiant, counting) vs Erin (anxious, pleading) read
  as two different people. Weak casts are INTERCHANGEABLE -- swap the names and nothing changes.
- **SUBTEXT / indirection.** Opus characters talk AROUND the thing (the river, the count, the
  company's lawyers) -- the fear is under the line. Weak ones STATE it flat ("We're playing with
  fire").
- **An IMAGE ending, not a thesis.** Opus closes on "lights on, book open on the desk... the answer's
  right here, dated, in my hand." Weak closes TELL the moral ("reminding us to guard the families").

## The general levers (creative additions to round 2)
These are the high-leverage, ledger-safe, model-agnostic ways to force the weak end toward what opus
does naturally. Each is a candidate for the coding/wiring passes.

1. **SPECIFICITY ANCHORS (highest leverage).** At story-setup, derive 3-5 CONCRETE anchors from the
   news brief + setting -- proper place names, a physical object, a number, a named bystander -- and
   REQUIRE the line composer to use them (and the critic/gate to reject lines that stay generic).
   This is the single biggest opus-vs-weak gap.
2. **A CENTRAL STORY OBJECT.** Add a derived `central_object` to the dramatic state (e.g. "the green
   ledger", "the sealed jars") that the FIRST act introduces, the MIDDLE complicates, and the CLOSE
   lands as the final image. One object, three meanings. (Pairs with the announcer-close
   final-image contract from chunk 2.)
3. **ESCALATION CONTRACT per act.** Each act's beat-objective must RAISE the concrete stake over the
   last (a bigger number, a closer threat, a higher cost) -- a per-phase "must escalate vs the prior
   phase" check, deterministic where possible, so weak models can't run flat.
4. **VOICE DISTINCTNESS, enforced.** F5's `speech_signature` exists but is weakly applied -- promote
   it to a hard per-line constraint (each speaker keeps their register/rhythm) AND derive CONTRASTING
   signatures at cast time (clipped vs verbose, plain vs ornate) so two characters never sound alike.
5. **SUBTEXT NUDGE (lighter touch).** A line-prompt instruction: "imply the pressure, don't name it"
   for high-tension beats -- and a gate that flags on-the-nose emotion statements ("I'm scared",
   "this is dangerous") for a targeted reroll. (Use sparingly -- weak models can over-correct into
   vagueness; keep it to the turn/climax beats.)

## How this composes with pass01 (one coherent round 2)
- Structural (pass01, do FIRST -- kills the universal warts): music_inter caption suppression;
  announcer-close final-image contract + thesis-phrase reroll; the cliche/stage-business reject gate
  + opposed-wants-into-the-line-prompt.
- Creative (this pass, the lift): specificity anchors; central object; escalation contract; voice
  distinctness; subtext nudge.
- Shared spine: ALL of it rides the EXISTING ledger {cast,lines,meta} (content-only), is MODEL-
  AGNOSTIC (it forces the weak end without hurting opus -- opus already passes the gates), uses
  TARGETED rerolls + a couple of cheap setup-time LLM calls (specificity anchors, central object,
  contrasting signatures), and chases CRAFT not word/beat count.

## Anti-regression (protect opus)
Every gate must be one opus already passes (opus uses proper nouns, has an object, escalates, has
distinct voices) -- so the strong-model output is untouched and only the weak end is lifted. The
re-soak QA compares the four structural metrics PLUS new craft signals (proper-noun density, central-
object recurrence, per-act stake escalation, voice-distinctness) before/after, on a weak-local +
a frontier leg.

## Sprint shape (the remaining passes the operator asked for)
- PASS 02 (CODING): turn pass01 + pass01b into exact seams -- which functions, which prompts, the new
  setup-time LLM calls (specificity anchors / central object / contrasting signatures), the
  deterministic gates, the targeted-reroll plumbing -- each a green chunk.
- PASS 03 (WIRING): confirm NO workflow-JSON/node change is needed (content-only inside
  OTR_LedgerScriptWriter + its modules), or pin the exact node/widget if any.
- PASS 04 (FINAL QA): the re-soak design + the before/after craft-metric scan + the opus-no-regress
  gate, so a green re-soak closes the campaign.
