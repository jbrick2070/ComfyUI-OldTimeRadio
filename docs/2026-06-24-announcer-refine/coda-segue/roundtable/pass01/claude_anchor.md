# R1 CLAUDE ANCHOR -- dynamic coda segue, creative coherence (grounded)

VERDICT: yes-with-fixes. The hybrid (dynamic happy-path + deterministic guardrails
+ fixed-phrase floor) resolves the operator/panel tension, but "dynamic but NOT
generic" and "teachable without a fixed phrase" each need a concrete mechanism or
the weak local model regresses to exactly the generic segue the operator dislikes.

## MUST-FIX BEFORE BUILD
1. [proposed/ask4] **"Not generic" must be FORCED by content reference, not adjectives.**
   A weak model's default under "write a segue" IS "And now, the real story...". Fix:
   the coda must NAME a concrete real-world specific drawn from `news_close_brief` (a
   real entity / number / place / date) and tie it to the episode's concrete final
   image. Make "contains a concrete news specific" a VALIDATOR requirement -- the
   anti-generic lever is the same gate as ask #1, not a style note.
2. [teachability/ask2] **Teach via a constant SHAPE, vary the words.** Pure freeform
   won't read as a recurring teaching beat. Recommend a fixed 2-clause SHAPE -- (a)
   the resonance of the fiction just heard, (b) the real fact -- with the TURN
   between them as the recognizable signal, words varying per episode. The taught
   thing is the pivot, not a tag. This is the middle path the operator is reaching
   for ("a segue between fiction and reality").
3. [ask1/ask3] **State the honest gate ceiling.** "Blend" is only PARTIALLY
   deterministically catchable: the `ending_change`-overlap check catches the
   obvious "restate the fiction as real" blend; it does NOT catch a subtle blend
   (the model inventing a plausible fake-real detail). So low-temp + the fixed-phrase
   FALLBACK FLOOR must carry the residual risk -- do not claim the gate fully
   prevents blends.

## SHOULD-FIX
1. [proposed] **Keep the fixed-phrase fallback floor and name it as the
   reconciliation.** Operator gets a dynamic, news-specific segue on the happy path;
   when validation fails (twice), the deterministic floor (a recognizable lead-in +
   `news_close_brief`) keeps every episode safe + teachable. This is what lets us
   say "yes" to dynamic without reopening the weak-model reliability hole.
2. [ask4] **Forbid a generic opener in the system prompt** (mirror the main
   campaign's "write only the fact body; no introductory phrase"), so the dynamic
   coda can't default to a throat-clear lead-in.

## OPTIONAL / NICE-TO-HAVE
- Seed-keyed micro-variation only inside the fixed FALLBACK (a small closed set), so
  even the safety floor isn't identical every time -- lower priority than the happy path.

## CUT
- Nothing. The design is already minimal (one LLM line + a gate + a floor).

## ASSUMPTIONS
- [ASSUMPTION] `news_close_brief` reliably contains a concrete real detail
  (entity/number/place) to name. If it is often vague/abstract, the "name a concrete
  news specific" requirement can't fire and the floor carries more episodes -- verify
  the brief's specificity on a soak sample; if weak, that is a news_interpreter
  upstream note, not a coda bug.
