# OPEN PLAN -- making LEMMY work across engines

**Status:** OPEN PLAN, not a build spec. For operator review, then a Codex pass.
**Date:** 2026-08-10. **Inputs:** Lemmy r1 (2026-08-08, 1 lane), r2 (2026-08-10,
2 lanes), a cold Fable design ruling, and a repo-wide search that **found the
audition everyone believed had never happened.**

---

## 0. THE TWO THINGS THAT CHANGED TODAY

**A. THE AUDITION EXISTS. It was run 2026-08-08 and never written down.**

`output/otr/episodes/voice_audition_cockney/` holds four takes:

| file | what it is |
|---|---|
| `1_algenib_plain.wav` | baseline, no dialect direction |
| `2_algenib_cockney.wav` | the Cockney read |
| `3_algenib_cockney_angry.wav` | **Cockney held under an emotional line** |
| `4_charon_plain_control.wav` | a CONTROL voice |

The operator listened and approved: the accent held, *including at emotion*.
This is a properly-shaped audition -- baseline, target, stress case, control --
and it has been sitting on disk unrecorded while every plan document asserted
that nothing had ever been auditioned. **The prior claim "no voice on any engine
is audition-proven Cockney" was FALSE.** It is corrected here.

The voice is `gt_algenib` (`config/voice_reference_bank.json:2823`):

```
engine        google_tts          provider_voice_id  Algenib
gender        male                measured_median_f0_hz  97.2
timbre        ['gravelly','authority']               age_band  adult
quality_tier  a                   commercial_clean   True
ref_path      cloud:google_tts:Algenib
gender_source measured_median_f0_2026-08-08
```

Its `timbre` already reads `gravelly` -- Lemmy's floor attribute -- and its
gender was MEASURED (97.2 Hz) on the audition date, not asserted.

**B. THE SCOPE BAR IS LIFTED.** Operator, 2026-08-10: *"we can build a million
new nodes, there are no rules."* r1's "no new node or widget surface" constraint
is withdrawn. This does NOT automatically make a new node the right answer --
see section 3 -- but it stops being a blocker.

## 1. THE PROBLEM, RESTATED CORRECTLY

The debate has been "which engines may render Lemmy, and how do we suppress him
where they can't." **Fable's grounded finding is that this is the second-order
problem, and the first-order one is worse:**

> On the engine production actually runs, Lemmy has no persistent identity at
> all. He is drawn fresh on **gender alone**, with a different roll every episode.

Chain, verified: the canonical workflow runs the character bus on `indextts2`
(nodes 80/81). `lemmy_row()` hard-pins him to **bark**, so on indextts2 that pin
is dead weight and he falls through to `assign_voice_for_slot`, which folds the
per-episode `episode_seed` into the draw. His slot facts arrive EMPTY because
pre-locked rows have no ensemble slot (`_otr_casting.py:1828-1838` -- the code
comment says so). So he is drawn from the male refs on gender alone, freshly,
every appearance.

**He is not "a Cockney rendered imperfectly." He is a different man each time.**
The accent question sits on top of an identity failure.

## 2. THE RULING -- PIN, DON'T SUPPRESS

Fable rejected all three options the r2 panel offered and named a fourth.
**Suppression is deleted from the design as a concept.**

Lemmy resolves to exactly ONE voice identity **per engine**, at CastLock:

1. **Qualified route** from `LEMMY_VOICE_POLICY.approved_native_routes[engine]`
   if one exists (validated by `is_qualified_route`, shipped `3864f517`).
2. **Otherwise a deterministic floor-scored pick keyed on a STABLE constant**
   (name + engine + policy version), **never the episode seed** -- so on any
   given engine he is the same man every time, from the first change onward,
   before any further audition.
3. The only hard stop is the one that already exists: `assign_voice_for_slot`
   raises `VoiceCastingError` when an engine has no castable male reference at
   all. That backstop is built and fail-closed; nothing new stacks on it.

**Why suppression was wrong:** a character who appears 11% of the time and gets
suppressed on the production engine *silently stops existing* -- nobody notices a
cameo that did not happen. And a hard render-killer on a surprise easter egg is
exactly the kind of guardrail this project has been stripping out.

## 3. THE FLOOR, CORRECTED

Fable's ranking of what actually carries a recurring radio character's identity:

1. **Verbal habits + function** -- the writer's engine-proof layer. Carries most
   of the recognition. Already settled and already right.
2. **Same-voice-WITHIN-engine persistence** -- **the floor's one real omission**,
   and precisely what production violates today.
3. **Gender + vocal weight** -- contradiction is instantly fatal. Note the code
   currently drops `gravelly` and `50s` even though `LEMMY_PROFILE` declares
   both: a confirmed bug under the settled floor's own terms.
4. **Age band.**
5. **Accent-as-rendered** -- the LEAST load-bearing across TTS engines, because
   the accent's weight already sits in word choice and rhythm. Read the "London
   working-class" clause as a **contradiction ban** (never American, never a
   voice that fights the phrasing), not a positive phonetic requirement. A
   neutral-British gravelly voice speaking Lemmy's written rhythms reads as
   Lemmy; a forced music-hall Cockney reads as a caricature of him.

**Even with new nodes allowed, do NOT build engine-conditional WRITING.** The
writing is the engine-invariant layer -- the one part of Lemmy that must never
fork per engine. That argument was never about scope; lifting the bar does not
revive it.

## 4. THE TENSION THIS PLAN DOES NOT HIDE

**The one proven route is on an engine production does not run.** The audition is
`google_tts` / `Algenib`; the canonical character bus is `indextts2`. And
`BatchCharacterVoices` exposes ONE engine for the whole bus, so Lemmy cannot be
routed to google_tts while everyone else stays on indextts2 -- not without the
per-character engine routing that does not exist today.

So the audition qualifies a route that today's graph cannot reach for him.
Options, for the operator:

* **(i)** Accept tier-2 determinism on indextts2 (he becomes consistent
  immediately, unauditioned), and register the google_tts route so it activates
  the day that engine is used.
* **(ii)** Spend one more sitting auditioning the male gravelly indextts2 refs --
  Fable estimates six to ten candidates, two fixed lines, roughly twenty minutes.
* **(iii)** Build per-character engine routing so Lemmy can take google_tts while
  the bus runs indextts2. **Now permitted** by the no-rules ruling, but it is the
  largest option and it makes one character special in the renderer.

**Recommendation: (i) now, (ii) when convenient, (iii) only if a listener can
tell the difference.** (i) fixes the first-order defect this week at near-zero
cost; (ii) buys the last mile where listeners actually are.

## 5. IMPLEMENTATION ORDER

**Lands first, ONE change, ONE declared re-baseline:**

1. **Feed Lemmy's slot facts** into `cast_voice_slots`
   (`_otr_casting.py:1828-1838`) from `LEMMY_PROFILE`: gender male, timbre
   `["gravelly"]`, age_band `"50s"`. The fields already exist and are simply
   unread. Pure bug fix under the settled floor.
2. **Stable identity key** for pre-locked recurring rows in
   `assign_voice_for_slot`: replace `episode_seed` with a constant key
   (name + engine + policy version). Assign him BEFORE the open slots so his
   exclusion-set ripple is deterministically ordered. Stamp the resolution
   (`pinned` / `deterministic_floor`) into the cast report and ledger meta.
3. **Register the REAL audition** as a qualified route: `google_tts` /
   `Algenib`, receipt filled from the four wav files (paths + sha256, both
   audition lines, the operator verdict, the 2026-08-08 date). The structure
   shipped in `3864f517` and is waiting for exactly this.
4. **CastLock resolves** `approved_native_routes[engine]` through
   `is_qualified_route` ahead of the deterministic pick; bark's hard pin migrates
   out of `lemmy_row()` into the route map -- `cast_pools.py` already says in its
   own comment that these are the same fact stated twice. Grep every reader of
   `lemmy_row()`'s `tts_model` first.

**Deferred, operator-paced:** further auditions, one engine at a time, each
upgrading that engine from tier 2 with no flag day and no partial-rollout hazard,
because tier 2 already holds the floor everywhere.

**Never built:** engine-conditional writing; suppression in any form; a new hard
fail on unpinned engines; phonetic respelling.

## 6. THE RE-BASELINE

Changing his casting moves the roll and breaks replay parity. **Fable's argument
that this makes the change MORE urgent, not less:** the perturbation is confined
to `lemmy_hit` episodes (~11%), which are exactly the episodes that are already
defective -- a re-baseline whose blast radius is precisely the broken set. And
the cameo roll is ALREADY `SystemRandom`, deliberately outside seed replay
(BUG-LOCAL-260), so Lemmy is already the declared exception to replay purity.
Bend it once, deliberately, declared, in the same change that makes him a
character.

## 7. THE STRONGEST ARGUMENT AGAINST THIS PLAN

Tier 2 ships an **unauditioned voice as Lemmy's standing identity**, laundered
through determinism -- and the operator's own 2026-08-10 cleanup was "a route
that cannot prove it was auditioned has not been auditioned." If the stable pick
lands badly, Lemmy is now *reliably* wrong on that engine, listeners learn the
wrong voice as him, and the eventual pin becomes a noticeable recast. Today's
chaos at least means no wrong voice has squatter's rights.

Held anyway, because a consistent near-miss beats a nightly stranger for serial
identity, the recast happens at most once per engine, and the ledger stamps the
unpinned state loudly so it cannot hide. But it is a real cost.

## 8. OPEN QUESTIONS FOR THE OPERATOR

1. Which of (i)/(ii)/(iii) in section 4?
2. Do you remember the two audition LINES? The receipt wants them verbatim; the
   wavs are on disk but the text is not recorded anywhere found.
3. `4_charon_plain_control.wav` -- was charon rejected, or untested? It matters
   only for the record.
4. Confirm the re-baseline in section 6 is declared and accepted.
