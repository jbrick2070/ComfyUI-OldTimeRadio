# RESEARCH PLAN -- how to get Cockney out of each voice engine

**Status:** research plan for operator review. Not a build spec.
**Date:** 2026-08-10. **Companion:** `docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md`.

---

## THE SPINE OF THIS PLAN

The seven char-voice engines (`nodes/cast_lock.py:44-46`) are not seven problems.
They are **two problems**, and one of them is already solved.

| kind | engines | how accent is obtained |
|---|---|---|
| **PROMPTED / catalogue** | `google_tts`, `elevenlabs`, `kokoro`, `bark` | you pick a voice, and on some you also steer with a style instruction |
| **REFERENCE-CLONE** | `indextts2`, `chatterbox`, `dia` | the engine imitates a reference WAV you supply -- **the accent comes from the clip, not from a setting** |

**The whole plan turns on that second row.** A clone engine will speak in
whatever accent its reference clip has. So we do not need to *find* a Cockney
voice three more times. We need **one** good Cockney clip and then feed it to all
three.

**And we already have one.** `output/otr/episodes/voice_audition_cockney/2_algenib_cockney.wav`
-- operator-approved 2026-08-08, with `3_algenib_cockney_angry.wav` proving the
accent survives emotional delivery.

> **THE CENTRAL HYPOTHESIS TO TEST:** use the proven `google_tts` / Algenib
> Cockney render as the CLONE REFERENCE for indextts2, chatterbox and dia. One
> audition propagates to four engines instead of one.

That is cheap to test, and it is the difference between a twenty-minute job and
four separate audition sittings.

## PER-ENGINE PLAN

### 1. `google_tts` -- SOLVED, just needs recording
Proven 2026-08-08: voice `Algenib`, `gt_algenib` in the bank, timbre already
`['gravelly','authority']`, gender measured at 97.2 Hz. Accent held under an
emotional line, with a control voice for comparison.
**Action:** none research-wise. Write the receipt (open plan, step 3).
**Research question:** was the Cockney obtained from the VOICE alone or from a
style instruction in the request? If a style prompt did the work, that prompt is
part of the receipt and must be recorded verbatim -- otherwise the result is not
reproducible and the receipt is decorative.

### 2. `indextts2`, `chatterbox`, `dia` -- the clone trio, and the real prize
These take a reference clip. `indextts2` is the **current production character
engine** (canonical nodes 80/81), so this is where the win actually lands.

**Test A -- synthetic reference (do this first, it is nearly free).**
Feed `2_algenib_cockney.wav` in as the reference on each of the three. Render the
same two audition lines. Judge: does the Cockney survive the clone? Clone engines
vary in how much accent they carry versus how much they normalise toward their
training distribution -- **that is exactly the unknown worth measuring**, and no
amount of reading settles it.

**Test B -- if the clone washes the accent out.** Try a longer reference (clone
quality usually improves with 10-30s of clean speech), and try the *angry* take
as reference, since stronger prosody sometimes survives cloning better.

**Test C -- fallback.** Search the ~40 existing refs per engine for the closest
gravelly male British-leaning candidate and accept "contradiction-ban" compliance
(never American, never fighting the phrasing) rather than positive Cockney.

**The question that must be settled BEFORE Test A ships anything:** Google's
terms on using generated audio as a reference/voice-cloning input. This is a
LICENSING question, not a technical one, and it is the single blocker that could
kill the whole spine. Note it is *not* the consent problem that would apply to
cloning a real identifiable person from found audio -- this is synthetic output
we generated ourselves -- but "we generated it" does not automatically mean "we
may use it as cloning input." **Settle this first.**

### 3. `elevenlabs` -- most likely to have a native answer
Has the largest curated voice catalogue of the seven and explicit accent
metadata. **Research:** does the library contain a gravelly male
Cockney/Estuary/London voice at 50s? If yes this needs no cloning at all.
Secondary: ElevenLabs supports voice-design-from-description on some tiers --
check whether that is available on this account before planning around it.
Cost note: this is a paid per-character lane; it is not the default engine and
should not become one for a cameo.

### 4. `kokoro` -- British, probably not Cockney
Ships British-tagged presets (`bm_*`). British is not Cockney, but under the
corrected floor (accent = contradiction ban, not phonetic requirement) a gravelly
British male may be entirely acceptable.
**Research:** audition the male British presets for vocal weight. This is the
cheapest engine to settle because the candidate set is tiny.

### 5. `bark` -- currently pinned, weakest control
`lemmy_row()` pins `v2/en_speaker_8`. Bark's accent control is the preset plus
whatever the text implies; it has no reference input and no style parameter.
**Research:** `scripts/bark_preset_audition.py` ALREADY EXISTS in the repo --
find out what it was built for and whether it produced results anyone kept. Then
audition the English-native presets for gravelly-male-British.
**Expect to accept "not Cockney but not contradictory" here**, and note bark is
not the production engine, so this is low priority despite being the current pin.

## WHAT TO MEASURE, SO THE ANSWERS ARE COMPARABLE

Every candidate, every engine, the same protocol -- this is what makes it an
audition rather than a listening session:

1. **The same two lines**, verbatim, on every candidate: one neutral, one
   emotionally active. (The 2026-08-08 audition already used this shape. **The
   line text is not recorded anywhere in the repo -- recover it from the operator
   or re-choose it now and freeze it.**)
2. **A control** on each engine -- one voice you expect to be wrong -- so
   "everything sounds fine" is detectable as a failure of the ear rather than a
   success of the voice. The 2026-08-08 set did this with `4_charon_plain_control.wav`.
3. **Measured median f0** for gender/weight, as `gt_algenib` already carries.
   Cheap, objective, and it caught nothing subjective.
4. **Operator verdict recorded per candidate**, not just for the winner --
   knowing what was rejected is what stops the next person re-auditioning it.

Fill `QUALIFICATION_RECEIPT_REQUIRED_FIELDS` (shipped `3864f517`) for whatever
wins. The structure exists; it wants data.

## ORDER, AND THE COST

| # | Do this | Cost | Unblocks |
|---|---|---|---|
| 1 | Settle the Google-output-as-clone-input licensing question | reading | the entire spine |
| 2 | Record the existing google_tts audition as a real receipt | minutes | one engine done properly |
| 3 | Test A -- Algenib Cockney clip as reference on indextts2 | ~20 min | **the production engine** |
| 4 | If A works: repeat on chatterbox + dia | ~20 min | three more engines from one audition |
| 5 | kokoro British presets | ~15 min | cheap, small candidate set |
| 6 | elevenlabs catalogue search | ~15 min | may need no cloning at all |
| 7 | bark presets, and read `scripts/bark_preset_audition.py` first | ~15 min | lowest priority; not production |

**Steps 3-4 are the ones that matter.** Everything else is tidying.

## THE HONEST RISK

The spine may simply not work: clone engines may normalise the accent away, and
if they do, there is no cheap path and each clone engine needs its own reference
sourced some other way. **Test A settles that in twenty minutes**, so run it
before planning anything downstream of it.

Second risk: this is all downstream of a licensing answer nobody has yet. If
Google's terms forbid using output as cloning input, the spine dies and steps 3-4
are replaced by "source three Cockney references some other way" -- a materially
bigger project.

## WHAT THIS PLAN DOES NOT DO

It does not touch the writing. Lemmy's dialect lives in word choice and rhythm,
and that layer is engine-invariant and already governed. No engine result changes
a line of his dialogue.
