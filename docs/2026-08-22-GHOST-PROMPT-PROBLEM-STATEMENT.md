# Problem statement: how Ghost Signal's prompts are built, and what is wrong with them

**Date:** 2026-08-22
**Branch:** `v2.0-alpha`
**Status:** problem statement. Nothing is built, no prompt behaviour is changed.
**Trigger:** the operator saw a real composed prompt and said *"is that prompt?
omg that's ugly, we need a real prompt"*, then asked for a written explanation of
how these prompts and inputs are created for this model.

---

## 1. What actually gets sent, today

One real composed prompt, `archival_documentary`, character beat, 220 of a
320-character ceiling:

```
archival documentary.                          <- 1. pack cue      STYLE   (21 chars)
a tall stooped archivist in a worn cardigan,   <- 2. subject       LEDGER
mid-shot or wider, whole figure legible,       <- 3. framing lock  ROLE
reaches for the reel canister,                 <- 4. action        LEDGER
resolute mood,                                 <- 5. emotion       LEDGER  (traits)
climax,                                        <- 6. story accent  LEDGER  (arc_phase)
steady legible silhouette, one clear action, unbroken shot   <- 7. shot law  CONSTANT
```

Negative:

```
text, watermark, caption, lettering, subtitles,          <- constant lettering defence
oversaturated, glossy, clean digital, plastic skin,      <- the STYLE pack's negative_tail
waxy skin, sterile studio lighting, cartoon, illustration
```

Seven slots, comma-joined, in a fixed order. `slots` is returned as a receipt so
a published episode can be asked what its prompt was made of.

---

## 2. Where each input comes from

| Slot | Source | Notes |
|---|---|---|
| pack cue | visual style pack | `compact_style_cue()`, front-anchored, additive only |
| subject | ledger, via `distill_subject_sigil` | *"the one slot that differs most"* per the code |
| framing lock | ROLE, via `GHOST_FRAMING` | character = mid-shot floor; protects faces |
| action | ledger `motion_clause`, else the pack's `motion_registers` | |
| emotion | ledger `traits` | |
| story accent | ledger `arc_phase` | entered RAW, trimmed to 48 chars |
| shot law | constant | never trimmed |

**What it may never contain**, and this is enforced by not being a parameter at
all: raw dialogue, the episode title, the M4 scene wall, a second person, or
proper-noun metadata. The docstring is explicit that keeping them out of the
signature is *"the only way to guarantee it"*. Any redesign must keep that
property.

**Budget:** `GHOST_PROMPT_MAX_CHARS = 320`. Over budget, slots are dropped in
`GHOST_TRIM_ORDER = ("story_accent", "emotion", "framing")` — except `framing`
is protected on character beats, because dropping the mid-shot floor is how this
lane starts trying to render faces again.

---

## 3. FINDING 1 -- Ghost uses 21 characters of a 262-character style pack

This is the big one, and it is the direct cause of the operator's reaction.

Each visual style pack carries curated visual language. For
`archival_documentary`:

| Field | Chars | Content |
|---|---|---|
| `positive_tail` | 116 | archival documentary still, careful restoration texture, tactile paper and film materials, grounded natural lighting |
| `broadcast_tail` | 57 | broadcast-history atmosphere without futuristic equipment |
| `image_grade_tail` | 50 | subtle archival patina, clean readable composition |
| `era_tail` | 39 | grounded archival documentary aesthetic |

**Ghost emits none of it.** `_prefix_pack_cue` calls `prefix_style_cue`, which
uses `compact_style_cue(vstyle)` — the bare token, `"archival documentary."`,
21 characters.

**The still-image lane DOES use the rich fields.** `otr_meta_brief_image_prompt.py`
appends `image_grade_tail` and reads `era_tail`. So the same style pack gives the
still lane a fully dressed prompt and the video lane a label.

The prompt above is 220/320. There are **100 characters of unused headroom** and
`image_grade_tail` is 50 of them.

**Open question:** should Ghost consume the richer tails, and in what order
against the existing slots? Note the trim order would have to be revisited: a
richer cue that pushes past 320 would start evicting `story_accent` and
`emotion` on every beat, which trades one problem for another.

---

## 4. FINDING 2 -- a raw story word is not a visual instruction

Slot 6 emits `arc_phase` verbatim: the literal word `climax`, or `setup`, or
`resolution`.

To a diffusion model those are close to noise. `_ARC_CLAUSES` at
`render_driver.py:1524` already exists as a table that maps an arc phase to an
authored CLAUSE, and Ghost does not use it — it receives the raw value as
`story_accent` (`render_driver.py:2807`).

Worse, `story_accent` is FIRST in the trim order, so on the crowded bookend
beats where budget bites, the arc contribution is thrown overboard anyway. It is
simultaneously the weakest slot and the first to go.

---

## 5. FINDING 3 -- the announcer and music action slot emits a truncated fragment

On non-character beats the action slot renders as:

```
moves with open the episode and orient the
```

The beat's `beat_intent` is jammed into an action phrase and cut mid-clause,
leaving a dangling article. This is the same defect class fixed this morning in
the TRIM path (`_DANGLING_TAIL_WORDS`), which never got applied to the INTENT
path.

This is a correctness defect, not a taste one: prompt budget is being spent on a
fragment ending in "the".

---

## 6. What is NOT wrong with it, and should not be "fixed" casually

**The tag-soup style may be correct for this model.** SD1.5 was trained on
alt-text and tag-like captions; comma-separated attribute lists are its native
idiom. Flowing natural-language prose is an SDXL/Flux-era habit and frequently
performs WORSE on SD1.5. So "make it read like a sentence" is an aesthetic
instinct that may cost image quality, and any redesign should be A/B'd rather
than assumed.

**The structure earns its keep.** The slot list is a receipt; the trim order is
deliberate; the framing lock protects face continuity; the never-parameters rule
is what stops dialogue leaking into the picture. A rewrite that produces prettier
text and loses those is a regression.

**The negative is doing real work.** The lettering defence
(`text, watermark, caption, lettering, subtitles`) is unconditional because SD1.5
volunteers lettering into anything resembling a sign or a dial.

---

## 7. Measured context

Style packs steer saturation hard, by instruction, in the NEGATIVE:

* `archival_documentary` negative leads with **oversaturated** -> measured
  SATAVG 0.61 to 2.50 across four episodes.
* `anime` negative contains **muddy color** and bans photorealism -> measured
  SATAVG 5.13 and 51.97 on two episodes.

So the archival look is desaturated *on purpose*, by the pack, not by the
domain adapter. The 10x gap between the two anime episodes is unexplained and
is the largest measured difference of the day.

---

## 8. The questions worth answering

1. Should Ghost consume `positive_tail` / `image_grade_tail` / `era_tail`, and
   what gets evicted to make room?
2. Should `story_accent` emit `_ARC_CLAUSES[arc]` instead of the raw word — or
   be dropped entirely, given it is first to be trimmed anyway?
3. Fix the dangling intent fragment (uncontroversial; it is a defect).
4. Is tag-soup or prose better ON SD1.5 AT 512x288? This is measurable now that
   `OTR_WRITER_SEED` can hold the script still, and it should be measured rather
   than argued.
5. Does the 320-char ceiling still make sense if the cue gets richer? It was
   chosen for CLIP's 77-token window; the arithmetic should be re-checked rather
   than inherited.

---

## 9. Constraints on any answer

* Never admit raw dialogue, episode title, scene wall, second person, or
  proper-noun metadata into the composer's signature.
* The `framing` slot stays protected on `character_video`.
* The lettering negative stays unconditional.
* The golden lane's prompts must not change without the operator seeing a
  before/after on the same seeded script.
* `workflows/otr_canonical.json` is untouched by any of this.
