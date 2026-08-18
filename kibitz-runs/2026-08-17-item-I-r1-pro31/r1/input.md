# Driver anchor -- item I, the wrong-person `character_description`

**Driver:** Claude (Cowork), sole judge. **Date:** 2026-08-17. **HEAD:** `08232661`.
**Panel:** Antigravity `Gemini 3.1 Pro (High)` + `Gemini 3.7 Flash (High)` + a
Fable pass. Codex is quota-held to 2026-08-19 and is **excluded** -- per the
operator's 2026-08-17 ruling a missing lane never blocks the arc, so this runs
without it and says so.

**CITE SYMBOLS, NEVER LINE NUMBERS.** 8 of 21 citations rotted within an hour on
2026-08-17. Every claim below names a function, constant or field and quotes it.

---

## 1. THE DEFECT, measured on published ledgers

A cast row's `character_description` describes a **different person entirely**.
The reported episode, `signal_lost_midnight_circuit_20260803_162229`, has **2 of
3 rows wrong**:

* row **RICK STEINER** (gender `male`): *"Late 50s, Seasoned yet vulnerable
  **LUCILLE PENNY**. Face: Oval, knitted brows ... Voice: Warm, calm urgency,
  like a mother in a storm."*
* row **NIA PHILBIN** (gender `female`): *"60s, **HAROLD 'HAL' BRIGHT**. Face:
  square jaw, heavy brow ... prominent forehead scar from a past mistake."*

**It reaches pixels.** `meta.visual_plan.characters[NAME].portrait_prompt`
carries the identical string, so the portrait is painted of the other person.

**Promoted already** as Bible `11.61` (survival-guide `ff0eb13`) --
PBUG-20260817-03. **The rule is banked; the CODE FIX is what this arc is for.**

## 2. ROOT CAUSE -- proven at the files, not guessed

**TWO AUTHORITIES NAME THE SAME CHARACTER, and a prompt is asked to referee.**

1. `_otr_original_radio` produces the pitch. Its `SelectCastEntry` /
   `CastSketchEntry` models carry a `name` field, and **the validator REJECTS an
   empty one** -- the failure strings are literally `"empty cast name"` and
   `f"pitch {i}: empty cast name"`. **So the pitch is STRUCTURALLY REQUIRED to
   invent names.** Measured in the reported episode:
   `meta.source_meta.selected_concept.cast` = `LUCILLE PENNY`,
   `HAROLD 'HAL' BRIGHT`.
2. That prose is restated in `meta.news.casting_brief`: *"We need a seasoned yet
   vulnerable LUCILLE PENNY, early 40s ... Our HAROLD 'HAL' BRIGHT should be a
   gruff, weathered man in his late 50s."*
3. The cast pool then assigns **different** names: `RICK STEINER`, `NIA PHILBIN`.
4. `_otr_casting.build_description_prompt` hands the model **BOTH**: the brief on
   the `Story:` line (`story_text = brief` when `casting_brief` is non-empty) and
   the assigned name on the `Name:` line -- **with no statement of precedence.**
5. Its own CHARACTER VISUAL CONTRACT reserves a free-text slot immediately after
   the age band: `"<age decade>, <story-linked role>. Face: ..."`. **A name is a
   plausible filler for "story-linked role" when the surrounding text keeps
   naming people.** The model is resolving an ambiguity we created.

**DISPROVEN, and the original log entry said it:** the `_format_prior_entry`
prior-cast-bleed theory. `LUCILLE PENNY` is not a cast row in that episode or
anywhere in the 1,710-ledger corpus, so the prior-cast echo cannot be the path.

**Note what is NOT the cause:** the news lane's own instruction is clean --
`news_interpreter` asks `casting_brief` for *"who belongs in the story:
occupations, dynamics, stakes"*, never for names. The names come from the
ORIGINAL lane's pitch, which is required to have them.

## 3. SCALE -- two detectors, two floors, union uncomputed

| detector | scope | hits |
|---|---|---|
| pitch-cast: a description contains a `selected_concept.cast[].name` no roster row owns | only the **124 of 1,710** ledgers that record a pitch | **28 rows / 20 ledgers (16%)** |
| name-shape: a person-name-shaped phrase in the identity slot the roster does not own | all ledgers | **18 rows / 14 ledgers** |

**The sets differ.** The second catches `the_wax_cylinders_whisper` (OYA SATO
carrying *"30s, Henry 'Hank' Griswold."*) and `nightshift_erasure` (RYAN KAPOOR
carrying *"60s, EDWARD 'ED' GRISWOLD."*) which the first misses; the first
catches LUCILLE PENNY which the second misses. **So both are floors and the true
total is unknown.** Computing it properly is part of this item.

**It survives a FREEZE:** `baked_ledger.json` carries the RYAN KAPOOR row in
fourteen copies.

**Gender-crossed in both directions** (RICK STEINER male <- LUCILLE PENNY; WENDY
PALMER female <- SIR REGINALD PENNYWORTH), which is why it accounts for part of
item G's portrait-conflict count -- and why item G's gender framing would have
LAUNDERED it.

## 4. THE FORK -- and every option has a real cost

The pitch cannot simply stop naming: its validator rejects an empty name, and a
pitch that says "a woman, 40s" writes a worse pitch. So somebody must reconcile.

| # | Shape | Cost / risk |
|---|---|---|
| **A** | **Rewrite the brief** -- map pitch names to assigned names and substitute before the brief enters any per-record prompt | Needs a reliable mapping. `meta.continuity` already contains `"Lucille (Nia Philbin)"` and `"Harold 'Hal' Bright (Rick Steiner)"`, so a mapping EXISTS somewhere -- but in the reported episode it is **CROSSED relative to the descriptions**, which is its own bug and must be understood before it is trusted |
| **B** | **Strip names from the brief** before the description prompt -- replace `LUCILLE PENNY` with the role words | Cheapest and lossy: "seasoned yet vulnerable LUCILLE PENNY" -> "seasoned yet vulnerable woman" is fine, but a brief built around two named people may read oddly |
| **C** | **Let the pitch NAME the roster** -- adopt the pitch's names as the cast names | Removes the conflict at the source and is the most honest, but collides with the cast pool, `10.08`'s coherence reconciliation, the gender ladder and replay determinism (`12.51`) |
| **D** | **State precedence in the prompt** ("the Name line is authoritative; ignore names in the Story text") | Persuasion, not structure. Bible `12.103`; and item F PROVED on this repo that an existing "invent none" instruction did not hold |
| **E** | **Post-hoc gate** -- reject/repair a description containing a proper name no roster row owns | Detection is proven (see the two detectors). But THE LAW forbids an audit FAILING an episode, so it must repair or log, never reject |

**These are not exclusive.** My prior is B+E, with A investigated only if the
continuity mapping turns out to be trustworthy. I want that broken.

## 5. What I am asking the panel

1. **Is the continuity mapping real or coincidental?** `meta.continuity.facts`
   carries `"Lucille (Nia Philbin)"` while the DESCRIPTIONS are crossed the
   other way. Find who writes that pairing and whether it can be trusted as the
   reconciliation source. **If it is itself wrong, option A is dead.**
2. **Where is the narrowest cut?** The brief is consumed by more than the
   description producer. Name every consumer of `casting_brief` and say which
   would break if the names were substituted or stripped.
3. **Does the repair belong before or after generation?** B prevents; E detects.
   Given THE LAW forbids rejection, is a detect-and-reroll even reachable, or
   does it have to be a prompt-side prevention?
4. **The wrong-character paste may not be one bug.** RICK STEINER carrying
   LUCILLE PENNY is a NAME in the identity slot; is that the same defect as a
   row carrying another row's whole face prose? Tell me if I am merging two.
5. **Tell me what I got wrong.** Panels corrected this driver three times on
   2026-08-17, twice on build-breakers. Assume a fourth.

## 6. Constraints -- a proposal breaking one is dead

* **THE LAW:** an audit may improve a story, never FAIL one for length, language,
  style or quality. No gate may reject an episode.
* **A render must degrade, never raise.**
* **No content guardrails on generated episodes** (2026-08-03).
* **Story quality is CLOSED** (2026-08-04). Naming the right person is
  correctness; writing a better description is not on the table.
* **`otr/obs/` is the success signal** (2026-08-17) -- no fix may reduce how many
  episodes publish.
* **Do not launder it as a gender fix.** That is exactly how it stayed hidden.
