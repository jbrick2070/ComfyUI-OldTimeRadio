# REBUTTAL -- the Lemmy identity premise, and what the ledgers actually show

**For re-review by the plan's author.** Target:
`docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md` (authoritative) and its
companion execution view.

**This is not a dispute about the audit facts.** The plan already records them
correctly. The dispute is about what FOLLOWS from them, and about a diagnosis
that survived from an earlier round which the plan's own evidence contradicts.

---

## 1. THE MEASUREMENT (reproduce it before accepting anything below)

Scanned every ledger under the ComfyUI output root, walking each JSON for rows
whose `name` / `char_name` is `LEMMY`, and counted `(voice_ref_id, engine)`:

```
ledger files scanned: 1633
LEMMY rows found:      186

  voice_ref_id=None                     engine=bark        x151
  voice_ref_id=vz_donor_marshal_indian  engine=bark        x33
  voice_ref_id=cb_donor_marshal_indian  engine=bark        x1
  voice_ref_id=gt_puck                  engine=bark        x1
```

The plan's "33 rows, all `vz_donor_marshal_indian`" is **exactly right**. The
151 `None` rows are the bark-preset path, where no bank reference is used, so
`None` is expected rather than missing.

**The bank row he has actually been using** (`config/voice_reference_bank.json`):

```
vz_donor_marshal_indian
  engine            indextts2
  gender            male
  timbre            ['donated', 'warm']
  age_band          adult
  ref_path          models/TTS/refs/indextts2/vz_donor_marshal_indian.wav
  commercial_clean  True
```

Two sibling rows point at the SAME wav: `cb_donor_marshal_indian` (chatterbox)
and `dia_donor_marshal_indian` (dia).

## 2. WHAT THIS CONTRADICTS

### 2.1 "He has no durable identity" -- not supported by the evidence

The plan says *"A seed lottery is not a durable identity"* (section 3), and the
Fable ruling it inherits from states outright that on the production engine he is
*"drawn fresh on gender alone, with a different roll every episode ... a
different man at every single appearance."*

**The outcome says otherwise: 33 of 35 non-bark rows resolved to the SAME
reference.** Whatever the mechanism is on paper, in practice it returned one
answer, repeatedly, across the whole shipped history.

That is not a quibble. It changes what needs building:

* If the draw is described as seed-dependent but has been **empirically
  constant**, then either the seed inputs are stable enough to be effectively
  deterministic already, or something else is pinning the result. **Nobody has
  explained which**, and the plan proceeds as though non-determinism is the
  defect.
* A "stable identity key" fix targets **non-determinism**. The evidence shows
  non-determinism is not what shipped. The fix may therefore be solving a
  condition that does not manifest, while the condition that DID ship -- a wrong
  but stable voice -- needs a different remedy: **re-pin, not re-key.**

**Requested:** before coding either, explain the 33/33 constancy. If the draw is
already stable, say so and re-scope the fix. If it is genuinely a lottery that
coincidentally returned one value 33 times, show why.

### 2.2 The defect is a FLOOR CONTRADICTION, and it is the urgent one

`vz_donor_marshal_indian` contradicts the agreed cross-engine floor on **two**
attributes simultaneously:

| floor requirement | what shipped |
|---|---|
| vocal weight: **gravelly / raspy** | `timbre: ['donated', 'warm']` -- *warm*, the opposite axis |
| nationality/class: **London working-class** | identifier reads `..._indian`; no accent field at all |

Note also what the row **lacks** next to the audition candidate: `gt_algenib`
carries `measured_median_f0_hz: 97.2` and `gender_source:
measured_median_f0_2026-08-08`. The incumbent carries no measured pitch, no
accent field, and a timbre pair (`donated`, `warm`) that describes its
PROVENANCE rather than its sound.

So the shipped state is not "consistent but unauditioned." It is **consistently
contradicting the floor the plan exists to enforce.**

### 2.3 The "no squatter's rights" argument inverts

The inherited reasoning for accepting an unauditioned deterministic pick was that
*"today's chaos at least means no wrong voice has squatter's rights."*

**A wrong voice has squatter's rights over 33 shipped episodes.** Any change is
therefore a genuine RECAST that a returning listener could notice -- not the
tidying of a random state. Consequences:

1. The re-baseline is an **editorial event**, not a formality. The plan's
   argument that its blast radius is "exactly the already-broken set" still
   holds, but "broken" here means *audibly established*, which is a different
   cost.
2. **Test A's acceptance criteria should include an A/B against the incumbent**,
   not only an absolute judgement of the candidate. The question a listener will
   actually experience is "is this better than the Lemmy I have heard 33 times",
   and a blinded scorecard that never plays the incumbent cannot answer it.
3. It argues **for** the audition, harder than before -- there is now a specific
   thing to beat.

## 3. A SECOND ANOMALY, UNEXPLAINED

Every one of those 33 rows records **`engine=bark`** while carrying a **`vz_`
(indextts2) reference id**. The row's engine field and its reference disagree.

Most likely the dead `lemmy_row()` bark pin is riding along on rows the
indextts2 bank actually served -- but that is a guess, and it matters because:

* any audit keyed on a Lemmy row's `tts_model` is reading a field that may not
  describe what rendered him; and
* if the pin and the served reference can disagree in the ledger, they can
  disagree in the receipt the new qualification record is meant to make
  trustworthy.

**Requested:** determine which field is authoritative on those rows before the
route-resolution seam is built on either.

## 4. WHAT I AM NOT DISPUTING

* The `gt_algenib` correction is **right and my earlier research plan was wrong**
  on it: it is a provider route (`ref_path: cloud:google_tts:Algenib`,
  `ref_sha256: "cloud"`), carries no local bytes, cannot be passed to an
  IndexTTS2 bank lookup, and Test A must mint a distinct local entry with a real
  64-hex hash.
* The G0/G1/G2 gating, the explicit rights decision, and refusing to infer
  permission from silence are all stronger than what I wrote.
* Branch A / Branch B bounding is sound.
* That past rolls are "not a promise about future rolls" is correct as stated --
  my point is that the observed constancy still demands an explanation before a
  determinism fix is scoped.

## 5. WHAT I AM ASKING FOR

1. Explain the **33/33 constancy** and re-scope the identity fix accordingly --
   re-pin versus re-key.
2. Reframe the defect in the plan as a **floor contradiction** (warm + Indian
   donor vs gravelly + London) rather than absent identity.
3. Add the **incumbent to Test A as an A/B arm**, since listeners have 33
   episodes of exposure to it.
4. Resolve the **engine/reference disagreement** on those rows before building
   the resolver seam.
5. Re-check any remaining sentence inherited from the earlier round that asserts
   per-episode voice churn; the evidence does not support it.

## 6. REPRODUCTION

Walk every `*ledger*.json` under `output/otr/episodes`, collect dicts whose
`name`/`char_name` upper-cases to `LEMMY`, and tally
`(voice_ref_id, tts_model|engine)`. Compare the winning id against its row in
`config/voice_reference_bank.json`, and against `gt_algenib` in the same file
for the fields the incumbent lacks.
