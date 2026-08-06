# PROBLEM STATEMENT -- the cast row's gender can be wrong in three places

**Date:** 2026-08-06
**Driver:** Claude (Cowork), CODER window. HEAD `e499b7fc` on `v2.0-alpha`.
**Status:** STATIC. All three sites are established by file read only. No live
artifact, so nothing here may enter `docs/PROD_BUG_LOG.md`.
**Gate:** full `kibitz-plugin:kibitz` four-round arc before code.

Context: PBUG-20260805-01 established that a generated gender contradicting the
source is a CORRECTNESS defect, not story quality, and therefore still open
under the 2026-08-04 "story quality is done" directive. These are three
remaining places the cast row's gender can end up wrong or unpaired.

---

## SITE 1 -- `_otr_casting.py:583-587`: an out-of-vocabulary prior gender inflates the whole ensemble target

```python
# nodes/_otr_casting.py:582-595
total = count + len(prior_genders)
prior_counts = {g: 0 for g, _ in _DEFAULT_GENDER_WEIGHTS}   # male/female/other
for g in prior_genders:
    g_norm = (g or "").strip().lower()
    if g_norm in prior_counts:
        prior_counts[g_norm] += 1
...
for gender, weight in _DEFAULT_GENDER_WEIGHTS:
    want_total = weight * total
    want_open  = want_total - prior_counts[gender]
    raw.append((gender, max(0.0, want_open)))
```

`_DEFAULT_GENDER_WEIGHTS` is `male 0.40 / female 0.40 / other 0.20`
(`:153-157`). A prior row whose gender is anything else -- `""`, `"unknown"`,
or any value a future producer introduces -- is **counted in `total` but
credited to no bucket**. Every `want_total` therefore rises while no
`prior_counts` entry rises, so all three `want_open` values are inflated by
that row's share.

Largest-remainder rounding still sums to exactly `count`, so the ensemble is
never the wrong SIZE -- but the split it lands on can differ, because the
remainders it ranks are computed from the inflated numbers.

**This MOVES THE GENDER ROLL**, so any change here needs a declared
re-baseline: existing seeds will produce different ensembles.

**Open question for the panel:** what is the correct treatment? Either
(a) exclude unrecognised priors from `total` as well, so the ratio is taken
over the slots that can actually carry a bucket, or (b) map them onto `other`.
These give different answers and the choice is a product decision, not a
mechanical one.

## SITE 2 -- `_otr_casting.py:469`: the prior-cast echo collapses `other` to "X"

```python
# nodes/_otr_casting.py:467-469
name = (row.get("name") or "?").upper()
g = (row.get("gender") or "?").lower()
g_short = "M" if g == "male" else "F" if g == "female" else "X"
```

This builds the prior-cast echo line the description LLM sees.

**CORRECTION TO THE INHERITED FRAMING, made before the panel ran.** The task
that queued this item described `:469` as a site that MOVES THE GENDER ROLL.
Read against the file, **it does not**: `_prior_cast_line` is a display/echo
formatter, it consumes no rng and writes no row. Its effect is on what the LLM
is TOLD, not on what is drawn. Recorded rather than repeated so the next reader
is not hunting a roll that is not there.

The real question is narrower and still worth asking: `other`, absent and
malformed genders all collapse to the same `"X"`, so the description model
cannot distinguish "this character is non-binary" from "nobody recorded a
gender". Whether that matters depends on whether the description prompt is
meant to act on the distinction.

## SITE 3 -- `story_orchestrator.py:448-453`: a merged row can take its voice from one character and its gender from another

```python
# nodes/story_orchestrator.py:448-453
if not win_row.get("voice_preset") and lose_row.get("voice_preset"):
    win_row["voice_preset"] = lose_row["voice_preset"]
if not win_row.get("description") and lose_row.get("description"):
    win_row["description"] = lose_row["description"]
if not win_row.get("gender") and lose_row.get("gender"):
    win_row["gender"] = lose_row["gender"]
```

This is the near-duplicate cast-name merge (the ANNOUCNER/ANNOUNCER typo
fold). The three fields are copied under **INDEPENDENT** guards, so a winner
row that already has a `voice_preset` but no `gender` keeps its own voice and
inherits the loser's gender.

**`voice_preset` and `gender` are not independent facts about a row.** The
voice was selected FOR a gender upstream; splitting the pair is exactly the
"a character's gender/voice contradicting the source" class the operator kept
open as a correctness bug. Item 8's `presentation_gender` receipt
(`d4e51b4d`) exists to make this detectable after the fact -- but detection is
not prevention, and this is the site that creates the mismatch.

**What is NOT established, and the panel must not assume it:** that this fires
in production. It requires a merge where the winner has one field and not the
other. Whether real near-duplicate rows reach this state is unverified.

## SITE 4 -- `_otr_casting.py:756-759`: VERIFIED INERT, recorded so it is not re-derived

```python
pins = {
    str(k).strip().upper(): str(v).strip().lower()
    for k, v in (gender_by_name or {}).items()
    if str(v).strip().lower() in _PINNABLE_GENDERS
}
```

The filter drops any pin whose gender is not `male` or `female`
(`_PINNABLE_GENDERS`, `:146`). **It can never drop anything today:**

* `RosterGenderVerdict.gender` is documented and produced as
  `"male" | "female" | "unknown"` (`_otr_roster_gender.py:212`), and
* `build_gender_by_name` **omits** unresolved names rather than recording them
  as `unknown` -- *"Unresolved names are omitted, never recorded as 'unknown':
  the caller pins what it knows and leaves the rest to the existing allocator
  untouched"* (`_otr_roster_gender.py:466-469`).

So every value reaching the filter is already `male` or `female`. The guard is
a fail-closed backstop against a future producer, not a live defect.
**No change proposed.** It is written up only so the next audit does not spend
a third session re-deriving that it is inert.

---

## WHAT THE PANEL IS BEING ASKED

1. Site 1's correct treatment (exclude from `total`, or map to `other`), and
   what a declared re-baseline must state and cover.
2. Whether Site 3 is reachable in production, and whether the fix is to couple
   the guards, to refuse the merge, or to recompute the voice from the winning
   gender.
3. Whether Site 2 is worth changing at all, given the correction above.
4. Anything that makes these three ONE defect rather than three -- if the cast
   row's gender has a single owner that all three violate, that is the finding.

## LINE-CITE WARNING

Commit `496d9d57` inserted roughly 90 lines near the top of
`nodes/_otr_roster_gender.py`. Any citation into that file taken from the
2026-08-06 gender-ladder r3 review has SHIFTED. Every cite in THIS document was
re-pinned against the working tree at `e499b7fc`.
