# Driver anchor -- voice-pool concentration (2026-08-18 night)

Written FIRST, from the code and the corpus, before any panel round. Every number
below was measured at HEAD `9665ebd7` on the real Windows tree; nothing here is
recalled or arithmetic. The driver remains the sole judge.

---

## THE HEADLINE: THE RECORDED CAUSE IS WRONG, AND THE REAL ONE IS A DIFFERENT SUBSYSTEM

`GO_FORWARD_PLAN.md` (queue row, written 2026-08-18 evening) says:

> **THE CAUSE IS THE MATCH LADDER, NOT A BLOCK.** ... production also passes the
> writer's voice-fit `timbre` + `age_band` (`cast_lock.py:909-926`), and
> `g+t+r+age` scoring narrows so hard that a couple of references win most slots
> before the ladder ever drops a term. **The pool is open; the MATCHING
> concentrates it.**

**That does not reproduce, and the subsystem it names is not the one that runs.**

Measured by calling the real `assign_voice_for_slot` across 400 episode seeds x 3
char_ids per ask (`indextts2`, `role="char_voice"`, `age_band="adult"`):

| ask | distinct drawn | top-5 share | uniform-over-N baseline |
|---|---|---|---|
| male + `warm` | 13 | 44% | 38% (5/13) |
| female + `warm` | 22 | 28% | 23% (5/22) |
| female + `bright` | 17 | 32% | 29% (5/17) |
| male, gender only | 18 | 33% | 28% (5/18) |

The deterministic scorer is **essentially flat** -- within a few points of uniform
in every case. `vz_bill_boerst`, the corpus's 23.4% winner, does not even reach
the top 6 of its own ask. Tier-1 pools are healthy, not thin: male+warm = 13,
female+bright = 17, male+donated = 12. **The ladder is not the concentrator.**

### The reason the ladder is innocent: it almost never runs

`_otr_casting.py:896` -- `hybrid_voice_fit_enabled()` returns True unless
`OTR_HYBRID_VOICE_FIT=0`. **Default ON.** So the live path is:

1. `_otr_casting.py:2004-2034` -- for each ensemble slot, `build_voice_cards()`
   builds a card list, `llm_propose_voice_ref()` asks the LLM to pick ONE, and
   `validate_voice_proposal()` checks it. The result rides
   `meta.voice_cast_decision`.
2. `cast_lock.py:884-906` -- CastLock reads that decision, and if the accepted id
   re-validates it stamps the row and **`continue`s**, never reaching
   `assign_voice_for_slot` at `:909-926`.

Measured over 1711 ledgers in `output/otr/episodes/`: **1871 rows were
hybrid-ACCEPTED against 82 fallbacks** (`no_cards` 76, `invalid_or_collision` 6).
The deterministic scorer the plan blames handles roughly **4%** of casting rows.
Tuning it -- "loosen the timbre weight, spread within a score tier, track recent
use" as the queue row proposes -- would change almost nothing.

---

## THE REAL MECHANISM: TWO STACKED NARROWINGS, NEITHER OF THEM THE LADDER

### Narrowing 1 -- `build_voice_cards` alphabetically truncates the pool, permanently

`nodes/_otr_voice_bank.py:667-672`:

```python
pool = sorted(
    (e for e in entries
     if e.engine == engine and e.gender == gnorm
     and getattr(e, "quality_tier", "") != "reject"),
    key=lambda e: e.voice_ref_id,      # <-- ALPHABETICAL
)[: max(0, int(max_cards))]            # <-- FIRST 12, max_cards default 12
```

It sorts by `voice_ref_id` and takes the first 12. Not a sample, not a
match-scored shortlist -- a fixed alphabetical prefix. **The same 12 voices are
offered on every episode, forever.** Measured at HEAD:

| gender | castable | offered to the LLM | never reachable |
|---|---|---|---|
| male | 18 | 12 | **7** |
| female | 22 | 12 | **10** |
| **total** | **40** | **23** | **17 (42%)** |

The 17 that can never be chosen while the hybrid path is on include every
LibriVox public-domain narrator (`vz_pd_librivox_mark_f_smith`,
`..._mark_f_smith_elder`, `vz_pd_librivox_phil_chenevert`, `vz_peter_yearsley`,
`vz_stuart_bell`) and both LJSpeech entries. The corpus agrees independently:
across 1711 ledgers only **26 distinct voices ever appeared on any card list**.

**The operator's words were literally accurate.** He said *"many voice models were
blocked and we're still limited to a handful."* The evening's answer -- "Nothing
is blocked ... 98.5% castable" -- measured the path that runs 4% of the time. On
the path that actually runs, **42% of the pool is unreachable**, and it is
unreachable by alphabet, which is why no quality-reject audit could see it.

### Narrowing 2 -- the LLM then collapses the surviving 12 onto a handful

Corpus, `meta.voice_cast_decision`, 1877 proposals:

| voice | proposals | share |
|---|---|---|
| `vz_bill_boerst` | 669 | 35.6% |
| `vz_caro_davy` | 516 | 27.5% |
| `vz_donor_andrea` | 202 | 10.8% |
| `vz_donor_glenn` | 200 | 10.7% |
| `vz_donor_kditz` | 197 | 10.5% |

**18 distinct proposals, top-5 = 95%.** Nine of the 26 voices ever offered were
**never once proposed**. Validation barely moves it (accepted: 17 distinct, top-5
95%) -- it only rejects 6 collisions, so the LLM's preference passes through
essentially unfiltered to the ledger.

Note the shape: `vz_bill_boerst` and `vz_caro_davy` are the **first male and first
female card** on their respective alphabetical lists. Primacy on a fixed list,
asked 1877 times, is the whole story of the 42.5%.

### The two narrowings compose

40 castable -> 23 ever offered (alphabetical truncation) -> 5 carry 95% of picks
(LLM primacy). The corpus outcome the queue row recorded is reproduced exactly:
character rows 1404, 56 distinct, top-5 66%, `vz_bill_boerst` 23.4% +
`vz_caro_davy` 18.7% = 42.1%. (The row said 1357/56/66%/23.7%/18.8% -- same
measurement, a few more episodes since.)

---

## SEPARATE AND MORE URGENT: THE LEMMY RESERVATION DOES NOT COVER THIS PATH

This is not a design fork. It is a live correctness defect with one right answer,
and it reopens a row closed yesterday.

`reserved_voice_ref_ids()` is applied in exactly one place --
`assign_voice_for_slot` (`_otr_voice_bank.py:528-531`). **Neither
`build_voice_cards` nor `validate_voice_proposal` knows reserved ids exist.**
`validate_voice_proposal` (`:726-750`) checks in-library, engine, gender, reject
tier and collision -- and stops there.

Measured live at HEAD, right now:

```
build_voice_cards("indextts2", "male")[0] == "idx_lemmy_algenib_cockney_v1"
validate_voice_proposal("idx_lemmy_algenib_cockney_v1", "indextts2", "male")
    -> "idx_lemmy_algenib_cockney_v1"        # ACCEPTED
```

Lemmy's qualified clone is offered as **card #1** to the LLM on every male slot,
and validation waves it through. Corpus: the LLM proposed a reserved id **21
times and it was accepted all 21**, stamping non-Lemmy characters -- DON PEDRO,
MARCELLUS, MOE GORDON, FLETCHER CORBEN, BANQUO, STARBUCK, FERDINAND, Dr. Alexei
Petrov. 20 leaked cast rows in total against 5 legitimate LEMMY rows.

### This corrects yesterday's closing diagnosis

`HANDOFF_LOG.md` explains the one post-fix sighting
(`signal_lost_rivers_embrace_20260817_233013`, ED HIBBERT, 16h after fix
`8f3c7615`) as the resident-server trap:

> The soak harness boots ONE server and never tears it down, so the evening leg
> still ran the module Python imported that morning. **A stale resident process
> reads exactly like a fix that did not work.**

**Process age is not what produced that row.** The hybrid path was never guarded,
at that commit or at HEAD -- the two live calls above are today's, on a freshly
imported module. `rivers_embrace` is the fix's coverage gap, not a stale import.

PBUG-20260817-08's closure is **half right and should be re-opened as a
narrowed row**: the deterministic path IS fixed and genuinely proven (480 seeded
draws, `tests/test_lemmy_voice_stays_reserved.py`). Those 480 draws all went
through `assign_voice_for_slot` -- **the path that carries 4% of production
casting**. Bible `12.114`'s first reusable half already states the general shape
and is, if anything, better evidence than the episode it was promoted on: *"a
reservation existing as a CONVENTION in one subsystem is invisible to another
enumerating the same catalogue."* That is precisely `build_voice_cards`.

The operator's *"i do feel we have been seeing the right amount of lemmy"* is
consistent with a 1.1% leak rate (21 of 1877), not with absence.

---

## A PANEL ALREADY CALLED THIS ON 2026-08-04 AND IT WAS NEVER ACTED ON

`kibitz-runs/2026-08-04-continuity-ultracode/input_voice-variety.json:174`, two
weeks before the operator raised the symptom:

> **CLAIM:** *"The plan never mentions the hybrid LLM voice-fit, which sits IN
> FRONT of the deterministic caster that steps 5 and 6 fix, and whose 12-card
> truncation is a harder variety cap than the tier-of-one. Steps 5/6 do nothing
> for any character whose LLM proposal is accepted, and the plan's 200-episode
> simulation measures a path production can bypass entirely."*
>
> **FIX:** *"Name the layer and give `max_cards` an owner in the same change:
> either raise/remove the 12-card cap and make the slice a seeded sample rather
> than an alphabetical head ... Add one assertion to the new
> `tests/test_voice_variety.py` that runs with the hybrid path ENABLED so a
> future working LLM cannot silently undo the win."*

Every element of the diagnosis above is in that paragraph, including the phrase
"an alphabetical head". It was correct, it was ignored, and the 2026-08-18
evening measurement then made the exact mistake it warned about -- sweeping
`assign_voice_for_slot` and concluding "the pool is open" from a path production
bypasses 96% of the time.

Two things follow. First, `max_cards` still has no owner: one caller
(`_otr_casting.py:2004`) and it never passes the argument, so the cap is 12 by
default and always has been. Second, the panel's own suggested guard -- a test
that runs with the hybrid path ENABLED -- is still missing, which is why the
480-draw Lemmy proof passed while the leak stayed live.

## THE DESIGN FORK FOR THE PANEL

The Lemmy hole has one answer (reserve-aware cards + validation) and takes no
arc under the 2026-08-17 amendment. **The concentration is the genuine fork**, and
the options behave differently:

1. **Raise / remove `max_cards`.** Offer all 18-22 same-gender voices. Fixes
   reachability outright; does nothing about primacy, and lengthens the prompt.
2. **Rotate or seed-sample which 12 are offered.** Keeps the prompt small and
   makes every voice reachable across episodes. Needs a seed that is stable
   within an episode but varies across them -- and the honest question is whether
   the LLM's judgement is worth keeping at all if the shortlist is random.
3. **Rank the cards by the deterministic scorer instead of alphabet.** The
   shortlist becomes match-relevant rather than accidental, and the measured-flat
   scorer does the spreading. Primacy then favours a *good* match rather than
   whichever id sorts first.
4. **Track recent use across episodes** (the queue row's third option) and
   down-weight or exclude recent winners. Fixes repetition directly; introduces
   cross-episode state, which nothing in casting currently has.
5. **Drop the hybrid pass.** The scorer measures flat and the LLM is the
   concentrator. **This one is governed by `CLAUDE.md`'s ledger rule:**
   removing an LLM pass demands every field it writes get a new owner --
   `meta.voice_cast_decision` (policy_version, bank_sha, engine, prompt_version,
   seed, candidate_ids, proposed_id, accepted_id, fallback_reason) is consumed at
   `cast_lock.py:688`. Not proposed here; listed so the panel prices it.

Options 1-3 are not exclusive; 3 + a raised cap is the driver's current lean,
because it fixes reachability and primacy with no new state and no ripped pass.

**Everything above measures on CPU.** No GPU leg is needed to decide it.

### Four of the driver's own break-attempts, already run

Stated so the panel spends its rounds on the fork rather than re-deriving these:

* **Is the hybrid path really default-on in production?** YES.
  `OTR_HYBRID_VOICE_FIT` appears in no profile, no launcher, no `.cmd`, no
  workflow JSON -- only in the function itself and in tests. Nothing in the
  shipped configuration turns it off.
* **Does any caller raise `max_cards`?** NO. Exactly one caller,
  `_otr_casting.py:2004`, and it never passes the argument, so the cap has
  always been 12. `max_cards` still has no owner -- which is the same finding
  the 2026-08-04 panel filed.
* **Is `indextts2` really the engine that matters?** YES.
  `default_char_engine` resolves to it, and across the corpus it carries
  **1376 of 1406 character rows (97.9%)**; google_tts, kokoro, elevenlabs, dia
  and chatterbox split the remaining 30. The 42% figure is therefore about
  essentially all production character casting, not one engine among six.
* **Are the two hot voices simply the best matches?** NO -- they are the first
  card. `vz_bill_boerst` is `male/[warm, baritone]/adult`, `vz_caro_davy` is
  `female/[warm, alto]/adult`; 13 other male voices match `warm` equally well
  and the flat scorer picks among them evenly.

### What the panel should still try hardest to break

* Whether alphabetical order is load-bearing anywhere else -- a determinism or
  replay-parity contract that a reordering would break. `stable_cast_seed`
  covers the SCORER's draw, not the card list, but this needs checking.
* Whether `VOICE_FIT_POLICY_VERSION` must bump on any of these, and what a bump
  re-baselines downstream.
* Whether spreading the LLM's choices is even the right goal, or whether the
  honest conclusion is that a model asked to rank 12 near-identical descriptors
  is not adding judgement worth its concentration cost.
* The driver's lean (rank cards by the deterministic scorer, and raise the cap)
  -- specifically whether a score-ranked shortlist just moves primacy onto a new
  fixed order rather than removing it. This is the weakest point in the lean and
  the panel should attack it first.
