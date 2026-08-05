# Continuity diagnostic: gender, voice, portrait (2026-08-04)

Operator observation while watching episodes: character image continuity,
gender association and voice association look wrong. This is the grounded
diagnosis. **Every claim below is a field the pipeline actually wrote into a
published ledger or a vendored sidecar** -- no models were run, no GPU used.

## 1. The defect, in published episodes

Four gender mismatches across the twelve most recent adaptation-lane
episodes, checked against the source's own canonical characters:

| Episode | Character | Ledger says | Source says |
|---|---|---|---|
| needle_in_the_marrow (Comedy of Errors 3.1) | ANTIPHOLUS | female | male |
| malvolios_yellow_stockings (Twelfth Night 2.5) | MALVOLIO | female | male |
| malvolios_yellow_stockings | MARIA | male | female |
| whispers_in_the_unseen (Tempest) | MIRANDA | male | female |

Note MALVOLIO and MARIA are **inverted in the same episode**. That is the
signature of a coin flip, not of a lexicon failing on an ambiguous name.

**Gender and voice are ONE defect, not two.** The voice picker selects from a
pool filtered by gender, so a wrong gender deterministically produces a wrong
voice. Fixing gender fixes both.

## 2. Root cause: two independent gaps

### Gap A -- the recorded answer never reaches casting (the big one, and free)

`nodes/_otr_character_roster.py` parses a play's dramatis personae and writes
a `characters` list into the provenance sidecar. It works, it has tests, and
the sidecars exist:

```
config/source_banks/shakespeare/sources/comedy_errors__act3_scene1.provenance.json
  characters: [{"name": "ANTIPHOLUS OF EPHESUS", "description": "a citizen of
  Ephesus", "gender": "unknown", "gender_source": "unknown"}, ...]
```

But **nothing in `nodes/` reads `source_meta["characters"]`**. Repo-wide grep
finds `parse_character_roster` imported by exactly one place --
`scripts/otr_fetch_public_domain.py`, a vendor-time script -- and by the tests.
The render path never consults it.

Instead `_otr_casting.precompute_ensemble_slots` decides gender by a pure-Python
statistical ensemble roll (~40/40/20 male/female/other). Documented as
"decide the whole ensemble's gender / timbre / role distribution up front. PURE
PYTHON -- no LLM." For a source-owned lane that roll is simply wrong: the
play already said who these people are.

**This is the classic unwired-code shape**: the work shipped, the consumer was
never connected. `docs/GO_FORWARD_PLAN.md` even predicted the exact failure --
"the statistical 40/40/20 allocator's rolled gender therefore stands and the
voice picker takes it, so a female lead draws a male Bark voice on roughly half
of seeds **the moment this lane wires up**." The lane wired up. The prediction
came true in published episodes.

### Gap B -- 38% of roster entries have no gender to read

Across every Shakespeare sidecar: **male 30 / female 23 / unknown 32**.

The parser is not broken; the *descriptions* are genuinely gender-neutral in
English. "ANTIPHOLUS OF EPHESUS -- a citizen of Ephesus". "DROMIO OF EPHESUS --
Antipholus of Ephesus's servant". No deterministic rule recovers "male" from
"a citizen", and the parser honestly records `unknown` rather than guessing.

`unknown` then falls through to the ensemble roll -- the coin flip in section 1.

## 3. What is NOT the problem

* **Voice pool size.** Every episode examined has exactly THREE speaking parts
  (announcer + two characters) against a 6-male/4-female Bark pool plus kokoro
  for the announcer. The pool is nowhere near stressed. Downloading more voices
  would not have changed a single row in the table above.
* **Preset collisions.** No episode assigned one preset to two characters.
* **Engine mixing.** kokoro for the announcer and indextts2 for characters is
  consistent across all twelve episodes -- that is by design, not drift.

## 4. Recommended fix, cheapest first

1. **Wire the recorded roster into casting (no LLM, no cost).** For lanes where
   `style_pool_class == "adaptation"`, a character whose sidecar carries
   `gender` must take that value and skip the ensemble roll. This alone fixes
   MIRANDA, LEAR, CORDELIA, PROSPERO and every other entry among the 53
   already-known -- roughly 62% of the corpus -- and it costs one wiring change.
2. **A vendor-time LLM tiebreak for the 32 unknowns.** This is where an LLM pass
   genuinely earns its keep: "a citizen of Ephesus" needs world knowledge, not
   parsing. Run it ONCE per source at vendor time, cache the answer in the
   sidecar under its own `gender_source` (e.g. `llm_vendor_v1`) so it stays
   auditable and never re-runs per episode. `GO_FORWARD_PLAN.md` already
   sanctions exactly this: "a vendor-time LLM/web lookup as the final tier for
   stragglers, recorded under its own `gender_source`".
3. **Only then consider more voices**, and only if cast sizes grow past the
   current three.

## 5. Portrait continuity -- NOT diagnosed

Episodes carry 6-7 images each, but this pass could not attribute them to
characters: the ledger's `images` structure did not expose a character key in
the shape probed. **Reported as unresolved rather than clean.** A second pass
should establish whether one character keeps one visual identity across beats,
which is the operator's actual complaint and is untested here.

## 6. Scope note

Gaps A and B are on the ADAPTATION lanes, where the source states the answer.
The invention lanes (`original`, `scifi_news`, `media_archive`,
`scifi_news_pro`) have no cast list, so the ensemble roll is the correct owner
there and must not be disturbed.
