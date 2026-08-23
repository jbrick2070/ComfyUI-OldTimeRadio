# Judgment -- Ghost Prompt v2.1, ONE round (r2)

**THIS IS ONE ROUND, NOT AN ARC.** The operator asked for one; r1, r3 and r4
were not run and nothing here may be described as a four-round campaign.

**Roster, stated exactly.** Driver: Claude (Cowork), sole judge, excluded from
the panel. Reviewers: **Codex `gpt-5.6-sol` (high)**, **Antigravity (Gemini)**,
**Cursor `cursor-grok-4.6-high`** -- three external calls, three different model
families. Cursor was added mid-round on the operator's explicit authorization.
All three returned substantive line-cited reviews (7.6 KB / 7.0 KB / 8.7 KB).

## Accepted and fixed

| # | finding | lanes | grounded |
|---|---|---|---|
| 1 | `_HUMAN_WORDS` still held bare `face`/`faces` -- the clock-hand bug in another word. This show's bookends are DIALS and v1's framing said "dial face centered". | Cursor | CONFIRMED |
| 2 | `_LETTERING_WORDS` banned `letter` while `MOTIF_PROP_WORDS` OFFERS `letter` as a prop -- the code killing a batch over its own motif. | Cursor | CONFIRMED |
| 3 | Pronouns and generic person nouns were dropped when the body parts went, so `"he turns the dial"` passed object mode. | Codex, Antigravity | CONFIRMED -- a regression I introduced |
| 4 | The retry re-sent byte-identical text at temperature 0.1, so it could not differ. | **all three** | CONFIRMED |
| 5 | A broad `except Exception` wrapped the parser and the validators, laundering a bug in our own code into "the model failed". | Codex | CONFIRMED |
| 6 | `already_used` reached only the deterministic path, so an authored leaf could duplicate a replayed one; the two paths also compared case differently. | Codex | CONFIRMED |
| 7 | Leaf-admission semantics changed without any hashed key changing, so a leaf admitted under v2's looser rules replayed untouched under v2.1. | Codex | CONFIRMED |
| 8 | `grain` / `noise` are polysemous -- a sack of grain, wood grain, a sudden noise -- and rejected concrete leaves. | Antigravity, Cursor | CONFIRMED |
| 9 | `emblem` and `field` were named as mush-makers in the receipt and nothing banned them. | Cursor | CONFIRMED |
| 10 | `"a angular figure"` -- the article was computed for colour and prop, not silhouette. | Antigravity | CONFIRMED |
| 11 | `_first_allowlisted` walked the VOCABULARY, not the source, contradicting its own docstring. | Codex | CONFIRMED |
| 12 | An exhausted fallback pool returned a silent duplicate while the authored path forbids them. | Codex, Cursor | CONFIRMED |
| 13 | "half of them show a person" is FALSE for counts that are not multiples of four. | Cursor | CONFIRMED -- measured floor(n/2) for n=1..7 |

## Accepted as a correction to the RECEIPT, not the code

**"Legibility tracks concrete nouns, and nothing else" is not isolated.**
Codex and Cursor both refused it, and they are right: v2.1 changed motifs, laws,
figure wording, garments, bookends, the cycle and validation simultaneously, and
the 6/8 result came from a DIFFERENT episode rather than the same-seed control.
It is a **concrete-subject hypothesis** with strong supporting evidence, not
demonstrated exclusive causation. The receipt is being amended to say so.

## REJECTED, with reasons

* **My own "negative prompt relative weight" theory** -- cut by Codex, and it is
  right. Positive and negative are encoded independently and handed to the
  sampler at fixed CFG 8.0; there is no length-derived weighting mechanism.
  Antigravity offered the same theory as a supporting mechanism and it is
  rejected on the same grounds. A wrong mechanism agreed on by two lanes is
  still wrong.
* **Removing `static` from the abstract list** (Antigravity). `"bands of static
  crush inward"` is the measured failure. Its polysemous neighbours went;
  `static` stays.
* **`--all-agents`-style expansion of the human/abstract dictionaries** (Codex
  cut, adopted). The lists are now narrow and evidence-led.

## DEFERRED to the operator -- not a coder's call

**Row-local salvage.** All three lanes call the whole-batch reject a must-fix:
one bad leaf discards seven accepted LLM leaves and has now cost two live
episodes. Codex further showed my stated rationale is self-defeating -- the
retry regenerates the good rows too, so results ALREADY depend on whether a
retry happened.

I have **not** implemented it, because "no rows are salvaged across attempts" is
an explicit, operator-approved line in
`docs/2026-08-22-GHOST-PROMPT-V2-CONTROLLED-ABSTRACTION-PLAN.md`. The informed
warmer retry (finding 4) removes most of the practical pain without touching
that contract. Changing the contract is his call.

## Not verified

* Cursor's claim that `style_id` belongs in the request hash (a pack switch
  replaying a leaf admitted under a shorter cue). Plausible and cheap, but I did
  not construct the failing case, so it is recorded rather than acted on.
* Cursor's `assert_shell_fits` claim that the hardcoded longest motif is not the
  real longest. Likely true; not measured.
