# Driver anchor -- 2.1 CHARACTER GENDER LADDER (r1)

Written by the driver (Claude, Cowork) BEFORE the panel, grounded on the real
Windows files. Every claim below was read, not recalled. The panel's job is to
break this framing, not to bless it.

## What is actually on disk today

`nodes/_otr_roster_gender.py` (566 lines) resolves a cast slot against the
sidecar roster. `resolve_roster_gender` (:374) is a FOUR-tier roster join, and
the first tier yielding any candidate wins:

| tier | rule (verbatim from the code) |
|---|---|
| `exact` | `slot in _candidate_names(r)` |
| `short_form` | `_strip_honorifics(slot)` equals `_strip_honorifics(n)` for some candidate name |
| `qualified` | some candidate name `.startswith(slot + " ")` |
| `contains` | `slot.startswith(n + " ")` |

`_candidate_names` (:318) returns the row's `name`, `roster_name`, AND the
stamper's `aliases` list.

**Two different axes are already distinguished, and conflating them is the first
way to get this wrong.** `RosterGenderVerdict` (:220) carries BOTH:
* `tier` -- HOW the roster join matched (exact/alias/qualified/contains/...)
* `gender_source` -- WHICH LADDER RUNG answered (:227 comment names
  `roster / pronouns / llm_recall / name_frequency`)

So the four tiers above are all ONE rung (`roster`). The spec's "tier 3" and
"tier 4" are RUNGS, not roster-join tiers.

## The work, stated exactly

**`llm_recall` and `name_frequency` DO NOT EXIST.** Verified by grep across
`nodes/`: they appear in exactly two places, both in this one file --- the
comment at :227 and the confidence map at :310
(`"llm_recall": "recalled", "name_frequency": "inferred"`). There is no call
site, no module, no index. The rungs were named and never built.

`nodes/_otr_gender_pronoun_scan.py` (340 lines) IS built and is the `pronouns`
rung.

## The abstention property that must survive

`_verdict_from` (:343) collapses a tier's rows and **abstains on disagreement**:
two matched rows with different binary genders return
`("unknown", "ambiguous_join")` -- the docstring calls picking one "a coin flip
wearing a citation". A tier matching only unknown-gender rows also abstains and
does NOT fall through to a looser tier, because "the source named this person
and declined to gender them, which is an answer."

**Any new rung must preserve this.** An LLM rung that answers where the roster
deliberately abstained would overwrite a confirmed source fact with a guess.
That is the single largest risk in this row and the panel should attack it.

## The surname fork, with the real collision path

The recorded example is COLONEL FITZWILLIAM vs Fitzwilliam Darcy
(PBUG-20260815-04 follow-up). Tracing the actual code:

* `_strip_honorifics("COLONEL FITZWILLIAM")` -> `"fitzwilliam"` (`colonel` IS in
  `_HONORIFICS`, :51).
* Whole-string compare against `_strip_honorifics("Fitzwilliam Darcy")` ->
  `"fitzwilliam darcy"`. **No match** -- so the row name alone is safe.
* The collision needs the ALIAS list: if the stamper wrote `Fitzwilliam` as a
  given-name alias for DARCY, `short_form` matches DARCY for a slot that means
  the COLONEL.

**Consequence the panel must reason about:** if BOTH rows match, `_verdict_from`
abstains only when their genders DIFFER. Here both are male, so the join returns
the right answer for the wrong reason. The defect is therefore NOT reliably
visible in output -- it is a latent wrong-character join that surfaces only when
a shared token spans two rows of different gender. Do not measure this row by
"did any episode look wrong."

## Binding rulings (a proposal violating one is invalid)

* NO web call, ever. The web-search tier is REPLACED, not plumbed.
* Shakespeare: fill ONLY the 32 `unknown` roster rows. KNOWN dramatis-personae
  rows are untouchable.
* ARIEL / PUCK / ROBIN stay on the roll; Dr. Lira Kell is female. Locked.
* Invented lanes (original, scifi_news_pro, media_archive) keep rolling -- their
  characters do not exist, so no lookup applies.
* Tier 4 is the DETERMINISTIC floor: an LLM failure still yields a gender.
* Tier 3 asks each name ONCE, EVER, via a PERSISTENT index. A repeat name costs
  zero LLM calls.

## Questions the driver wants broken

1. Is the surname rule best expressed as a RESTRICTION on `short_form`, an
   ABSTENTION on multi-row token collisions, or removal of the tier? Name what
   regresses under each.
2. Where does the persistent index live so it survives runs, is safe under the
   two-box split (5080 + 4060 both push), and cannot be corrupted by concurrent
   legs? Is git-tracking it an asset (shared knowledge) or a liability (merge
   conflicts on a hot file)?
3. What EXACTLY does the LLM rung do when the roster abstained on purpose? The
   driver's position: it must NOT run there.
4. The module docstring says it "never calls an LLM" (:20). That line becomes
   false. What else in the codebase depends on that promise?
