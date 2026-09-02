# Character gender ladder -- SPEC v2 (the rewrite)

**Supersedes `docs/2026-08-05-character-gender-ladder-SPEC.md`.** That spec
took THREE NOs across two review rounds (Codex r2: 11 must-fixes; r3: Codex NO
with 7, agy yes-with-fixes with 9). The DIAGNOSIS always survived -- 32 of 85
Shakespeare roster rows are `unknown` today, and Comedy of Errors ships 7
characters, every one of them unknown. What did not survive was the MECHANISM.
This rewrite keeps the diagnosis and replaces the mechanism.

Every cite below was re-pinned against HEAD on 2026-08-28. The r3 reviews cite
`nodes/_otr_roster_gender.py` at pre-`496d9d57` line numbers and every one of
them has moved; do not carry those numbers forward.

## What the operator ruled on 2026-08-28, and what it dissolves

**Ruling 1 -- Shakespeare: fill ONLY the `unknown` rows.** KNOWN rows parsed
from the dramatis personae stay untouchable. The ladder may fill the 32 blanks.

**Ruling 2 -- THE WEB TIER IS REPLACED, NOT PLUMBED.** In his words: *"just
have the LLM decide -- ask what the likely gender of this person name is, have
the LLM decide, and keep that in an index of names."*

Ruling 2 is the important one, because **it deletes the single defect that
killed both prior rounds.** r2 and r3 both found that the spec passed a
tools/plugins argument to `OpenRouterBackend.generate`, which swallows unknown
kwargs through `**_ignored` -- no error, no search, and a confident answer from
a model that never looked. There is now no web call to be silently swallowed.

On the operator's own question -- *"is it not easy to query a search engine?"*
-- the honest answer is no, and it is worth recording so nobody re-proposes
it: keyless search scraping is blocked and against terms, keyless APIs are thin
and rate-limited, and the RSS precedent does not extend here because a feed is
published to be fetched while a search page is not. The LLM-verdict design
avoids the question entirely and stays offline-first.

## The ladder -- FOUR tiers, still TOTAL

TOTAL means it always returns a castable gender. A voice has to be cast; there
is no "unknown" outcome downstream that does anything but roll blind.

| tier | source | when it answers |
|---|---|---|
| 1 | **Roster** -- the parsed dramatis personae | a KNOWN row exists. Untouchable, and the ONLY tier permitted to answer for a Shakespeare KNOWN row. |
| 2 | **Pronouns in the source text** | a mechanical scan finds an unambiguous pronoun pattern for that character. |
| 3 | **LLM verdict on the character IN ITS WORK, cached** | tiers 1-2 abstain. One question naming the character and its work -- RECALL, not name-shape guessing. Never any source text. |
| 4 | **Name-frequency floor** | everything above abstained. Deterministic, offline, always answers. |

**Tier 4 is the floor and it never abstains.** That is what makes the ladder
total without any tier having to guess beyond its evidence.

**The invented lanes never enter the ladder at all** (standing ruling,
unchanged): `original`, `scifi_news_pro`, `media_archive` roll as they do
today. Their characters do not exist, so a name lookup risks matching a real
person. The announcer likewise stays randomly male/female by design.

## The four blockers, and how each is answered

### B1. "LLM extraction over the FULL unit text" cannot run -- ANSWERED BY NOT DOING IT

r2 measured it: `beckoning_fair_one.txt` is 143,176 bytes and 58 of 65 source
files exceed 12,000 bytes against a ~32,768-token per-call cap.

**Tier 2 is a MECHANICAL scan, not an LLM call.** It streams the text and
matches pronoun patterns near the character's name; it never sends the text
anywhere. **Tier 3 sends only the character's name and its work's TITLE** -- never a
passage, never the work's text. So the size ceiling that killed the old spec is not approached by any
tier in this one. This must stay true: a future "just give the model a bit of
context" is exactly how B1 comes back.

### B2. Blanket surname aliases are identity-unsafe -- ANSWERED BY ABSTAINING

Two roster rows sharing a surname currently produce a confident pin. **A shared
surname must ABSTAIN, not pin** -- the existing `ambiguous_join` verdict
(`nodes/_otr_roster_gender.py:298`) is the right shape and the alias path
should reach it. An abstention costs one step down the ladder; a wrong pin puts
a woman's voice on a man's part, which is the defect class the operator
explicitly keeps open as a CORRECTNESS bug.

### B3. The manifest sequencing deadlock -- ANSWERED BY MOVING THE STAMP

Both r3 lanes found it independently: the stamper was specced to run per-unit
INSIDE the vendor fetch loop, but the manifest is written only AFTER the loop,
so the stamper can never see the unit it was called for.

**The stamp runs as its own pass, after the manifest exists.** It reads the
manifest, resolves each unit's roster, and writes the sidecar. This also makes
it re-runnable over already-vendored units, which the in-loop design could not
be -- and the 32 Shakespeare rows are already on disk, so re-runnability is not
a nicety here, it is the only way to fix them without re-fetching.

**The sidecar carry-forward is a live trap, already documented in the code**
(`nodes/_otr_roster_gender.py:222-231`): the fetcher rebuilds its provenance
dict from scratch and overwrites the file, so without the explicit
carry-forward a routine re-fetch DELETES the gender roster and drops the unit
back to the blind roll -- PBUG-20260815-04, reintroduced by the tool least
likely to be suspected. Any new writer obeys that carry-forward.

### B4. `RosterGenderVerdict` cannot carry the ladder's output -- SMALLER THAN r3 THOUGHT

r3 said adding `gender_source` / `gender_confidence` means "changing every
verdict-construction path". **Re-grounded 2026-08-28: there are exactly SIX
construction sites and all six are in ONE file** --
`nodes/_otr_roster_gender.py:298, 310, 311, 334, 364, 468` -- and ZERO in
`tests/`. With defaults on the new fields the change is ADDITIVE: existing
sites keep working untouched, and only the sites that actually know a source
pass one.

The dataclass today is `gender / evidence / tier / matched` (`:209-219`), where
`tier` already carries a vocabulary (`exact | alias | qualified | contains |
supplement | none`). **Do not overload `tier` with ladder tiers** -- it
describes HOW the roster join matched, which is a different axis from WHICH
ladder rung answered. Add `gender_source` for the rung and keep `tier` as it
is.

**CONSUME the existing `normalize_gender` boundary** (`:100`), do not add a
second normalization path. Note its contract: it maps to `male | female |
other`, while the verdict's `gender` field uses `male | female | unknown`.
That mismatch is real and the rewrite must state which vocabulary the sidecar
records -- resolve it explicitly rather than letting each caller decide.

## Tier 3 in detail -- the LLM verdict and its index

**ASK WITH THE WORK'S TITLE, NOT THE BARE NAME. This is the difference between
recall and guessing, and it is the whole reason tier 3 is trustworthy on the
lane that needs it.**

*"What gender is Malvolio in Shakespeare's Twelfth Night?"* is a KNOWLEDGE
question with a right answer -- the model has read the play many times over.
*"What gender is someone named Malvolio?"* is a guess from the shape of the
word, and a much worse one. On the public-domain and Shakespeare lanes the
character exists in the training data, so the model is RECALLING a fact, not
inferring from name frequency. Tier 4 is where name-shape inference belongs,
and it is deliberately the floor.

So the question carries the work title whenever the lane has one, and the
index records which form was asked -- a verdict recalled from a named work
deserves more confidence than one inferred from a bare name, and a reader of
the index should be able to tell them apart.

**This also right-sizes the operator's worry.** A full hand-built Shakespeare
gender index is not needed, and not because it would be too much work -- it is
only 32 rows -- but because those 32 answers are ones the model already knows
cold, and the index CACHES them permanently after one cheap pass. Thirty-two
entries is small enough to read in one sitting and correct by hand, which is
the real safety property: this is not a black box, it is a short list the
operator can audit.

* **One question, one name (plus its work).** No work text, no passage, no
  context that could grow.
* **The index is the point, not an optimization.** A persistent name -> verdict
  map means each distinct name is asked ONCE, ever. It makes the pass cheap to
  re-run, deterministic across runs for names already seen, and auditable --
  the operator can read the index and correct a row by hand.
* **The index records provenance and is human-correctable.** Each entry keeps
  the model that answered and the date, so a bad row can be found and fixed
  rather than silently re-asked.
* **A refusal or an unparseable answer ABSTAINS to tier 4.** It never guesses
  and never blocks: tier 4 always answers, so a dead LLM degrades the ladder's
  precision, never its totality.
* **Local model, offline-first.** No paid writer is adopted for this
  (2026-08-04 ruling); the local stack answers.

## What this spec does NOT do

* It does not touch Shakespeare KNOWN rows, ever.
* It does not enter the invented lanes.
* It does not send source text to any model.
* It does not add a web/search dependency, a token, or a paid service.
* It does not change how the announcer is cast.
* It does not auto-flip an existing bank gender -- the standing rule is that a
  gender change the operator has heard gets his ear, not an automatic rewrite.
  `glenn` is settled and must not regress.

## Acceptance

1. The 32 `unknown` Shakespeare roster rows resolve to a castable gender, and
   the 53 KNOWN rows are byte-identical before and after.
2. Re-running the stamp over an already-vendored unit is idempotent and does
   not destroy the sidecar's other fields.
3. A shared surname abstains rather than pins, with a test that would fail on
   the old blanket-alias behaviour.
4. No tier ever sends source text to a model -- pinned structurally, the way
   the acceptance grader's import isolation is pinned.
5. The full suite is green and `normalize_gender` remains the only
   normalization path.

## Review routing

This is a rewrite of a spec that already took three NOs, and it is a DESIGN
with more than one defensible answer -- so it gets a real round, not a
finished-diff review. Run it against the r2 + r3 finding lists explicitly:
a reviewer should be asked whether each of B1-B4 is genuinely answered or
merely restated.

## Amendments, 2026-09-02 (after the one review round; see docs/2026-09-02-gender-ladder/driver_anchor.md section 8)

* **Tier 4 is NOT total.** It is the curated first-name pool (`config/cast_pools.gender_of_first_name`)
  on the honorific-stripped first token; unlisted, unisex and descriptive names ("the Creature")
  DECLINE and the render-time 40/40/20 roll remains. Totality of the SYSTEM comes from the roll,
  which is castable; the ladder itself is conservative by design.
* **Every tier-3 ask carries the work's title.** The `asked_as: bare` form is cut; no caller exists.
* **No pronoun scan on scene text.** Measured on the real Folger text the scan mis-gendered LUCE
  from her own lines. Shakespeare unknown rows go supplement -> recall -> name pool.
* **ARIEL and PUCK (and ROBIN, Puck's Folger speech prefix) stay on the roll** as operator-locked
  index entries with an empty gender; the corpus test keeps asserting 40 of 42.
* **The receipt lives in `meta.cast_source_contract.evidence[<NAME>]`** (`gender_source`,
  `gender_confidence`), not on the cast row; `Ledger.set_cast` keeps its fixed row.
* **Confidence vocabulary:** `known` (roster / pronouns / supplement / the fetcher's relation,
  title, group, back_reference), `recalled` (llm_recall), `inferred` (name_frequency).
