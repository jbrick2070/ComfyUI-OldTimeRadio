# Title provenance -- tag every hand the title passes through

**Operator, 2026-08-17:** *"we need to tag episode_title and title in an episode
run ledger or some artifact so we can compare"* ... *"along with a timestamp of
when it was stamped, and telemetry so we catch future drift."*

**Status: SPEC ONLY. No code written.** The exact hops to instrument come from
the in-flight title trace; this document fixes the SHAPE so that trace can be
dropped straight into it.

---

## Why the field we already have cannot do this

`meta.title_source` exists and today reads `"user"` or `"llm_post_composition"`.
It is **one slot**, so it records only the LAST writer. That is why the harness
label reached a published title card without anything noticing: the value was
overwritten and the receipt simply reported the winner, with no trace of the
title it replaced.

**A single field cannot catch drift. A CHAIN can.** Drift is by definition a
value changing hands, so the receipt has to record the hands.

## The precedent to copy -- this repo already does this once, well

`_otr_source_identity.SourceIdentity` carries a `provenance` map, and its own
docstring states the reason:

> *"`provenance` names the exact meta path each populated field came from, which
> is what makes a wrong value fixable at its source instead of argued about
> downstream."*

Same idea, one level richer: the title needs **when** and **who** as well as
**where**, because the title is written more than once per episode while a
source identity is derived once.

## The shape

`meta.title_provenance` -- an **APPEND-ONLY list of stamps**, oldest first. Every
code path that sets or replaces the title appends one entry and never edits an
earlier one.

```json
{
  "stamps": [
    {
      "value": "CHUNKB ACCEPT forced lemmy scifi_news_pro",
      "source": "user",
      "stage": "launch_input",
      "symbol": "OTR_LedgerScriptWriter.run:episode_title_widget",
      "at": "2026-08-16T18:52:34Z",
      "replaced": null
    },
    {
      "value": "The Blackwood Enigma",
      "source": "llm_post_composition",
      "stage": "post_composition_title",
      "symbol": "<the generator symbol the trace names>",
      "at": "2026-08-16T19:04:11Z",
      "replaced": "CHUNKB ACCEPT forced lemmy scifi_news_pro"
    }
  ],
  "final": "The Blackwood Enigma",
  "final_source": "llm_post_composition",
  "writes": 2
}
```

**Field by field, and why each earns its place:**

| field | why it exists |
|---|---|
| `value` | the title as of this stamp -- the thing being compared |
| `source` | the vocabulary already in `title_source`; keep the SAME strings so old ledgers stay readable |
| `stage` | pipeline stage, so two writers in the same module are still distinguishable |
| `symbol` | the function that wrote it. **Symbol, never a line number** -- they rot |
| `at` | UTC ISO-8601. The operator asked for this explicitly, and it is what makes "when did this start" answerable |
| `replaced` | the previous value, or `null` on first write. **This is the drift detector**: a stamp whose `replaced` is non-null is a title changing hands |
| `writes` | count, so a one-line query finds episodes where the title was touched more than expected |

`title_source` and `episode_title` **stay exactly as they are.** This block is
additive; nothing downstream has to change to keep working, and the two must
agree (see the check below).

## The telemetry -- what "catch future drift" means concretely

Three checks, all deterministic, all cheap, none of which may ever fail an
episode (THE LAW: an audit may improve a story, never fail one):

1. **AGREEMENT.** `meta.episode_title == title_provenance.final` and
   `meta.title_source == title_provenance.final_source`. If these disagree,
   something wrote the title without stamping -- **the receipt is lying and that
   is the loudest possible signal.**
2. **UNSTAMPED-WRITER DETECTION.** `writes` of 0 on an episode that has a title
   means a writer exists that this system does not know about. Log LOUD.
3. **A DRIFT LEDGER, corpus-wide.** For every episode, emit
   `(episode_id, final_source, writes, replaced-chain)`. That single table
   answers the questions we could not answer today without a corpus scan:
   * how many published episodes carry a launch label as their title;
   * **when the rate changed** -- which is exactly the question the item I
     roundtable had to reconstruct by hand, and it would have been one query;
   * whether a new writer appeared after some date.

## What this would have caught

* **The published harness title cards.** A stamp with `source: "user"` surviving
  as `final` on an episode that published is the whole defect, visible in one
  field without watching a video.
* **Item I's August regression.** The roundtable measured a 6.8% -> 50% jump by
  writing throwaway detectors against 1,700 ledgers. With a drift ledger it is a
  `GROUP BY month`.
* **A future silent overwrite.** Any new pass that starts writing the title
  appears as a third stamp the day it lands.

## Rules for whoever builds it

* **Append, never mutate.** A stamp is a fact about a moment; editing one
  destroys the only evidence of the hand-off.
* **Stamp at the WRITE, not at the end.** A single stamp written during
  finalization can only ever record the winner -- the exact failure this replaces.
* **Never let stamping raise.** A telemetry block that can kill a render is worse
  than the defect it measures. Degrade, never raise.
* **`otr/obs/` volume must not drop.** Operator ruling, same day: publication is
  the success signal, so no check here may gate a publish.
* **Same vocabulary as `title_source`.** Do not invent a parallel set of source
  names; a second vocabulary is a second authority, which is the shape behind
  three separate defects logged this week.

## Open, for the trace to answer

1. Every symbol that writes a title today (the in-flight trace enumerates these).
2. Whether the harness label should stamp as `"user"` or gain its own
   `"harness_label"` source -- the latter makes "is this a real episode" a field
   lookup rather than a vocabulary guess.
3. Whether `title` (the other field the operator named) is a distinct value or an
   alias of `episode_title` -- **not proven yet**, and it decides whether this
   block tracks one value or two.
