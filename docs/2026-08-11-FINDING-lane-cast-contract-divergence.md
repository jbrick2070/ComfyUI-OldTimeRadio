# FINDING -- two lanes write no cast contract, and nobody would have noticed

**Date:** 2026-08-11. **Found by:** a six-bank live render sweep, not a test.
**Status:** OPERATOR DECISION OWED. No production code changed on the strength of
this; the only change made was correcting a claim I had written without measuring.

## What was run

One 30-word `otr_w45_still_flat` render per runnable source bank, with the LEMMY
cameo **FORCED** (`lemmy_cameo="always include"`) on every leg. Forcing it on all
six is the point: "always include" should mean two different correct things
depending on the bank, and one sweep then proves both halves instead of only the
happy one.

| bank | result | `lemmy_policy` | verdict |
|---|---|---|---|
| `original` | LEMMY cast, qualified IndexTTS2 route | `operator_cameo` | PASS |
| `media_archive` | LEMMY cast, qualified IndexTTS2 route | `operator_cameo` | PASS |
| `public_domain` | no LEMMY | `source_fidelity_exclusion` | PASS |
| `shakespeare` | no LEMMY | `source_fidelity_exclusion` | PASS |
| `scifi_news` | no LEMMY | **none recorded** | **FINDING** |
| `scifi_news_pro` | writer crashed before casting | n/a | separate issue |

The two PASS/exclusion rows are the ones worth dwelling on: absence alone proves
nothing, because the cameo is an ~11% roll and "no Lemmy" is the DEFAULT outcome.
A broken exclusion and a working one look identical unless the ledger says a
decision was taken. Those two say it.

## The finding: `scifi_news` writes an EMPTY cast contract

    original     cast_contract keys: cast_seed, cast_seed_source,
                 casting_attempts, lemmy_hit, lemmy_policy,
                 num_characters_locked, num_characters_request
    scifi_news   cast_contract keys: (none)

`scifi_news` runs the `scifi_news_circuit` pipeline, which never calls
`_otr_casting.lock_cast()` -- the function that stamps the cast contract and
handles the cameo. Three consequences, all silent:

1. **`lemmy_cameo` is ignored.** "always include" produced no Lemmy and no
   refusal. The operator's setting had no effect and left no trace.
2. **`num_characters` is ignored.** The leg asked for 2 and got 3 (Ada, Kai,
   Dr. Elara).
3. **No `cast_seed` is recorded**, so that lane's casting is not reproducible
   from the ledger the way every other lane's is.

**This may be entirely correct.** A news-circuit lane plausibly owns its own
cast the way a source-faithful adaptation does. What is NOT correct either way is
that it is unrecorded: `public_domain` refuses the cameo and says so; `scifi_news`
refuses it and says nothing. An absence with no stated reason is indistinguishable
from a bug, which is the whole thesis of the voice-route work this sweep was
validating.

**Operator question:** is `scifi_news` (and its `scifi_news_pro` sibling) meant to
support the cameo and the character count at all? If yes, the circuit lane needs
to route through `lock_cast`. If no, it should stamp a policy saying so --
`lane_owns_its_cast` or similar -- so the ledger records a decision rather than a
silence.

## The correction I had to make to my own work

`BANK_CAMEO_POLICY` in `tests/test_cast_lock_policy_repin.py` asserted
`scifi_news: cameo_allowed`. I wrote that map hours earlier from the bank list
**without measuring anything**, and the sweep disproved it. It now records what
was observed, and marks `scifi_news_pro` explicitly `unmeasured` rather than
inheriting an assumption from its sibling -- assuming from a sibling is what made
the map wrong the first time.

## The separate issue: `scifi_news_pro` fails in the writer

    node 1 OTR_LedgerScriptWriter
    [scifi_fable2] pass 'script' failed after 4 attempt(s):
    markup ladder exhausted; last defects: - BAD_LINE

It died in the writer, before any casting ran, so nothing about the voice route or
the cameo is implicated. Untested hypothesis worth one cheap experiment: forcing
the cameo pre-locks a THIRD speaking character, and this lane may not fit three
voices into 30 words past its markup validator. Re-running it at 30 words WITHOUT
the forced cameo, and at a longer budget WITH it, separates "this lane is broken"
from "three characters do not fit in 30 words here" -- very different findings,
and the second is not a defect at all.

Not admitted to `PROD_BUG_LOG.md`: the admission rule wants a verified production
failure, and this one has a plausible benign explanation that has not been ruled
out. Re-run first, then log it if it survives.

## What this sweep did prove

Branch A works end to end in production, not just in tests. On both cameo-allowed
lanes Lemmy was cast on the qualified route with the reference the operator picked
blind on 2026-08-10 -- `idx_lemmy_algenib_cockney_v1`, route
`lemmy-indextts2-algenib-cockney-v1` -- through the real canonical workflow, and
both episodes published to `otr/obs/`. The last unticked Branch A acceptance row
is now ticked.
