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

## `scifi_news_pro` -- probed, and the answer was not the one I expected

Two diagnostic legs were run specifically to separate "this lane is broken" from
"three characters do not fit in 30 words here". The hypothesis was that forcing
the cameo pre-locks a THIRD speaking character and the lane's markup validator
cannot fit three voices into 30 words -- which would have made it a non-defect.

**That hypothesis is wrong.** The matrix:

| leg | words | cameo | outcome |
|---|---|---|---|
| sweep | 30 | **forced** | writer dies: markup ladder exhausted, `BAD_LINE` |
| probe A | 30 | natural roll | writer + casting + audio OK; **video** dies: no still for `music_closing_001` |
| probe B | 90 | **forced** | writer dies: markup ladder exhausted, `BAD_LINE` |

Reading it:

* **The word budget is not the factor.** Forcing the cameo breaks the writer at
  30 AND at 90 words. Raising the budget changes nothing.
* **The cameo IS the factor for the writer failure.** With the cameo on its
  natural roll the writer sails through -- and so does the entire casting and
  audio chain, nodes 1/62/63/80-83 all executed.
* **There is a SECOND, unrelated defect underneath.** Probe A got past the
  writer and then died at node 92 `OTR_VideoRenderBatch`:
  `still-spine handoff missing materialized scene still for shot
  shot_music_closing_001 beat music_closing_001 engine still_flat`. Five other
  banks on the SAME `otr_w45_still_flat` profile produced that still and
  published fine.

Both are now recorded: **PBUG-20260811-01** (forced cameo kills the
`scifi_fable2` writer, reproduced twice) and **PBUG-20260811-02** (missing
closing-music still, seen once). Neither root cause is established, and both say
so rather than guessing.

**One reachability point I owe, because it is my own change.** `lemmy_cameo` was
whitelisted for headless drivers in `baf338ee` so a qualification run could force
the cameo deterministically. That did not create PBUG-01 -- the widget always
existed and the GUI could always set it -- but it made the failure reachable from
the sanctioned runner, which is how it surfaced. Four other banks force the cameo
without trouble, so the whitelist is not the thing to revert.

## What this sweep did prove

Branch A works end to end in production, not just in tests. On both cameo-allowed
lanes Lemmy was cast on the qualified route with the reference the operator picked
blind on 2026-08-10 -- `idx_lemmy_algenib_cockney_v1`, route
`lemmy-indextts2-algenib-cockney-v1` -- through the real canonical workflow, and
both episodes published to `otr/obs/`. The last unticked Branch A acceptance row
is now ticked.
