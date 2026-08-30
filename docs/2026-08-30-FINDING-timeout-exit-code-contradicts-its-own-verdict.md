# FINDING -- `otr_canonical_api_run.py` exits 1 on a timeout it calls "not a failure at all"

**Status: FLAGGED, NOT FIXED. Deliberately.** Found by reading code, never
observed on a live leg, so under the admission rule in `CLAUDE.md` it does **not**
enter `docs/PROD_BUG_LOG.md` and gets no PBUG id. It is recorded here so it is
not lost.

Found 2026-08-30 by the 5080 while chasing an unrelated 4060 report; independently
verified by the 4060 against the same file.

## The defect

`classify_timeout` (`scripts/otr_canonical_api_run.py:214`) exists specifically to
stop a poll timeout being misread as a dead render. Its own docstring says of the
`still_running` verdict (`:236`):

> "`--timeout` defaults to 5400s and a full wan_ti2v episode on the 16 GB box
> exceeds it, so this is the COMMON case for the slowest lane -- **and it is not
> a failure at all. The episode still publishes.**"

The verdict is used at `:424`, and the `still_running` branch prints, correctly:

> "...BUT THE RENDER IS STILL ALIVE: the server reports N running / M pending.
> This process stopped WATCHING; it did not stop the render, and the episode
> should still publish to otr/obs on its own."

That branch then falls through to `:444`:

```python
    if status != "SUCCESS":
        if err:
            print(f"[canonical-api] ERROR {err}", flush=True)
        return 1
```

So the process **exits 1** on the exact case the file documents as *not a
failure*. A human reads the message and is fine. An automated caller reads the
exit code and records a FAIL for an episode that goes on to publish.

## Blast radius -- and it lands on the worst-placed person

**Neither box is exposed today**, which is precisely why this has never fired:

| caller | timeout | longest observed leg | exposed? |
|---|---|---|---|
| 5080 overnight loop | `--timeout 10800` | 4595 s | no |
| 4060 drill legs | `--timeout 14400` | 121 min (7260 s) | no |
| **anything on the default** | **5400 s** | -- | **yes** |

The 4060 also notes its watcher scores by grepping the leg log for
`RESULT SUCCESS|FAIL` rather than by exit code, so a timeout would fall out of
its regex and surface as "still running" -- prompting an investigation, not a
misreport.

**The exposed caller is therefore an automated one that has not tuned
`--timeout`: a fresh clone, a CI lane, a new user's first soak.** That is also
the person least equipped to work out why a published episode was scored a
failure. A 5-act 12B episode on a throttled 8 GB card legitimately runs past
90 minutes, so the default is reachable in ordinary use, not just pathological
use.

## Why this is not a solo fix

Per the amended review rule in `CLAUDE.md` -- *is there a design choice with more
than one defensible answer?* -- there are three, and each costs something real:

1. **Return 0.** Matches the docstring's "not a failure", but a genuine hang and
   a healthy long render become indistinguishable to a caller. Hides a real
   fault.
2. **Keep 1.** Status quo. Honest that the watch did not confirm success, but
   contradicts the module's own stated semantics and produces false failures.
3. **A third code for "indeterminate"** (e.g. 2, or `EX_TEMPFAIL`). The most
   truthful, and it **changes the exit contract for every existing caller** --
   including the 5080 overnight loop and the 4060 watcher, both of which treat
   any non-zero as failure. Fixing the script alone would not fix the reporting;
   each caller has to be updated in the same change.

Three defensible answers, and option 3 touches both machines' harnesses. That
routes to a panel **before** code, not a solo swing. Flagging with evidence is
the rule-following move here, not the cautious one.

## If it is picked up

Whoever takes it owns the whole set, not just the script:
`scripts/otr_canonical_api_run.py`, the 5080 overnight loop's `rc` handling, the
4060 watcher's scoring, and any CI lane that shells the runner. A change to the
script alone leaves callers reading the new code with the old contract, which is
a worse state than today's consistent-but-wrong 1.
