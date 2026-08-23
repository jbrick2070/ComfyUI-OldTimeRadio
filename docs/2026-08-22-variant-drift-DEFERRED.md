# DEFERRED (operator, 2026-08-22): `build_variants.py --check` fails on the ship candidate

**Status: RECORDED, NOT CHASED.** The operator said `defer` when this surfaced.
It is written down so the next window does not have to rediscover it, and so
nobody runs the generator without knowing what it would do.

## The observation, measured

At HEAD `c2381e1d` on `v2.0-alpha`:

    python scripts/build_variants.py --check
    CHECK FAIL: otr_ghost_signal_v3.json: DRIFT vs regeneration
                (variants are generated, never hand-edited)
    CHECK FAIL: otr_ghost_signal_v3.launch.md: recipe DRIFT vs regeneration
    check: 54 variants, 2 failures

**The plan's recorded baseline is 54 variants / 0 failures**
(`docs/GO_FORWARD_PLAN.md`, the Ghost Prompt v2 gate block). So this drift
appeared AFTER that gate, on this branch.

## Why it matters, and why it is not merely cosmetic

`config/profiles/otr_ghost_signal_v3.json` is the **presumptive ship
configuration** -- operator, 2026-08-22 evening: *"i think v3_haunted 2.1 will
ship and we may ditch the rest we shall see."*

Two commits edited that file directly:

* `ab0a7809` -- "Ghost v3 profile: promote the technical slot to
  google/gemma-4-12b-it"
* `2b3d3dd0` -- "Ship intent: v3_haunted + Prompt v2.1, and the dropdown now
  says so"

`build_variants.py` holds that variants are GENERATED and never hand-edited
(`scripts/build_variants.py:341-347`).

**THE HAZARD, and it is the reason this is worth a file:** if the committed
variant is hand-edited while its generator recipe is not, then running
`scripts/build_variants.py` WITHOUT `--check` would regenerate the file from the
recipe and **silently revert the operator's ship-candidate settings** -- the
promoted `technical_model` and the dropdown label that exists precisely because
"the default being non-obvious is exactly what cost a day of confounded
comparisons".

`verify:` that hazard is a READING of the two failures, not a measurement. It
was NOT confirmed by diffing a regeneration against the committed file, because
the operator deferred the item before that was run. **Anyone picking this up
should start there**: regenerate to a scratch path and diff, and find out
whether the fix is (a) update the generator recipe to match the shipped
settings, or (b) the committed file is right and the recipe is stale.

## What this is NOT

Not a render defect. Nothing observed suggests a published episode is wrong --
the live Ghost episodes in `otr/obs/` were rendered from the committed profile
as it stands. This is a build-reproducibility gate failing, with a revert trap
behind it.
