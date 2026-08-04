# D1 PLAN -- make the "still unmaterialized" failure name its own branch

> **SUPERSEDED 2026-08-04 by the shipped state:**
> `kibitz-runs/2026-08-04-d1-still-unmaterialized/r3/final.md`.
> Read that first. This draft is kept for the arc record only. Two things in it
> are WRONG and were corrected by the r2/r3 panel: **D1d was CUT** (it could not
> preserve the refused prompt, and `CanonicalImage` forbids the field), and the
> acceptance criterion "one 320-word leg still publishes" was replaced (gating
> an observability change on a run with a ~1-in-6 failure rate is a coin flip).

Companion to `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`.
Scope: OBSERVABILITY ONLY. No behavior change, no fix to the suspected branch,
no weakening of any gate. The bug is stochastic (~1 in 6 at 320 words) and its
explanation is currently destroyed four different ways; this plan stops the
destruction so the next occurrence is self-diagnosing.

## The incident in one paragraph

A 320-word Shakespeare still leg failed in `OTR_ImageGenDispatcher` with
`required scene image targets missing or unmaterialized before video dispatch:
still_b007, still_b008`. Nineteen of twenty-one scene stills materialized; two
did not. The relaunch published 11/11 legs including the same profile at the
same word count, so this is stochastic, not deterministic.

## Branch algebra -- why only ONE silent exit survives

Both objects PRINTED the dispatcher resolve line:

    [OTR_ImageGenDispatcher] resolve: object=still_b007 kind=scene_character
      role=character_video -> slot=character_image_model engine=z_image_turbo

That single log line is load-bearing evidence, because it is emitted at
`otr_image_gen_dispatcher.py:873` -- AFTER the capability gate and AFTER engine
resolution. Walking the per-object loop (`:823-1098`), every exit that leaves no
ledger row:

| Site | Exit | Status for b007/b008 |
|---|---|---|
| `:824` | not a dict -> `continue` | EXCLUDED -- object exists, resolve printed |
| `:843` | empty `oid` -> `continue` | EXCLUDED -- oid printed |
| `:845` | unknown role -> **raise** | EXCLUDED -- no raise naming these |
| `:850` | capability `None` -> **raise** | EXCLUDED -- no raise |
| `:860` | capability false -> `log.info` + `continue` | EXCLUDED TWICE: resolve prints later, and capability is per-ROLE, so all `character_video` beats would skip together (b002-b005 materialized) |
| `:880` | no `engine_id` -> warn + `continue` | EXCLUDED -- resolve printed `engine=z_image_turbo` |
| **`:893`** | **`_assert_not_path` ValueError -> `warnings.append` + `continue`** | **THE ONLY SURVIVOR** |
| `:901-949` | cache hit | EXCLUDED -- always appends a row, or falls through to fresh render |
| `:953/:972/:991` | engine/adapter/gen_fn unusable -> **raise** | EXCLUDED -- no raise |
| `:1017-1049` | render failure -> **raise** | EXCLUDED -- no raise |

So either `_assert_not_path` refused two prompts, or the loop has an exit not on
this map. **The panel's highest-value job is to attack this table.**

## Why `_assert_not_path` is more likely than it looks

`otr_image_gen_dispatcher.py:211-223`:

    looks_pathy = (
        os.sep in p or (os.altsep and os.altsep in p)
        or p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )

**On Windows `os.altsep` is `/`.** Verified live: `os.sep='\\' os.altsep='/'`.
So this guard refuses any prompt containing a single FORWARD SLASH anywhere.
Composed image prompts plausibly carry one -- "black/white", "sci-fi/noir",
"1950s/60s", a fraction, a ratio. The guard's name and docstring say "path vs
prompt socket crossing"; its predicate is far broader than that intent on this
platform. That gap is the strongest available hypothesis, and it is UNPROVEN.

Counter-evidence that keeps it unproven: the failed episode's frozen dialogue,
cast `appearance` values, and the `recur_frac` style tails are all slash-free
(checked live). The composed prompt is built by `OTR_MetaBriefImagePromptGen`
(node 89) from more than the dialogue, so the slash could enter from the setting
string, an era tail, or a directive -- none of which survive to disk.

## Four mechanisms destroy the evidence (all four are D1 targets)

1. **Wire-only warnings.** `warnings.append(...)` at `:895` lands in
   `ledger["images"]["warnings"]`, stamped at `:1132` -- AFTER the completion
   gate raises at `:1128`. The explanation is discarded by the very error that
   needs it.
2. **The raise carries no evidence.** It reports object ids only, not
   row-missing vs file-missing, and not the warnings mentioning those ids.
3. **Boot-log truncation.** `scripts/_otr_soak_server_launch.cmd:139` redirects
   with `>`, so the 23:06 reboot destroyed the 22:11 server log.
4. **The prompt string is never persisted.** Ledger image rows carry
   `prompt_hash`, never `prompt` (verified: 0 prompt strings across 12 episode
   ledgers). Even a surviving ledger cannot show what was refused.

## The changes

**D1a -- the completion gate raises with its evidence.**
`otr_image_gen_dispatcher.py:1104-1130`. For each missing target, classify and
report: `no_row` (never appended) vs `dead_path` (row exists, file absent, path
quoted). Append every accumulated `warnings` entry whose text mentions that
object id. Keep the exception type (`ImageRenderError`) and keep failing closed.

**D1b -- the two silent-skip branches log at skip time.**
`:881` and `:895` currently only `warnings.append`. Add a matching
`log.warning("[OTR_ImageGenDispatcher] SKIP %s: ...", oid, ...)` so the reason
reaches the server log immediately, independent of whether the wire survives.
For `:895` include the offending prompt's first 200 chars and which separator
matched -- that alone would have closed this investigation.

**D1c -- the boot launcher stops destroying the prior run's log.**
`scripts/_otr_soak_server_launch.cmd`. Before the `>` redirect, if `%1` exists,
rename it to `<name>.<timestamp>.log`. **The caller contract is preserved** --
the fresh log still lands at exactly `%1`, so every existing harness that reads
that path is unaffected. This is the safest form of the change; do NOT
repoint `%1` itself.

**D1d -- persist the prompt string on the image row.** Add `"prompt": prompt`
alongside the existing `prompt_hash` on the row built at `:1069`. Additive
only. If size is a concern, cap at 1000 chars with a `prompt_truncated` flag.
*Open question for the panel:* does any consumer diff whole image rows, or
assert an exact row key set, such that an additive key breaks it?

## Explicitly NOT in scope

- Do **not** loosen `_assert_not_path`. It guards a real defect class. If D2
  confirms it, the fix is D3 and belongs where the slash ENTERS the prompt --
  or in a predicate that detects actual paths rather than any separator.
- Do **not** build "the collapse guard mints a still". Its premise is dead: 70
  whiffs across 11 published episodes.
- Do **not** revive the portrait-init fallback (deliberate rip, 2026-06-18).
- Do **not** weaken or bypass the completion gate. It worked.

## Acceptance

- Full suite green (`pytest -q -p no:cacheprovider`, ~8374 passing baseline) plus
  the Bug Bible regression.
- AST parse on every touched `.py`; UTF-8 no BOM; canonical workflow JSON
  untouched (no node/widget/link change in this plan -- if that turns out to be
  false, it is a finding, not a silent edit).
- A deliberately slash-bearing prompt in a unit test produces a raise whose text
  names the object, the branch, and the separator.
- One live 320-word Shakespeare still leg still publishes (proves no regression).

## D2/D3 (not this change)

D2: reproduce at 320 words, ~1 in 6, now self-diagnosing. D3: fix the branch
D2 names, at its root. `PROD_BUG_LOG.md` gets its entry when the mechanism is
confirmed -- so it records a mechanism, not a guess.
