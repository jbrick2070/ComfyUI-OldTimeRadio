# PROBLEM STATEMENT -- the indextts2 adapter is byte-locked by Lemmy's voice qualification

**Date:** 2026-09-04
**Found by:** the scan-collapse migration, batch (b), when the full suite went red.
**Status:** RULED AND CLOSED (operator, 2026-09-04). Option A stands indefinitely;
option D is not worth an arc and must not be re-opened. See THE RULING below --
it corrects two errors in the analysis that precedes it.
**Nothing is blocked on this.** The migration routes around it and continues.

---

## The one-paragraph version

`nodes/_otr_audio_engines/eng_indextts2.py` cannot be edited -- by anyone, for any
reason, including a comment -- without demoting the shipped Lemmy voice route to
the ordinary cast draw. The file's sha256 is half of a runtime fingerprint that
`nodes/_otr_voice_route.py` compares against the value stored in Lemmy's
qualification record, and a mismatch means "the code that renders this voice is
not the code the operator approved by ear, so stop selecting this route." That
mechanism is correct and was built on purpose. The side effect is that a
spelling-only refactor -- which cannot change one audible sample -- pays the same
price as a change to the seed path.

---

## What actually happened

Batch (b) of the registry scan collapse migrated 13 audio adapters from
`os.environ` / `subprocess` onto the two owners (`nodes/_otr_shared/env.py`,
`nodes/_otr_shared/proc.py`). The edit to `eng_indextts2.py` was six
`os.environ.get(` -> `otr_env.get(` and one `subprocess.Popen(` ->
`otr_proc.popen(`, plus the two-rung import ladder. No env name, no default, no
cast, no argv, no kwarg moved.

Six tests went red:

```
tests/test_voice_identity_fix.py::test_the_shipped_lemmy_route_is_selected_again
tests/test_stale_ledger_voice_guard_removed.py::test_editing_the_shared_dispatcher_no_longer_costs_the_voice
tests/test_make_portable_voice_bank.py::test_exact_exception_does_not_waive_revoked_qualification
tests/test_make_portable_voice_bank.py::test_exact_portable_exception_skips_private_route_and_casts_generic_lemmy
tests/test_make_portable_voice_bank.py::test_missing_private_route_still_fails_closed_with_typo_exception
tests/test_make_portable_voice_bank.py::test_private_route_id_on_wrong_engine_is_present_and_fails_closed
```

Reverting that single file restored all 122 voice tests. The other 12 adapters
migrated cleanly and shipped.

**The failure mode in production would NOT have been a crash.** A fingerprint
mismatch DEMOTES -- by the module's own law, "an audit may never fail an episode;
a render degrades." Lemmy's cast row would have taken the ordinary voice draw and
the episode would have published to `otr/obs/` looking completely normal. The
only signal is a warning in the server log. That is the part worth pausing on:
this class of defect ships silently and is discovered by ear, later.

## The mechanism, exactly

`nodes/_otr_voice_route.py`:

```python
RUNTIME_FINGERPRINT_SOURCES = {
    "indextts2": (
        "nodes/_otr_audio_engines/eng_indextts2.py",
        "scripts/_otr_indextts2_worker.py",
        "nodes/_otr_resolved_request.py",
    ),
}
```

`live_engine_impl_version("indextts2")` reads each file's bytes, normalizes CRLF
and lone CR to LF, sha256s each one, then sha256s the joined
`<path>:<digest>` lines and keeps the first 16 hex characters.
`select_policy_route` compares that against
`qualification_record.runtime.engine_impl_version` on the route.

Current state, measured just now:

| | |
|---|---|
| live `indextts2` fingerprint | `d47779386ce91209` |
| stored on the shipped Lemmy route (`config/cast_pools.py:847`) | `d47779386ce91209` |
| verdict | MATCH -- the route selects, and the tree is back to green |

This is not an accident of implementation. The module's own header says it
plainly: the field was "stored, described, and never read" until 2026-08-18, so
"a route qualified by ear in August stayed 'qualified' through every subsequent
edit to the code that produced the sound the operator actually approved." Closing
that gap is the whole point of the mechanism.

## Blast radius: exactly one file, and it is bounded

Checked, not assumed:

* `RUNTIME_FINGERPRINT_SOURCES` is the **only** mechanism in the repo that hashes
  Python **source bytes** at runtime. Swept `nodes/` and `config/` for any other
  code that reads a `.py` and digests it -- there is none. The other `sha256` /
  `impl_version` fields found (`cast_lock` bank source, `_otr_engine_profiles`
  profile source, `_otr_readiness` TTS text, `_otr_audio_cache` keys) all hash
  CONTENT, never code.
* `OTR_LedgerScriptWriter.py`'s "moved ... byte-identically, with a sha256 per
  block in the commit" is **historical prose** about the 2026-08-23 lean-mean
  split -- a one-time verification method, not a live gate. That file is free to
  migrate in batch (d).
* Of the three files in the recipe, only one is in the migration's path:
  * `nodes/_otr_audio_engines/eng_indextts2.py` -- **BLOCKED** (6 env reads, 1 spawn);
  * `scripts/_otr_indextts2_worker.py` -- `scripts/` is explicitly out of scope for
    this arc;
  * `nodes/_otr_resolved_request.py` -- has **zero** env or spawn sites, so the
    migration never touches it.
* Only `indextts2` has a recipe at all. Every other engine is absent from the map
  and is therefore never demoted on this ground.
* The second stored fingerprint (`b965453f355661a3`, `config/cast_pools.py:993`)
  is under `superseded_native_routes` -- evidence, deliberately never selectable.
  Unaffected.

## What it costs to leave it blocked

The registry scan's env rule fires **once per FILE**; the subprocess rule fires
**once per CALL SITE**. So this one file leaves behind:

* **1** `python_environment_manipulation` finding -- and it carries the
  `credential-access` tag, so the collapse's headline result becomes "the
  credential-access tag on TWO files" instead of one;
* **1** `python_command_injection_risk` finding.

Against a projected floor of about 9 findings, that is roughly a 20% miss on a
number whose whole purpose is to be small enough for a human reviewer to read in
one screen. Real, but not fatal, and not worth a voice the operator approved.

## What I did, so it is not rediscovered

* Reverted `eng_indextts2.py`. It stays in both ratchet PENDING sets -- it still
  offends, and the guard still says so.
* Added a `BLOCKED` table to `tests/test_env_single_owner.py` and
  `tests/test_process_single_owner.py` naming the file, the reason, and what
  unblocks it, plus a test asserting a blocked file is still pending and still
  exists. A future batch that sweeps it up mechanically now has to delete a
  paragraph that explains what that costs.

## THE RULING (operator, 2026-09-04): A STANDS. D IS NOT WORTH AN ARC. DO NOT OPEN IT.

Whole-file hashing of the IndexTTS2 adapter stays exactly as it is.
`eng_indextts2.py` is not migrated until the next Lemmy re-audition that was
wanted for some other reason. Two scanner findings is the price, and it is paid.

**This section exists because the analysis above got the central point WRONG,
and a future window must not repeat it.** The memo framed D -- "make the hash
smarter so a non-audible edit stops demoting" -- as the one untried option that
would fix the class rather than the instance. That is false. The class-level fix
ALREADY SHIPPED, and the incident that triggered this one is the third of its
kind, not a new class.

### D already shipped, in the only form that is safe

`6f509b16` (2026-08-19), *"The fingerprint landmine is defused, and the fix was
the recipe not the mechanism"*. It did the safe D and refused the unsafe one:

* **Changed WHICH FILES are in the recipe, decided by measurement.**
  `_otr_voice_node_common.py` had produced 18 false demotions against 1 true one
  in 60 days, and the one true event also touched the adapter -- so dropping the
  shared dispatcher lost nothing real.
* **Left whole-file hashing alone.** AST hashing was measured the same day:
  **43 of 44 commits still moved the hash.** A comment plus a log line is a new
  AST statement. It would not have saved the incident that prompted the
  narrowing.

What is left of D is a different invention -- making the ADAPTER'S OWN BYTES
ignore "non-audible" edits -- and that is precisely the mechanism the module
header already rejects: *a fingerprint that under-reports is a false claim of
proof; one that over-reports is an inconvenience.*

### AST hashing would not even have saved THIS edit

The migration replaces `os.environ.get("X")` with `otr_env.get("X")`. In the AST
those are not the same shape:

```
os.environ.get(...)  ->  Attribute( Attribute( Name('os'), 'environ' ), 'get' )
otr_env.get(...)     ->  Attribute( Name('otr_env'), 'get' )
```

An AST hash still moves. To make D pass the edit that raised it, you would need
a semantic-equivalence normalizer that understands module aliases, re-exports and
argument forwarding -- high complexity, zero runtime gain, and a new mechanism
that can be wrong in the fail-open direction.

### Why a function-subset hash fails in the dangerous direction

The adapter is not a high-churn shared file; it is the engine-specific rendering
code, and it still owns things a listener can hear: `OTR_INDEXTTS2_FP16`, the
worker script path, and which Python binary starts the sidecar. A "hash only the
synthesize function" rule misses those and **fails OPEN** -- a route that still
says "qualified" after the code that made the sound has moved. That is
PBUG-20260817-09's class over again, which is the defect this whole mechanism
was built to end.

### This is the THIRD time, and the rule was already written down

`2abb6d86` (2026-08-31) branched `_venv_python` so IndexTTS2 could start on
Linux. Windows was measured BYTE-IDENTICAL in behaviour. The fingerprint moved
anyway (`d47779386ce91209` -> `c1b64d5c5f6c2f9f`), Lemmy dropped to the ordinary
draw, and the episode still published. `327e6004` reverted it, and its message
states the rule in one line:

> on a fingerprinted engine, "same behaviour" is not the same claim as "same bytes"

Today's env/spawn spelling swap makes exactly that claim: no name, default, argv
or kwarg moves, and the fingerprint must still move, because it hashes the CODE,
not the behaviour. Reverting the file was not a workaround. It was obedience to a
rule already in the history.

Since the 08-19 re-record, every adapter edit that would have moved the hash was
either reverted (the Linux path) or netted back to the fingerprinted bytes (the
2026-09-01 error-text / packaging pair). The live hash still matches the stored
`d47779386ce91209` because the adapter's bytes are still the 2026-08-19 bytes.

### The verified boundary question, and why it is an argument FOR A

Checked against the live map, not inferred: **`nodes/_otr_shared/env.py` and
`nodes/_otr_shared/proc.py` are in NO fingerprint recipe.** So if the adapter
were migrated, its six knob reads -- including `OTR_INDEXTTS2_EMO_ALPHA` and
`OTR_INDEXTTS2_EMO_MASS_CAP`, the emotion blend the module header calls one of
the two things a listener judges -- would route through files the fingerprint
does not watch.

The knob NAMES and DEFAULTS stay in the hashed adapter, so the audible decision
stays covered today. But that is only true while the owners keep their contract
of live, uncast, undefaulted reads, and nothing ties that contract to the
fingerprint. Adding the owners to the recipe is not the answer either: they are
shared files that change for unrelated reasons, which is the shared-dispatcher
mistake -- 18 false demotions in 60 days -- put back on purpose. **Extracting
behaviour out of a hashed file into an unhashed helper is a cost of migrating
this adapter, not a reason the hash is wrong.**

### The three options that remain, and what each is for

* **A -- leave it blocked until an audition is wanted anyway. STANDING ANSWER.**
  Cost: 2 findings. Risk: zero. The shipped route selects cleanly and all 122
  voice tests are green.
* **B -- re-record the fingerprint without an ear. NEVER.** That is the lie the
  2026-08-18 gate was built to end. Note that the 08-19 rewrite of the stored
  value from `9bee950a7920fd00` to `d47779386ce91209` was a RECIPE change with
  UNCHANGED adapter bytes -- not a precedent for bumping the number after a real
  edit.
* **C -- migrate and re-audition. Honest, and a bad trade on its own.**
  **CORRECTION to the earlier draft of this memo: re-auditioning does NOT unlock
  the file permanently.** Under byte comparison the very next edit re-blocks it.
  C resolves one revision; it does not end the maintenance cost. Bundle it with a
  Lemmy listen already wanted -- then the spelling swap rides along, the two
  findings disappear, and the new hash is a real claim.

## DOES THIS COST THE CLEAN REGISTRY ENTRY? NO. Not one step of it.

The operator's actual goal is a clean Comfy Registry listing, so this is the
question that matters, and the answer is measured rather than hoped:

**The collapse was never going to reach zero findings, with or without this
adapter.** The closed plan says so in its own opening
(`kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md`):

> The gate is ZERO findings or a manual admin approval; nothing here reaches zero
> (ffmpeg is a subprocess and that is the render path).

A pack that renders video runs ffmpeg. ffmpeg is a subprocess. The scanner flags
subprocess calls. There is no version of this pack that scans clean, and no
amount of migration changes that -- which is exactly why the plan's target was
never "zero" but "about nine lines instead of 158".

So the route to an Active listing is the MANUAL ADMIN REVIEW (GO_FORWARD item 4),
and `docs/GO_FORWARD_PLAN.md` already records that there is no publisher
self-service alternative:

> STILL FLAGGED IS THE EXPECTED OUTCOME for (a), not a failure. There is no
> publisher self-service route to Active.

**What the blocked adapter actually costs, precisely:** the human reviewing that
request reads about ELEVEN lines instead of about NINE, and the
`credential-access` tag appears on TWO files instead of one. That is the entire
delta. It does not change whether the review can be filed, whether it can be
approved, or how long it takes -- a reviewer who is reading nine lines is not
turned back by eleven.

**If the review is ever refused ON THESE TWO FINDINGS SPECIFICALLY**, that is new
evidence and it reopens option C on its own merits -- migrate the adapter and pay
for an honest audition, because then the two findings are buying something real
instead of buying two lines of report. Until a reviewer actually says so, paying
a GPU leg and the operator's ear up front for two `info` lines is a bad trade,
and the ruling above stands.

## Standing policy

1. **A stands indefinitely.** `eng_indextts2.py` stays in the `BLOCKED` tables of
   `tests/test_env_single_owner.py` and `tests/test_process_single_owner.py`.
2. When Lemmy's voice or IndexTTS2 next gets a GPU audition for real voice or
   model work, the migration (6 `otr_env.get` calls, 1 `otr_proc.popen` call)
   rides along with it and the new fingerprint is stamped behind an honest ear
   check.
3. **Do not open D.** Do not re-raise it as a fresh finding. The measurement that
   settled it is `6f509b16`, 2026-08-19.

## Grounding

* `nodes/_otr_voice_route.py` -- `RUNTIME_FINGERPRINT_SOURCES`,
  `live_engine_impl_version`, `stale_runtime_fingerprint`, and the header block
  that states the doctrine.
* `config/cast_pools.py:847` -- the shipped Lemmy route's stored value; `:882`
  the historical 4-file value; `:993` the superseded route.
* `6f509b16` -- the recipe narrowing, and the AST measurement (43/44) that
  refused the mechanism change.
* `2abb6d86` -> `327e6004` -- the Linux `_venv_python` branch and its revert;
  the "same behaviour is not the same claim as same bytes" line.
* `docs/PROD_BUG_LOG.md` -- PBUG-20260817-09, the defect the mechanism closed.
* `kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md` -- the closed
  migration plan, which did not know about this constraint.
