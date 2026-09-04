# PROBLEM STATEMENT -- the indextts2 adapter is byte-locked by Lemmy's voice qualification

**Date:** 2026-09-04
**Found by:** the scan-collapse migration, batch (b), when the full suite went red.
**Status:** WORKED AROUND, NOT SOLVED. The workaround is safe and costs 2 registry
findings. The underlying question is a design call and is the operator's.
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

## The options, with honest costs

**A. Leave it blocked until the next Lemmy re-audition happens anyway.**
*(what is in place now)* Cost: the 2 findings above. Risk: zero. The migration
rides along with the next audition instead of forcing one. **Recommended as the
default** -- it spends nothing and loses nothing that cannot be recovered later.

**B. Migrate it and re-record the fingerprint without re-auditioning.**
Cost: a false claim of proof. The record would assert that this code was approved
by ear when it was not, which is exactly the failure the mechanism was built to
end. **Do not do this.** It is listed only because it is what a hurried future
window would reach for, and the `BLOCKED` table exists to stop that.

**C. Migrate it and re-audition the route.**
Cost: one GPU audition plus the operator's ear. Buys the 2 findings and unblocks
the file permanently. Sensible **only if bundled with a re-audition already
wanted for another reason** -- paying a GPU leg and an ear to remove two `info`
findings is a bad trade on its own.

**D. Narrow the recipe so a provably non-audible edit stops demoting.**
This is the only option that fixes the CLASS rather than this instance -- every
future refactor of this adapter hits the same wall. There is precedent: the
recipe was already narrowed once, on 2026-08-19, and the narrowing was decided by
MEASUREMENT (`_otr_voice_node_common.py` had produced 18 false demotions against
1 true one in 60 days) rather than by preference.

But it is genuinely harder than it looks, and the counter-argument is strong:
* whole-file hashing is chosen deliberately, because "a fingerprint that
  under-reports is a false claim of proof; one that over-reports is an
  inconvenience";
* any smarter rule (hash the AST, ignore imports and comments, hash only named
  functions) is a NEW mechanism that can itself be wrong, and it would be wrong
  in the direction that silently keeps a stale qualification alive;
* the operator has already met this once -- a COMMENT demoted the route, and the
  answer was to revert the comment, not to soften the hash.

**This has more than one defensible answer, so under the standing rule it needs
an arc before any code.** It is not scheduled and nothing waits on it.

## The question for the operator

Only one, and it is not urgent:

> Is option **D** worth an arc -- i.e. should a provably non-audible edit to a
> fingerprinted adapter stop costing a re-audition -- or does whole-file hashing
> stay exactly as it is, with **A** as the standing answer every time this comes up?

Until that is answered, **A stands** and the adapter stays blocked.

## Grounding

* `nodes/_otr_voice_route.py` -- `RUNTIME_FINGERPRINT_SOURCES`,
  `live_engine_impl_version`, `stale_runtime_fingerprint`, and the header
  comment block that states the doctrine.
* `config/cast_pools.py:847` -- the shipped Lemmy route's stored value;
  `:882` the historical 4-file value; `:993` the superseded route.
* `docs/PROD_BUG_LOG.md` -- PBUG-20260817-09, the defect the mechanism closed.
* `kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md` -- the closed
  migration plan, which did not know about this constraint.
