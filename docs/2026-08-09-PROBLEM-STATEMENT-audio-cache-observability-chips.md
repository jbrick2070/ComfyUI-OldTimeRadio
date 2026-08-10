# PROBLEM STATEMENT -- two audio-cache observability chips

**Date:** 2026-08-09. **Operator selected:** run these (list item 4), with a
kibitz panel. Both are follow-up chips owed from the SF#1 tombstone.

**Scope:** `nodes/_otr_voice_node_common.py`, `nodes/_otr_audio_cache.py`
(read-only if possible), `tests/`. **No video. No render path. No workflow JSON.**

---

## CHIP A -- a dying line reports `cache=off` when the cache was ON

**Grounded, confirmed in source.** `nodes/_otr_voice_node_common.py`:

* `:829` `cache_status = "off"` -- the initial value.
* Every reassignment happens AFTER the generate call: `"hit"` `:836`,
  `"degraded_write"` `:891`, `"miss"` `:911`, `"degraded_write"` `:925`.
* `:937` emits the per-line P-OBS tail `f" cache={cache_status}"`.

So when `generate_voice` raises **before** any status assignment, a
**cache-ENABLED** line emits `cache=off`. That is the exact line a person reads
while diagnosing the failure, and it tells them the cache was disabled when it
was not. It ships from the day chunk 2 landed.

**Why it matters beyond cosmetics:** this is the BUG-12.86 family -- *a field
that reads as evidence and is not*. It is worse than a missing token, because
`off` is a confident, plausible answer.

**The fix is not simply "initialize to unknown".** `off` is the CORRECT value on
the genuine cache-off path, and that path must keep emitting `off` byte-
identically (the SF#1 tombstone pins a cache-off byte-identity test). So the
initial value has to distinguish "cache disabled" from "cache enabled, outcome
not yet determined" without changing the disabled-path output.

**Open questions for the panel:**
1. Is a third token (e.g. `pending` / `unknown`) the right shape, or should the
   initial value be derived from `cache_enabled` so the disabled path is `off`
   from the start and the enabled path starts at something else?
2. Does any downstream consumer PARSE this token? If a log scraper or the
   post-run audit reads `cache=`, a new token value is a contract change.
   **This must be answered before code.**
3. Is the P-OBS tail emitted on the exception path at all, or only on success?
   If a raise skips `:937` entirely, the defect is narrower than stated and the
   fix changes shape.

## CHIP B -- seven corruption branches, zero tests

The chip as written said "`_write_audio_atomic` logs a bounded warning on
partial-crash recovery; there is no test asserting the warning text."

**Grounding shows it is WIDER than that.** `nodes/_otr_audio_cache.py` carries at
least seven distinct `log.warning` branches, each classifying a DIFFERENT
corruption mode on load:

| line | condition |
|---|---|
| `:333` | `cache_key` mismatch |
| `:343` | audio file missing |
| `:348` | `sample_rate` mismatch |
| `:351` | `channels` mismatch |
| `:355` | npy load produced a non-array |
| `:361` | `sha256` mismatch |

`grep -rn "degraded_write\|_write_audio_atomic" tests/` returns **nothing**.

The module's own docstrings promise the behaviour -- `:175` *"Corruption is a
silent miss with ONE bounded warning log line"*, `:325` *"bounded `log.warning`
per miss-due-to-corruption naming the cache key"* -- so there is a documented
contract with no executable proof. Any of these branches could stop firing, or
start firing twice, or lose the cache key from its message, and nothing notices.

**What the panel should decide:**
1. Is the deliverable ONE caplog test per branch (six or seven tests), or one
   parameterized test over a corruption matrix? Parameterized risks a single
   fixture that cannot actually produce all six states.
2. **"ONE bounded warning" is the load-bearing half of the promise.** A test that
   only asserts "a warning appeared" would pass if the code logged six. Assert
   the COUNT, not just presence.
3. Does each branch degrade to a MISS (regenerate) rather than an error? The
   docstring says silent miss; that behaviour deserves the assertion more than
   the string does. Prefer pinning behaviour over log text where they compete --
   log text is the weaker contract and churns.
4. Is `_write_audio_atomic` itself even reachable in a partial-crash state from a
   test, or does proving it require simulating an interrupted write? If the
   latter, say so rather than shipping a test that never enters the branch.

## Constraints

* **A test that cannot fail is worse than no test.** Both chips are about
  observability, so every test added here must be shown to go RED when the
  behaviour it guards is removed. Mutation-check each one.
* Do not weaken the cache-off byte-identity guarantee from SF#1.
* No new dependency, no network, no GPU.

## Baseline

Suite `9520 passed / 111 skipped / 3 deselected / 1 xfailed, exit 0` at
`17db273a`. The 3 deselected belong to a concurrent window's uncommitted
`eng_wan_i2v.py` and are not ours. Gate: full suite by EXIT CODE, then the Bug
Bible, then push.
