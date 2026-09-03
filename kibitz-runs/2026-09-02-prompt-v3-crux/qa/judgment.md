# Finished-diff QA -- Prompt v3 Half A

Reviewer seat: **Sonnet** (Cowork subagent), scoped to eleven named functions
rather than the whole diff. Driver: Claude (Cowork, 5080).
Returned 13 numbered findings: 7 defects, 5 contracts confirmed clean, 1 bound
documented.

**Five defects fixed, two answered by making a docstring honest. Nothing was
waved through.**

---

## Fixed

**F2 -- the vantage was removed with an unguarded `str.replace`.** No `count`,
so every occurrence of that exact span would be deleted, not just the trailing
one the composer appended. Not reachable with today's constants (checked: no
collision between the three vantage strings and the 96 motion clauses), but the
mechanism offered no protection. **Now trimmed as a SUFFIX**, which can only
touch the one the composer just added.

**F5 -- `IS_CHANGED` truncated each input to 4096 characters.** The ledger JSON
is far longer than that and its first kilobytes are metadata, so two different
episodes could share a prefix and collide -- in the one method whose entire
purpose is preventing a stale-cache collision. **Now hashes the full value with
sha256.** (Repairing this also caught a shell-escaping accident that had written
two real control bytes into the source; the file was rewritten from a script and
verified null-free, BOM-free and AST-clean.)

**F6 -- the drop receipt named units that were never there.** The fitter walked
`GHOST_V3_DROP_ORDER` blind, so an over-budget `signal` beat published
`prompt_dropped: ["light"]` even though `resolve_world_light` returns "" on that
mode and there was no light clause to drop. **Now a unit is receipted only if
removing it actually changed the text.** That makes `dropped` a SUBSEQUENCE of
the order rather than a prefix, and the test and the fuzz assert the new
invariant.

**F7 -- `--ab` printed per-shot PASSes over a truncated `zip`.** When the two
arms had different trace lengths the overall verdict was correctly FAIL, but the
three per-shot lines a reader skims were each computed over the shorter prefix
and could all print PASS. **Now the per-shot checks are skipped with a printed
reason when the lengths disagree.**

**F8 -- `--ab` with one replay silently did nothing.** A proof tool may not print
a clean run for a comparison it never made. **Now it fails with a named check.**

## Answered rather than patched

**F1 and F3 -- "`finalize_ghost_prompt_v3` is not TOTAL".** Correct as stated,
and the right response is a precise docstring rather than a `try`. It can still
raise for two reasons: a row whose `role` is not a real role or whose `mode` has
no vantage (`compose_ghost_prompt_v3` refuses, exactly as the v2 composer
refuses), and an unavailable SD1 tokenizer in production (shared with v2). Those
are **malformed input and missing infrastructure, not length**, and the law on
this lane is already that a malformed object fails closed rather than
downgrading into something that looks like a healthy render. The contract v3
owes is *never raise FOR BUDGET*, and the docstring now says that in those
words instead of overclaiming.

## Documented, not changed

**F4 -- the motion pool's non-repeat guarantee is bounded at 32.** True: each
bucket holds exactly 32 clauses and `(start + ordinal) % 32` wraps at ordinal
32. The docstring already scoped the claim ("for any episode no longer than the
bucket"), the longest planned episode observed on this lane is 29 shots, and the
mode cycle spreads beats across three buckets so no single bucket approaches 32
in practice. The test pins the bucket size and the uniqueness across a full
bucket.

## Confirmed clean by the reviewer

Contract C (nothing writes to `story_brief_terms`, so `brief_hash` and every
seed hold), contract D (banana applied once, re-measured after), contract F (the
drop order stops as soon as it fits), contract A (the kernel ladder is total on
every hostile shape), and the odometer arithmetic in both resolvers.

## Independent evidence run beside the review

* **Fuzz:** 5,314 combinations -- 13 hostile meta shapes (empty, `None`,
  whitespace-only, wrong types, 300-character entries, 16 objects) x 9 packs x 7
  role/mode pairs x 6 ordinals, plus 400 randomised injected budgets. No
  contract violated: nothing raised, no empty prompt, no lost component, no
  dangling separator, no over-window prompt, no sliced subject.
* **Driver integration:** the real published ledger plus three broken variants
  (failed brief, fields absent, fields null) through
  `build_request_from_shot`, the same entry point ShotLock uses as its
  cast-time preflight. Clean on all four.
* **Targeted tests:** 321 passed across the five Ghost files and the render
  batch, 42 passed across the three replay files.
