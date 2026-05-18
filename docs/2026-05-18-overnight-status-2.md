# Overnight status #2 — 2026-05-18 Sprint H §3.7 retest #9

**Status:** HALT — new defect class surfaced.
**No fix applied.** Awaiting Jeffrey sign-off on the recommended
one-line smoke-config tweak.

---

## TL;DR

Smoke target_words went from 30 → 300 per Jeffrey's go-forward
plan. Outline failures vanished as predicted -- but a sibling
smoke-config inconsistency surfaced before the writer phase ever
ran. The episode-budget validator caught it and self-documented
the fix in its own error message:

```
act_count=1 below default 3 for target_words=300.
Override upward only -- pick 3 or higher.
```

Recommended one-line fix in the very next commit:
`scripts/worker_iter.py:548` -- raise smoke `act_count` from
`1` to `3` (or higher). Pasting "ship it" gets the next iter
back to writer phase, which is where the gate-fire telemetry
actually lives.

---

## What ran

Commit `252ea1f` on `v2.0-alpha` pushed clean. Pre-flight
verification:

- AST parse on `worker_iter.py` + `overnight_bug_hunt.py`: clean.
- Regression sweep: 78 passed, 2 skipped, 2 xfailed (baseline).
- Local HEAD == origin HEAD.

§3.7 retest #9 launched at 2026-05-18T09:34:22 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`. Both iters
ran with the new smoke profile (target_words=300, num_characters=2,
act_count=1) against canonical `_full.json`.

## What halted us

Supervisor stop rule fired at 09:36:54:
`STOP_DECISION: halt: 2 consecutive unknown failures`.

### Iter 1 (worker_iter_001.json)

```
status:        error
failure_class: unknown     <- classifier doesn't yet route
                              InvalidEpisodeBudgetError
exception:     InvalidEpisodeBudgetError
message:       act_count=1 below default 3 for target_words=300.
               Override upward only -- pick 3 or higher.
executed_count: 4
peak_vram_gb:   9.11
wall_time_s:    66.45
prompt_id:      ba28bf41-15bb-4f74-b2f4-52cf6ec1d78b
```

### Iter 2 (worker_iter_002.json)

Identical defect, identical message. wall=70.74s, VRAM=9.13.
prompt_id: `7b8a91f5-1724-4a61-a2ea-596a92fa76ab`.

## Why this is a new defect class

Different validator from retest #8:

| Retest | Phase reached            | Validator that fired                            | Cause                                                  |
|--------|--------------------------|-------------------------------------------------|--------------------------------------------------------|
| #8 i1  | outline (post-style)     | `OutlineFailedError` (pydantic on beats schema) | Per-beat target_words=0 because total budget too small |
| #8 i2  | outline (post-style)     | `OutlineFailedError` (OutlineBudgetViolation)   | Mistral-Nemo arc_phase mismatch                        |
| #9 i1  | episode-budget preflight | `InvalidEpisodeBudgetError`                     | act_count=1 inconsistent with target_words=300         |
| #9 i2  | episode-budget preflight | `InvalidEpisodeBudgetError`                     | Same                                                   |

Same family ("smoke-config artifact, fix smoke config") per
Jeffrey's 2026-05-18 framing. But a new exception type and a
new phase (preflight, not outline). Per directive's letter:
"halt on new defect class." Done.

## FluxBranchGate verdict: still unknown

The gate node never executed. The episode-budget validator
fires BEFORE any LLM call, before the writer phase, before the
freeze cascade. Peak VRAM both iters: 9.1 GB -- well under the
14.5 GB ceiling. The eager-FLUX-GPU-load question retest #8
was supposed to answer remains open after retest #9 too.

## Recommended one-line fix

`scripts/worker_iter.py` line 548:

```python
patch_widget_by_name(wf, 1, "act_count", 1, schemas)
```

becomes:

```python
patch_widget_by_name(wf, 1, "act_count", 3, schemas)
```

Rationale:
- The episode-budget validator explicitly documents this in its
  error message: "Override upward only -- pick 3 or higher."
- target_words=300 / act_count=3 = 100 words per act. Outline
  budget allocator can give each act 2-4 beats with target_words
  >= 20 each. Well inside Mistral-Nemo's reliable zone.
- Wall-time impact: outline writes 3 acts instead of 1. Iter 1
  of retest #8 ran 167s on 1 act / outline phase. Linear scale
  to 3 acts is ~500s, still inside the 900s exec timeout. Add
  cast + dialogue + ledger freeze = projected ~700s total. Both
  iters fit inside the 1200s worker outer wait.
- Same smoke-config principle Jeffrey just stated for the
  target_words=300 tweak.

## Optional classifier follow-up (separate commit, not required)

`InvalidEpisodeBudgetError` currently routes to `unknown`.
Adding it to the writer_outline class -- OR a new
`writer_budget` class -- would surface the real defect to the
supervisor's same-class halt rule. Three-line patch in
`_classify_failure`. Not required to unblock retest #10; flagged
for batching with other classifier upgrades.

## What we did NOT do (per directive)

- Did NOT apply the recommended `act_count=3` fix.
- Did NOT add `InvalidEpisodeBudgetError` to the classifier.
- Did NOT touch any workflow, node, or model file.
- Did NOT modify the harness beyond commits `0ce8d2b` +
  `252ea1f` (the reconcile + smoke-budget + outline-class
  commits Jeffrey already approved).
- Did NOT bump a version label.

## Files this session

- `scripts/worker_iter.py` (M; smoke + classifier; in commit `252ea1f`)
- `scripts/overnight_bug_hunt.py` (M; FAILURE_CLASSES set; in `252ea1f`)
- `docs/2026-05-18-overnight-status-2.md` (N -- this file)

## Halt closed

Awaiting "ship it" on the `act_count` raise. Same posture as
status #1: pre-authorized fixes overnight remain same-pattern
co-residence OOM only; halt-and-report conditions unchanged;
hard stops unchanged.
