# Overnight status — 2026-05-18 Sprint H §3.7 retest #8

**Status:** HALT — new defect class surfaced.
**No fix applied.** Awaiting Jeffrey sign-off on outline-phase scope.

---

## What ran

Per Jeffrey 2026-05-17 overnight directive:

1. **Reconcile commit** landed clean:
   - Commit `0ce8d2b` on `v2.0-alpha`, pushed to origin.
   - `scripts/worker_iter.py` WORKFLOW_PATH flipped from
     `otr_scifi_16gb_bughunt.json` to `otr_scifi_16gb_full.json`.
   - `workflows/otr_scifi_16gb_bughunt.json` deleted
     (sibling that diverged from canonical).
   - `workflows/otr_scifi_16gb_full.json.bak-normalize-1779012582`
     deleted (stale normalize backup).
   - `tests/test_otr_api_companions.py` renamed `_BUGHUNT_WORKFLOW`
     → `_CANONICAL_WORKFLOW` + helper / test renames.
   - Regression sweep clean: 17 + 38 + 23 = 78 passed,
     2 skipped, 2 xfailed (baseline).
   - Local HEAD == origin HEAD verified.

2. **§3.7 retest #8 launched** via `sweep_and_launch.bat
   --iters 2 --inter-iter-sec 10` at 2026-05-18T01:31:11.
   - Pre-launch sweep: killed 2 Windows-MCP python helpers,
     no ComfyUI residue.
   - Supervisor real_pid=54624, parent_pid=40440, keep_pids=[40440, 54624].

## What halted us

Both iters classified as `failure_class=unknown` (the classifier
does not yet route `OutlineFailedError` to a named class).
Supervisor stop rule fired at iter-2 end:
`STOP_DECISION: halt: 2 consecutive unknown failures`.

### Iter 1 (worker_iter_001.json)

```
status:        error
failure_class: unknown
exception:     OutlineFailedError
sub-error:     Outline generation failed after 3 attempts.
               ValidationError: 2 validation errors for Outline
               - beats.5.target_words: Input should be >= 3,
                 input_value=0
               - beats.5.mood: Field required (missing)
executed_count: 4
peak_vram_gb:   9.33
wall_time_s:    167.18
prompt_id:      60f74237-881a-4f80-a012-796e0793e32b
```

### Iter 2 (worker_iter_002.json)

```
status:        error
failure_class: unknown
exception:     OutlineFailedError
sub-error:     OutlineBudgetViolation: Beat b003 has
               arc_phase='setup'; not in budget
               arc_phases=['scene']
executed_count: 4
peak_vram_gb:   9.38
wall_time_s:    186.99
prompt_id:      62c60dfa-ca6a-4ce9-a5de-62b25c8bc7a6
```

## Why this is a new defect class

Different family from retest #7's blocker:

| Retest | Failure                          | Phase reached            | VRAM peak | Cause                                          |
|--------|----------------------------------|--------------------------|-----------|------------------------------------------------|
| #7     | StyleGenerationFailedError (OOM) | style invention          | 15.91 GB  | FLUX/Gemma co-residence on bughunt clone       |
| #8 i1  | OutlineFailedError               | outline (post-style)     | 9.33 GB   | Mistral-Nemo outline schema validation         |
| #8 i2  | OutlineFailedError               | outline (post-style)     | 9.38 GB   | Mistral-Nemo arc_phase budget mismatch         |

- Retest #7 was a **topology / VRAM** problem (FLUX resident when
  Gemma needed VRAM). Jeffrey pre-authorized auto-fix via gate
  recipe. FluxBranchGate (commit `4220b22`) was the proposed
  resolution.
- Retest #8 is a **writer-prompt-output validation** problem on
  Mistral-Nemo. Peak VRAM 9.3 GB across both iters — well under
  the 14.5 GB ceiling. The FluxBranchGate **never fired** because
  the writer failed in its outline phase, before the freeze
  cascade ever emitted `script_json`.

This is outside the pre-authorized "same-pattern co-residence OOM"
auto-fix scope. Per directive: **halt and report. No fix applied.**

## FluxBranchGate verdict: unknown

The gate's telemetry was never observed. The writer never reached
the freeze cascade, so the gate node was never executed. Whether
the gate would have deferred FLUX consumers correctly under the
intended load pattern is **still an open question** that retest
#8 did NOT answer.

To answer it, the writer's outline phase must complete cleanly
(or the workflow needs a temporary bypass to short-circuit
outline → cast → directly drive the freeze cascade with a stub
script, just to exercise the gate). Both options are outside
the overnight pre-authorized scope.

## What we did NOT do (per directive)

- **Did NOT apply any outline-phase fix.** Outline prompt /
  schema / budget tuning is Jeffrey's scope.
- Did NOT bump any version label.
- Did NOT swap any model file.
- Did NOT modify pagefile settings.
- Did NOT modify the harness beyond the reconcile commit
  (`0ce8d2b`) and the prior pre-overnight commits
  (`0facea7` crash_process, `4c1ed2d` chain port).
- Did NOT add a new failure_class to the worker classifier.
  `OutlineFailedError` and `OutlineBudgetViolation` would route
  to e.g. `writer_outline_validation` if added, but that's a
  classifier change Jeffrey signs off on.
- Did NOT advance to §3.8 or §3.9 — §3.7 retest #8 did not GREEN.

## Pre-existing classifier mis-routes (deferred, unchanged)

- `StyleGenerationFailedError` routes to `unknown` (should be
  `llm_oom` when the underlying cause is a CUDA allocation
  failure inside `load_llm`).
- `OutlineFailedError` routes to `unknown` (no existing class
  for writer-phase schema validation failures).

Both still deferred per the §3.7 retest #7 follow-up list.

## Supervisor iter-1-END anomaly (still deferred)

The prior retest #7 left an `ITER 1: worker PID=66756` line with
no matching END line in `overnight_supervisor.log`. Retest #8
ITER 1 + ITER 2 both wrote clean END lines, so the anomaly is
not currently reproducing on the post-reconcile harness. Keep
deferred — it may have been a Ctrl-C left-over from an earlier
attended-cancel.

## Files touched this session

- `scripts/worker_iter.py`            (M)
- `tests/test_otr_api_companions.py`  (M)
- `workflows/otr_scifi_16gb_bughunt.json` (D)
- `workflows/otr_scifi_16gb_full.json.bak-normalize-1779012582` (D)
- `docs/2026-05-18-overnight-status.md` (N — this file)

Commit: `0ce8d2b` on `v2.0-alpha`, pushed.

## Next operator decisions (none auto-taken)

1. **Outline validation defect investigation.** Two distinct
   sub-failures in two iters suggests Mistral-Nemo + outline
   schema have a tight-budget compliance gap at `target_words=30`.
   Options:
   - Loosen the smoke profile's outline budget (e.g.
     `target_words=120` instead of 30).
   - Tighten the outline LLM prompt to enforce `target_words >= 3`
     per beat and `arc_phase` consistency.
   - Add a normalizer that clamps `target_words` to `[3, ...]`
     and drops out-of-budget beats before validation.
   - Switch the writer's `creative_writing_model` to one that
     historically honored tight budgets better (per
     memory: Gemma-4-E4B-it worked under retest #7's bughunt
     clone, although that path also OOM'd).
2. **Classifier upgrade.** Add `writer_outline_validation` (or
   similar) and route `OutlineFailedError` /
   `OutlineBudgetViolation` to it so the supervisor's same-class
   halt rule sees the actual defect.
3. **FluxBranchGate exercise.** Either fix the outline defect
   first (preferred — gets the workflow back to end-to-end), or
   craft a minimal subgraph that drives the freeze cascade
   directly to exercise the gate in isolation.

Halt closed. Awaiting direction.
