# Overnight status #3 — 2026-05-18 Sprint H §3.7 retest #10

**Status:** HALT — outline-validator (different module than
EpisodeBudget) per Jeffrey's halt rule.
**No fix applied.** Recommended next smoke-config knob included
below for paste-back approval.

---

## TL;DR

EpisodeBudget validator now passes cleanly (act_count=3 fix
held). Writer phase enters outline generation. Mistral-Nemo
produces outlines containing **music-interlude beats**
(`beat_id='m001'`, `'m002'`) which violate the outline schema's
`beat_id` pattern `^b\d{3}$`. Schema only accepts voiced/announcer
beats with b-prefix IDs. Three repair attempts fail; outline
phase classifies as `writer_outline` (the new class).

Single next smoke-config knob: set `include_act_breaks=False` in
the worker patch so `music_inter_count` becomes 0 and the outline
LLM stops emitting music beats entirely. That is the FOURTH and
last writer-side widget the worker can touch (target_words,
num_characters, act_count are already aligned).

Classifier victory: the supervisor's halt line read
`STOP_DECISION: halt: 2 consecutive writer_outline failures`
instead of `unknown`. Same-class halt rule now actually reports
the defect family. Worth keeping even after the next retest.

---

## What ran

Commit `0ebef36` on `v2.0-alpha` pushed clean. Pre-flight:

- AST parse on `worker_iter.py` + `overnight_bug_hunt.py`: clean.
- Regression sweep: 78 passed, 2 skipped, 2 xfailed (baseline).
- Local HEAD == origin HEAD verified.

§3.7 retest #10 launched at 2026-05-18T09:48:26 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`. Smoke
overrides:

| Widget          | Smoke value | Validator at tw=300 | OK? |
|-----------------|-------------|---------------------|-----|
| target_words    | 300         | >= 30               | ✓   |
| num_characters  | 2           | >= 1                | ✓   |
| act_count       | 3           | default(300)=3      | ✓   |
| (untouched)     | --          | <= max(300)=6       | ✓   |

EpisodeBudget validator passed both iters. Writer entered
outline phase.

## What halted us

Supervisor stop rule fired at 09:56:31:
`STOP_DECISION: halt: 2 consecutive writer_outline failures`.

### Iter 1 (worker_iter_001.json)

```
status:        error
failure_class: writer_outline    <- correctly routed
exception:     OutlineFailedError
sub-error:     OutlineBudgetViolation: Phase 'complication' has
               4 voiced beats (target 6, allowed 5-8). Add or
               remove voiced beats in that phase.
executed_count: 4
peak_vram_gb:   9.45
wall_time_s:    232.04
prompt_id:      397aaad6-105a-4464-b6bf-cea72b71dc58
```

Mistral-Nemo gave the complication act 4 voiced beats; budget
demands [5, 8]. Same beat-count compliance issue as retest #8.

### Iter 2 (worker_iter_002.json)

```
status:        error
failure_class: writer_outline    <- correctly routed
exception:     OutlineFailedError
sub-error:     ValidationError: 8 validation errors for Outline
               - beats[4].beat_id 'm001' does not match ^b\d{3}$
               - beats[4].intent missing
               - beats[4].target_words missing
               - beats[4].mood missing
               - beats[10].beat_id 'm002' does not match ^b\d{3}$
               - beats[10].intent missing
               - beats[10].target_words missing  (truncated)
executed_count: 4
peak_vram_gb:   9.61
wall_time_s:    237.03
prompt_id:      3adf992d-bc3b-4ebb-85f0-bff7f19ed37c
```

Mistral-Nemo placed two music-interlude beats (`beat_id='m001'`,
`'m002'`) inline in the beats list, with only `beat_id` and
`speaker` fields populated. The outline schema is strict on:
- `beat_id` regex `^b\d{3}$` (voiced beat IDs only)
- `intent`, `target_words`, `mood` all required

So music beats can never round-trip through this schema as-is.
With `include_act_breaks=True` the budget records
`music_inter_count=2`, which the outline prompt apparently asks
the LLM to author inline (rather than emit them as a separate
field).

## FluxBranchGate verdict: still unknown

Third retest in a row where the gate node never executed. Writer
phase did not reach the freeze cascade. Peak VRAM both iters:
9.4-9.6 GB -- nowhere near the 14.5 GB ceiling.

## Recommended next smoke-config knob

`scripts/worker_iter.py` after the existing act_count patch:

```python
patch_widget_by_name(wf, 1, "include_act_breaks", False, schemas)
```

Effect on the EpisodeBudget:
- `music_inter_count = (act_count - 1) if include_act_breaks else 0`
- include_act_breaks=False  ->  music_inter_count = 0
- Outline schema no longer asks for music-beat slots.

This is the FOURTH (and last) widget the worker can touch
without bumping into widgets that have linked inputs (model_id,
script_json, etc.). If iter 1 after this knob still fails the
outline phase, the issue is genuinely in the writer prompt /
Mistral-Nemo compliance and not in any remaining smoke-config
knob.

## Cumulative pattern (three halts on writer phase)

| Retest | Halt cause              | Smoke fix                  |
|--------|-------------------------|----------------------------|
| #8     | OutlineFailedError      | target_words 30 -> 300     |
| #9     | InvalidEpisodeBudgetError | act_count 1 -> 3         |
| #10    | OutlineFailedError      | (proposed) include_act_breaks False |

If retest #11 after the include_act_breaks knob still fails,
the writer prompt is the bottleneck. At that point Jeffrey has
two paths:

1. **Writer prompt scope** — Tune the outline LLM prompt to
   strictly emit only b-prefix beat_ids + per-beat target_words
   >= 3, OR add a structured-output decoder that drops malformed
   beats before validation. This is the writer-fix path the
   smoke-config principle was deferring.
2. **Bypass the writer for gate exercise** — Inject a synthetic
   pre-baked ledger downstream of the writer (or directly into
   the freeze cascade input) so the FluxBranchGate fires without
   needing Mistral-Nemo to produce a valid outline. Surface the
   gate-fire telemetry decoupled from writer-side stability.

The Cowork harness can stage either, but both require Jeffrey's
sign-off because they go beyond smoke-config knobs.

## Classifier upgrade landed cleanly

This was the first retest where the supervisor's STOP_DECISION
line names the actual defect class (`writer_outline`) rather than
defaulting to `unknown`. Worth keeping in mind: future retest
halts will now be self-documenting in the supervisor log.

`writer_budget` was tested implicitly -- it would have fired if
the act_count fix hadn't held, but EpisodeBudget passed cleanly.
Module-level marker (`_otr_episode_budget`) was the defensive
catch for siblings; not exercised this iter (no siblings exist
yet).

## What we did NOT do (per directive)

- Did NOT apply the `include_act_breaks=False` smoke knob.
- Did NOT touch the writer prompt or outline schema.
- Did NOT bypass the writer with a synthetic ledger.
- Did NOT touch any workflow, node, or model file.
- Did NOT bump a version label.
- Did NOT advance to §3.8 or §3.9.

## Files this session

- `scripts/worker_iter.py` (M; smoke act_count + classifier
  writer_budget tuple + writer_outline module marker; in commit
  `0ebef36`)
- `scripts/overnight_bug_hunt.py` (M; FAILURE_CLASSES set; in
  `0ebef36`)
- `docs/2026-05-18-overnight-status-3.md` (N -- this file)

## Halt closed

Awaiting paste-back. Same posture as status #1/#2: pre-authorized
fixes overnight remain same-pattern co-residence OOM only;
halt-and-report conditions unchanged; hard stops unchanged.
