# Overnight status #4 — 2026-05-18 Sprint H §3.7 retest #11

**Status:** HALT — `writer_outline` after Gemma-4 +
`include_act_breaks=False`. Per Jeffrey's branching rule, writer
prompt tuning vs synthetic-ledger pivot is Jeffrey's call.

**No fix applied.** Outline-compliance failure family is now
confirmed **model-agnostic** (hit on both Mistral-Nemo and
Gemma-4-E4B-it). Smoke-config knobs are exhausted.

---

## TL;DR

Gemma-4-E4B-it loaded cleanly. Writer phase reached outline
generation with EpisodeBudget passing
(`act_count=3, music_inter=0` per the include_act_breaks=False
overlay). Outline phase failed both iters; classifier routed
both as `writer_outline`.

Gemma-4's failure flavor is **under-then-over** (LLM produces a
60-65-word resolution then over-corrects to 290-320 words in
setup on the repair attempt). Mistral-Nemo's failure flavor was
**over-then-still-over** (4 voiced beats vs 5-8 required, plus
m-prefix music IDs). Different sub-patterns; same family:
**writer-prompt-output compliance with the outline schema.**

FluxBranchGate verdict: still unknown across 4 consecutive
retests. Peak VRAM 10.05 GB across both iters — well under the
14.5 GB ceiling, no OOM, gate never executed.

---

## What ran

Commit `6add3fc` on `v2.0-alpha` pushed clean.
- Workflow `_full.json` node 1 writer widgets flipped to
  `google/gemma-4-E4B-it`.
- C7 canonical baseline drift guard re-baselined
  (`test_writer_both_slots_gemma_4`).
- Smoke overlay added `include_act_breaks=False`.

Pre-flight regression: 78 passed, 2 skipped, 2 xfailed.

§3.7 retest #11 launched at 2026-05-18T10:07:00 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`. Smoke
overrides at submit time:

| Widget               | Smoke value           | Validator OK? |
|----------------------|-----------------------|---------------|
| target_words         | 300                   | ✓ (>= 30)     |
| num_characters       | 2                     | ✓             |
| act_count            | 3                     | ✓ (= default) |
| include_act_breaks   | False                 | (new)         |

Phase 2A budget came out:
`act_count=3, arc_phases=['setup', 'complication', 'resolution'],
per_phase_words=[84, 132, 84], per_phase_beats=[4, 6, 4],
words_per_beat_range=[20, 35], music_inter=0`

`music_inter=0` confirms the include_act_breaks knob took effect.

## What halted us

Supervisor stop rule fired at 10:18:33:
`STOP_DECISION: halt: 2 consecutive writer_outline failures`.

### Iter 1 (worker_iter_001.json)

```
status:        error
failure_class: writer_outline
exception:     OutlineFailedError
attempts:      3 (1 fresh, 2 fresh, 3 repair at temp=0.30)
attempt 1:     resolution 60 words (allowed 67-101) -- short
attempt 2:     resolution 65 words (allowed 67-101) -- short
attempt 3:     setup 290 words (allowed 67-101) -- over (repair flip)
peak_vram_gb:  10.05
wall_time_s:   341.4
prompt_id:     ad94e6ad-e515-46e8-a8b8-986a5d061cd3
```

### Iter 2 (worker_iter_002.json)

```
status:        error
failure_class: writer_outline
exception:     OutlineFailedError
attempts:      3
attempt 1:     resolution 35 words (allowed 67-101) -- short
attempt 2:     setup 320 words (allowed 67-101) -- over (flip)
attempt 3:     (repair) same family
peak_vram_gb:  10.05
wall_time_s:   336.4
prompt_id:     c1fd80e7-a36d-4bf9-aaf2-0924e0a3aebe
```

## Outline failure family — now confirmed model-agnostic

| Retest | Writer model    | Failure mode in outline                                  |
|--------|-----------------|----------------------------------------------------------|
| #8     | Mistral-Nemo    | per-beat target_words=0; arc_phase mismatch              |
| #10    | Mistral-Nemo    | complication beat count 4 vs 5-8; m001/m002 IDs invalid  |
| #11 i1 | Gemma-4-E4B-it  | resolution 60->65 words then setup 290 (over-correct)    |
| #11 i2 | Gemma-4-E4B-it  | resolution 35 words then setup 320 (over-correct)        |

The defect is in how the writer LLM (either model) honors the
per-phase word budget. The repair attempt at temp=0.30
over-corrects in the opposite direction. Smoke knobs cannot
push the LLM to satisfy `[67-101]` consistently.

## FluxBranchGate verdict: still unknown

Fourth retest where the gate node never executed. Writer phase
did not reach the freeze cascade. Peak VRAM iter 1 / 2:
**10.05 GB / 10.05 GB** -- noticeably higher than Mistral-Nemo
runs (9.4-9.6 GB) but still well under the 14.5 GB ceiling.

The smaller Gemma-4 disk footprint (~6 GB vs Mistral-Nemo's
~24 GB) does NOT translate to lower VRAM peak in this writer
configuration. The dynamic offloader keeps both models at
roughly the same operational footprint. This is informative for
the FLUX-gate timing question even though the gate didn't fire:
**writer VRAM peak shrinks marginally with Gemma-4**, not by an
order of magnitude.

## Smoke-config knobs are exhausted

Four writer-side widgets the worker can touch:

| Widget               | Default in JSON | Smoke override | Why |
|----------------------|-----------------|----------------|-----|
| target_words         | 350             | 300            | retest #8 unblock |
| num_characters       | 2               | 2 (no change)  | already valid     |
| act_count            | 3               | 3              | retest #9 unblock |
| include_act_breaks   | true            | False          | retest #10 unblock |

All four are now aligned to the EpisodeBudget + outline-schema
constraint set. No remaining smoke knobs to flip. The next
unblock requires touching either the writer prompt, the outline
schema, or bypassing the writer entirely with a synthetic
ledger -- all of which Jeffrey explicitly reserved for sign-off.

## Two follow-up paths (Jeffrey's call)

Per status-3's enumeration, refined with retest #11 evidence:

### Path A — Writer prompt tuning

Make Mistral / Gemma honor `[67-101]` per-phase word ranges.
Concrete moves:
- Stronger constraint language in the outline LLM prompt (an
  explicit table per-phase + 1-shot exemplar showing the right
  shape).
- Per-beat structured output via GBNF grammar that constrains
  `target_words` to a small integer range, with the upstream
  Python normalizer redistributing words across beats to hit
  the per-phase budget exactly.
- Reduce the repair-attempt over-correction with bounded
  word-delta hints in attempt 3's prompt.

Scope: 2-4 commits on `nodes/_otr_outline.py` and its prompt
template. Audit gate: re-run §3.7 retest after each commit; the
classifier will correctly route or pass.

### Path B — Synthetic-ledger gate exercise

Inject a pre-baked minimal ledger (single character, single act,
single line) directly upstream of the `OTR_LedgerFreezeCascade`
input that the FluxBranchGate's `gate_signal` consumes. Skips
the writer phase entirely. Validates the gate behavior in
isolation:
- gate fires at low VRAM -> ComfyUI executor deferred FLUX's
  GPU materialization (the design intent).
- gate fires at high VRAM -> CheckpointLoaderSimple pre-loaded
  FLUX regardless of consumer readiness. Follow-up commit
  replaces CheckpointLoaderSimple with an OTR wrapper that
  defers `model_management.load_models_gpu()` until
  `gate_signal` is ready.

Scope: 1 new ComfyUI node (OTR_LedgerStub) + 1 workflow rewire
(inject between cascade input and the gate). Doesn't touch the
writer. Gives Jeffrey the gate-fire telemetry without
unblocking the writer phase first.

### Recommendation

I'd suggest **Path B first** for the data, then **Path A** for
the production unblock. Path B is the smaller commit, decouples
two questions that are currently entangled, and gives the
post-Sprint-H roadmap a clean answer to the eager-FLUX-load
question. Path A is larger and the writer-prompt tuning may
require its own round-robin with the externals.

Either is Jeffrey's call -- no autonomous start.

## What we did NOT do

- Did NOT touch the writer prompt or outline schema.
- Did NOT build the synthetic-ledger node.
- Did NOT touch any node Python file.
- Did NOT modify the EpisodeBudget validator.
- Did NOT bump a version label.

## Files this session

- `workflows/otr_scifi_16gb_full.json` (M; writer widgets → gemma-4;
  in commit `6add3fc`)
- `tests/test_workflow_canonical_baseline.py` (M; W-1 re-baseline;
  in commit `6add3fc`)
- `scripts/worker_iter.py` (M; include_act_breaks overlay; in
  commit `6add3fc`)
- `docs/2026-05-18-overnight-status-4.md` (N -- this file)

## Halt closed

Awaiting Path A / Path B / something-else direction. Same
posture as status #1/#2/#3: pre-authorized fixes overnight remain
same-pattern co-residence OOM only; halt-and-report conditions
unchanged; hard stops unchanged.
