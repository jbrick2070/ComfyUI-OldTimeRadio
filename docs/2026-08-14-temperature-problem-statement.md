# Temperatures, the creativity knob, and whether a pass is unlucky or broken

**Problem statement, 2026-08-14. Operator-requested.** For a panel, or for the
next window. Nothing here is a decision; it is the evidence and the questions
it raises.

## The question the operator asked

> "Is it always failing at the first time? That could mean an issue with
> prompting or architecture." … "Or are we using the wrong temperatures?" …
> "Do we need to customise temperatures per model? Are we using the right
> temperatures for Mistral, gemma, etc. in general? I know we have a
> creativity knob — maybe we shouldn't have it, we should just keep the most
> stable, but we don't know until we test."

A retry ladder is a good answer to a STOCHASTIC failure and a terrible answer
to a SYSTEMATIC one. If a pass never survives its first attempt, the prompt or
the schema is wrong and the rerolls are paying up to nine calls to hide it.
From inside one run the two look identical.

## What the corpus actually says

Measured with `scripts/otr_ledger_view.py --ladder`, which reads the call
journal every episode already carries. 60 episodes, 101 with journals.

| pass | ladders | 1st attempt ok | py-fix | avg rungs | abandoned |
|---|---|---|---|---|---|
| P0 fact index | 26 | **0 (0%)** | 25 | 1.1 | 1 |
| P1 question | 20 | 19 (95%) | 0 | 1.0 | 0 |
| P2 cast | 20 | 17 (85%) | 1 | 1.0 | 0 |
| P3 score | 31 | **14 (45%)** | 0 | 1.4 | **11** |
| P5 script | 21 | 18 (86%) | 0 | 1.1 | 1 |

`py-fix` = accepted after a DETERMINISTIC repair: Python fixed it, no second
model call.

**Two passes are systematic, three are healthy.**

### P0 never passes first — and that turns out to be FINE

P0's first attempt is rejected in 26 of 26 ladders, which looks alarming and
mostly is not. It is repaired by PYTHON, not by a model, so it costs **zero
extra LLM calls**. This is not a "we need more passes" situation.

**A first draft of this document said the repair costs "roughly two evidence
rows per episode". That was wrong by about eight times.** Measured properly
across 101 episodes carrying a repair receipt:

| | |
|---|---|
| P0 deterministic repairs | 49 |
| repairs that corrected the offsets and **lost nothing** | **31** |
| evidence rows actually dropped | 26, or **0.26 per episode** |

```
quote_not_literal              30
no_literal_source_spans_remain 30
fact_removed                    3
```

The repair does not blindly drop. `_otr_scifi_source_repair` SEARCHES the
payload for the model's quote and, when it finds it, corrects the coordinates
— the model is bad at counting characters and Python is perfect at it, which
is a sensible division of labour rather than a defect. It only drops when the
quote appears nowhere in the source (the model paraphrased) or in more than one
field ambiguously.

**So the residual question is small and specific:** ~0.26 facts per episode are
lost because the model invented a quote. That is not a temperature problem —
P0 already runs at 0.2 base / 0.1 retry, near-greedy, and no temperature makes
a model reproduce an exact substring dependably. If it is ever worth closing,
the architecture answer is to have the model emit OFFSETS ONLY and let Python
slice the quote, so an invented quote becomes impossible rather than repaired.
**Not currently worth doing on this evidence.**

### P3 abandons a third of its candidates

P3 survives its first attempt 45% of the time and **abandons 11 of 31
ladders** outright — each abandonment throwing away a whole candidate and
starting cold. The dominant terminal error is not prose at all:

```
cast_coverage: every planned cast member must own at least one beat;
               3/4 covered, missing: announcer
```

This is the failure `016ad146` made recoverable on purpose, and it is why
`scifi_news` went from 0-for-4 to reliable. But "recoverable" has become
"routinely spent": a third of P3 draws plan a story that forgets the
announcer. **Question: is that a temperature problem, a prompt problem, or a
schema problem?** A schema that made the announcer beat structurally
un-omittable would remove the reroll entirely.

### P1, P2, P5 are fine

85–95% first-attempt. Rerolls there are doing the job they were designed for.

## The temperatures actually in use

| pass | base | structural retry | notes |
|---|---|---|---|
| P0 | 0.20 | 0.10 | technical slot |
| P1 / P2 / P3 / P5 family | 0.72 | 0.32 | creative slot |
| typed repair rung | `REPAIR_TEMPERATURE` | — | shared constant |
| repair syntax retry | — | floor 0.25 | |

The corpus shows historical drift — P3 rung 1 appears at 0.1, 0.32 AND 0.72
across archived episodes — so these values have moved over time and the older
runs in the table above were not all drawn at today's settings. **Any
conclusion about temperature drawn from the archive is therefore weak.** A
clean comparison needs fresh runs at pinned settings.

## The creativity knob, and a live inconsistency

`creativity` maps to `(temperature, top_p)` in `OTR_LedgerScriptWriter`
(`balanced` → 0.85 / 0.95) and drives the WRITER lane's line composer.

**The codex lane hard-codes `base_temperature=.72` on every pass and never
reads it.** So on `scifi_news` the operator's creativity knob does nothing to
the story passes. On the writer-lane banks it does. That is a real
inconsistency: the same widget means two different things depending on the
bank, and on one of them it means nothing.

Three coherent positions, and they should be chosen between rather than
drifted into:

1. **Delete the knob.** Pick the most stable value and pin it. The operator's
   instinct. Costs a dial nobody has proven is useful.
2. **Make it real everywhere.** Route it into the codex lane too, so the
   widget means one thing on all six banks.
3. **Keep it, scope it honestly.** Rename it so it says which lanes it
   governs, and stop implying it steers `scifi_news`.

Doing nothing keeps a widget that lies on one bank, which is the worst of the
three.

## Per-model temperatures

Not currently supported: temperature is per-PASS, never per-model. The
question is whether the same 0.72 is right for `Mistral-Nemo`, `gemma-4-12b`
and `gemma-2-2b`, whose sampling behaviour differs.

**One live data point, and it is suggestive.** The 2026-08-14 `scifi_news`
leg on `gemma-4-12b-it` (4-bit NF4) halted **three verbatim-cycle runaways** in
one episode — P2 twice, P3 once — each caught by the two-signal decode guard
and rerolled cooler. P2 needed all three rungs. The same lane at the same
settings has produced clean single-attempt runs before. Whether that is the
model, the quantisation, the source, or the day is exactly what a matrix would
settle.

**Caveat on the ladder as a temperature experiment:** the third rung is the
INFORMED repair — the model is told what was wrong. But a decode that was
HALTED produces no artifact to repair, so on a cycling pass the repair rung is
effectively just another cooler roll. Reading rung-3 success as "the repair
works" would be wrong on those.

## What would settle it

A matrix, one variable at a time, on a pinned source so the story is not a
confound:

1. **Model** — Mistral-Nemo, gemma-4-12b, gemma-2-2b at today's temperatures.
   Record first-attempt survival per pass, halts, and wall time.
2. **Temperature** — the winner at 0.72 / 0.5 / 0.3 base.
3. **The knob** — whether any creativity setting changes an F1/F2 outcome, or
   only the prose. If only the prose, the 2026-08-04 directive says it is not
   worth a widget.

`--ladder` already reports 1 and 2 from the journals with no extra
instrumentation. Three runs per cell is the minimum that distinguishes bad
luck from a broken pass.

**Not started, by operator direction 2026-08-14:** "maybe now's not the time
for a multi-test; we need to regress and fix bugs." This document exists so
the question survives that decision.
