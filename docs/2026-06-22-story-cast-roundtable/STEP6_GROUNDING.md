# STEP 6 ground-first (go/no-go) -- beat-planning arc

2026-06-22. The plan's STEP 6 first sub-task: read the beat/outline planner and
MEASURE whether beat_tension escalates + objective/turn are strong, BEFORE
defining any change. Do NOT add a SceneArcContext (LineRequest already carries the
arc fields). Grounded against the real Windows source + the live re-soak ledgers.

## Finding 1 -- the composer IS arc-rich (confirmed; no SceneArcContext)
`_otr_line_composer.LineRequest` (~L571-725) already carries arc_phase,
dramatic_question, beat_objective, beat_obstacle, beat_turn, beat_subtext,
next_turn, outline_spine, current_beat_block. The writer populates them at
`OTR_LedgerScriptWriter.py:~3420-3458` (objective/obstacle/turn/subtext via the
A5 news-driven slot contract). A SceneArcContext would duplicate these -- DO NOT
ADD ONE (matches the plan).

## Finding 2 -- arc_phase ESCALATES (measured, real ledgers)
Both re-soak episodes show a clean monotonic phase arc, exactly as the outline
validation enforces (`_otr_outline.py` #5: voiced beats grouped by arc_phase in
budget order):
  datas_out:        setup x4 -> complication x5 -> resolution x4
  scorched_streets: setup x3 -> complication x6 -> resolution x4
So the arc STRUCTURE is sound and escalating; the "uneven arc" is NOT a broken
phase order.

## Finding 3 -- beat_objective / beat_turn are STRONG (measured)
Per-beat intents are concrete and dramatic (real ledger samples): "secretly
broadcasts a distress call, bypassing...", "tenders his resignation in protest",
"defies the mayor's order to suspend...". Objectives/turns are NOT the weak link.

## Finding 4 -- THE GAP: beat_tension is UNWIRED (the actual lever)
`beat_tension` (LineRequest field, "0 = unset; renders only when 1..5",
composer L719/L1213) is **never assigned a value anywhere in the tree**. Repo-wide
grep: the ONLY hits are the dataclass default (0), the render guard, a writer
comment "(beat_subtext / beat_tension) stay empty for Path A", and NO
`beat_tension=` at the LineRequest construction (writer ~3452 passes
objective/obstacle/turn/subtext but omits tension). So the composer's
"Tension: N/5" cue NEVER renders -- there is no per-beat intensity gradient
feeding the line writer, even though the PHASE gradient exists.

## GO / NO-GO: GO (small, contained, deterministic)
The lever is to WIRE beat_tension as an escalating 1..5 signal and feed it to the
composer -- NOT to restructure the arc (sound) or rewrite objectives (concrete).

Proposed STEP 6 (define, then build):
- In the beat planner / writer LineRequest construction, derive a deterministic
  escalating `beat_tension` from the beat's arc_phase position (phase index across
  budget.arc_phases) + ordinal within the phase: e.g. setup -> 1..2, complication
  -> 3..4, resolution/climax -> 4..5, climbing toward the final phase. Source it
  from data already on the validated outline beat (arc_phase + beat index); no new
  socket, no JSON change.
- Wire `beat_tension=<derived>` into the LineRequest at writer ~3452 (the same
  block that already delivers beat_objective/turn). The composer's existing
  L1213 guard then renders "Tension: N/5" so the line writer sees the gradient.

## STEP 5 pairs with it (flat rubric + failed_dimension)
The critic's flatness judgment should reference the per-beat tension TARGET: a
`character` line is flat if it does not advance its beat_objective AND does not
move the scene toward its target tension. Add the shared 5-dimension rubric to the
critic PROMPT + a `failed_dimension` enum, updating the critic output schema and
the `_otr_reroll.py` hint parser/consumer in the SAME change (rubric-guided LLM
judgment, not a deterministic code test).

## Re-soak cross-check (both legs, bypass OFF)
datas_out: frozen_with_doctor_edits, diverged=False, cycles=2, outstanding=['b010']
scorched_streets: frozen_with_warns, diverged=False, cycles=2, outstanding=3
-> STEP 4 convergence + repair-then-ship confirmed on real episodes; the residual
outstanding beats are the subjective-quality flags STEP 5/6 target.
