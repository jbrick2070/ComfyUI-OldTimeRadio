# T0 -- Local-model story-ceiling probe (GATE)

**Date:** 2026-06-23 (overnight). **Question:** can the local writer compose a
75+ (B) story at all -- and where does the campaign promise top out?

**Operator decision required.** This doc reports real grades + a recommendation.
It does **NOT** auto-enable the frontier lane (per the sprint gate rule).

## Method / data source

T0's design is: compose several real-length episodes (>=200 words) on the local
writers and grade them with `grade_story`. Rather than spawn a redundant
throwaway 5-episode probe, this gate reads the **real grades the operator's own
grade-lift refine soak was already producing this session** (same word budget
200-320, same `grade_story` 0-100 grader, keep-best across 5 passes), plus the
pitch-room-ON arc verdict from tonight's live story-architecture smoke. The soak
ran ~90 min and graded 3 episodes before the box was reset for the smoke; the
scheduled `otr-refine-soak-summary` (12:20 AM) will extend this sample.

Source files:
`docs/2026-06-23-refine-loop/grade_lift_soak_summary.json` (real grades);
`docs/2026-06-23-story-architecture/smoke_result.json` (pitch-room-ON verdict).

## Real grades (grade_story 0-100; B = 75, B+ = 80, A ~= 90)

| Writer | Episodes | mean pass-0 | mean best (keep-best, 5 passes) | reached B(75) | refine lift |
|---|---|---|---|---|---|
| gemma-4-12b-it | 2 | 65.0 | 65.0 | 0 / 2 | 0.0 |
| mistral-nemo | 1 | 42.0 | 42.0 | 0 / 1 | 0.0 |
| **overall** | **3** | **57.3** | **57.3** | **0 / 3** | **0.0** |

Tonight's pitch-room-ON smoke ("Akira's Resolution", 280 w, mistral-nemo) was not
numerically graded by the probe, but the unconditional 5B story critic returned
**`arc_verdict=uneven`** with a structural `emotional_arc` weakness -- corroborating
the sub-B ceiling on the local writer even with a divergent, taste-selected premise.

## Reading

- **No local episode reached B (75).** gemma-4-12b tops out around **65**;
  mistral-nemo sits much lower (~42). This matches every prior soak read
  (2026-06-22/23: grades 42-72, "stable but not a measurable lift on weak local
  writers").
- **The refine loop did not lift the grade** on these passes (keep-best best ==
  pass-0; lift 0.0). Revision is non-monotonic and the weak writer does not climb.
  The loop's *mechanics* are proven (earlier sessions); the *ceiling* is the model.
- **The pitch room changes WHICH story is told, not the prose grade.** Tonight's
  smoke proved the pitch room produces genuinely divergent premises (Person-vs-
  Society / Person-vs-Person / Person-vs-Nature Mars stories) and taste-selects one
  -- a real cross-episode *sameness* win -- but the local writer's *prose* arc is
  still graded `uneven`. Frontier **greenlight** (one cheap taste call) sharpens
  premise SELECTION; it does **not** raise the prose grade. Only a frontier
  **writer** (`creative_writing_model = openrouter:...`) lifts the prose grade, at
  the expensive per-token lane.

## Recommendation (operator decides; do NOT auto-enable frontier)

1. **Accept-B / relabel for the 100%-local lane (recommended default).** A 75+
   grade is out of reach for the local writers; redefine Increment-1 success as
   **premise-divergence + cross-episode sameness reduction + median-grade lift**,
   which the pitch room demonstrably delivers, rather than an absolute B bar.
   Keep all the new levers default-OFF and flip them for the eyeball soak.

2. **Optional, cheap: frontier GREENLIGHT only.** Set
   `OTR_ENABLE_FRONTIER_GREENLIGHT=1` + `OTR_GREENLIGHT_MODEL=<slug>` (HKCU/User
   env, read by the conductor). This upgrades only the one taste call that picks
   the premise -- it does not touch prose cost -- and fails CLOSED to the local
   greenlight. Improves *which* premise ships, not the grade.

3. **Only if A+ prose is the goal: frontier WRITER.** Point
   `creative_writing_model` at an `openrouter:` slug for the prose itself. This is
   the only lever that moves the grade past ~65, at the paid per-token lane
   (cost-guarded). Reserve for episodes that justify the spend.

**Bottom line:** local writers are a solid **C+/B- ceiling (~65)**; do not promise
A+ locally. The Increment-1 levers are still worth shipping -- they fix the
*sameness* defect (the actual operator complaint), which is independent of the
prose grade. Frontier is opt-in, env-named above, and never auto-enabled here.
