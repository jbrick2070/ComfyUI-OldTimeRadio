<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. Missing local greenlight fallback and integration gaps for scoring-penalty threading, fingerprint propagation, and staging enforcement in the re-outline loop make build impossible as-is.

MUST-FIX BEFORE BUILD:
1. [Candidate 1 / Frontier greenlight] No local greenlight fallback defined. When frontier call fails or is disabled, the pipeline has no way to select a pitch. The plan says "fail-CLOSED to local" but the local mechanism is not specified. Add a local greenlight module (e.g., rubric scoring via a local model or deterministic heuristic using the same rubric axes) and wire the fallback path.
2. [Candidate 2 / Premise fingerprint] The fingerprint from PitchCandidate must survive from the pitch room to the critic escalation handler for PREMISE reroll. The plan says "stored in meta" but does not define the meta‑threading path. Ensure the fingerprint is attached to the episode meta (e.g., `meta["pitch_fingerprint"]`) and accessible when the escalation decision is made, otherwise Tier 2 cannot exclude the failed premise. (ASSUMPTION: episode meta is available to the critic and cascade.)
3. [Candidate 4 / Staging enforcement & Candidate 2 re‑outline] Staging enforcement is POST‑outline, PRE‑composition. Candidate 2’s Tier 1 re‑outline must include the staging penalty in the selection step, else re‑outlined episodes will bypass the climactic guard. Integrate the staging penalty computation into the re‑outline candidate scoring loop (the plan hints “May fold into C2 Tier 1” – this is a hard requirement for a coherent loop).
4. [Candidate 4 / score_outline interface] Adding a penalty parameter to `score_outline` breaks every caller not updated. All call sites (including `select_best_outline`, potential direct callers) must be updated to pass the new parameter, and a regression test must verify byte‑identical output when the penalty is empty. Audit all callers before build.

SHOULD-FIX:
1. [Candidate 2 / EscalationScope.PREMISE] Adding `EscalationScope.PREMISE` before Tier 2 is implemented will crash the freeze cascade if a critic returns `premise_clarity`. Either defer adding the enum value, or add a temporary handler that routes PREMISE → EPISODE.
2. [Candidate 0 / Local probe grading] The probe grades a single‑scene episode; `grade_story` may expect a full multi‑scene structure. Verify that `grade_story` can accept a partial episode or truncate it (e.g., compose an entire episode but grade only the first scene).
3. [Candidate 1 / Greenlight brief length] The `brief_for_outline` mapped to `script_brief` must not exceed the macro prompt’s token budget. Add a length limit and truncation/validation during handoff, else the outline macro prompt will be diluted.
4. [Candidate 2 / Fail‑axis penalty mapping] The translation from failing axis (e.g., `emotional_arc`) to a numeric penalty for `select_best_outline` is unspecified. Define a constant penalty per axis to make the re‑steering reproducible.
5. [Candidate 0 / Gate sequencing] The pipeline must explicitly enforce that Candidate 0 is built and passes before any work on Candidate 1 begins. Hard‑code this gating in the workflow definition or build script.

OPTIONAL / NICE-TO-HAVE:
- Add a separate feature flag for staging enforcement (C4) to allow rolling back independently.
- Log the greenlight decision rationale and fingerprint per pitch for audit trails.

CUT THESE (over-engineering): none evident; Candidate 0 is a well‑scoped gate and the evidence‑quote removal is already done.