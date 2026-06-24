<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — The core insight (vary premise, close critic‑planner loop) is sound, but the plan lacks the critical feedback path from critic to re-pitch and defers definition of the taste selector; B4 is premature for the initial sprint.

MUST-FIX BEFORE BUILD
1. [Section 2 #2, Section 3 #8] The structural‑failure route sends the critic’s verdict to the pitch room, but *does not inject the critic’s report into the new pitch*. Without that feedback, the loop is a blind reset; the next premise may repeat the same flaws. Fix: when Wave‑1C escalation fires, pass `StoryCriticReport.arc_verdict` and `flat_lines` as a “showrunner note” into the pitch‑room prompt, requiring the new pitch to confront those weaknesses (e.g., “last take had an off‑stage climax and middle flat‑lines; generate a take with an on‑mic climax and rising tension”).
2. [Section 2 #1, Section 3 #3] The greenlight taste selector is described only as “showrunner with taste and no budget”. No selection rubric, ranking method, or prompt exists. A naive LLM call may pick at random or bland entries. Fix: design a concrete showrunner prompt that forces explicit criteria (emotional hook, surprise, stakes, clarity) and a forced ordinal ranking; ground it against the existing `score_outline` to ensure it does not collapse to the deterministic metric.
3. [Section 2A “On B4”, Section 6 sprint 3] The B‑ladder climb (B3→B4) introduces a whole‑episode‑prose‑to‑ledger parser whose design is the “make‑or‑break”. Including it in the initial sprint sequence risks stalling the primary lever (premise divergence) while the parser is unproven. Fix: remove B3/B4 from the immediate sprint list; treat it as a parallel research track with its own SPEC and parser feasibility study. The first sprint must deliver only pitch room + critic‑loop closure + `use_exchange` flip.

SHOULD-FIX
1. [Section 3 #1] The assignment‑desk “surface 3 headlines” step does not specify whether the news interpreter emits multiple script_briefs in one call or is called three times. A single‑call prompt may produce less variety. Clarify the expected implementation and confirm the chosen method yields measurable divergence.
2. [Section 5] The `use_exchange` flip is gated on a live metric (VRAM ≤ 14.5 GB, zero slot drift, N=3) that requires a GPU. The SPEC must define the exact test harness, the pass/fail criteria, and a fallback (keep default OFF) if the live test fails, to avoid a single‑run bias.
3. [Section 0, Section 2 #2] The claim that Wave‑1C “re‑runs the same beat‑planner, which produces the same shape” should be verified in the real code. If the planner already has stochasticity, the loop‑fix rationale still holds, but the wording should be tightened. The SPEC should include a short code‑inspection note to confirm the degree of determinism.
4. [Section 5] The operator decision (frontier lane vs accept‑B) is unresolved and could render the whole campaign moot if the local model cannot produce an A‑worthy pool. Add a gate: before committing to the pitch room, run a small experiment (e.g., 10 local pitch‑room sets→full compose→grade) to see if any episode reaches 75+. If none do, pause and escalate the frontier‑lane decision.

OPTIONAL / NICE-TO-HAVE
- The “beat temperature” rule (every beat must change power) is a line‑level refinement that can follow after premise diversity is proven; it is not needed for the initial sprint.
- Character interviews for distinct voices can be a follow‑on, not part of this core architecture effort.

CUT THESE (scope / over‑engineering)
- Cut “Theme & ending first” (Section 3 #4) from the immediate spec. It adds a separate mini‑cycle that is not essential when premise divergence already targets the root cause; can be resurrected if pitch‑room alone fails.
- Cut the detailed survey of external repos (Section 4) from the SPEC; only extract specific prompt snippets if directly used. The SPEC should not be a reference library.
- The “fix non‑monotonic refine” lever is a separate engineering defect best handled after the new levers are stable; keep‑best already masks it sufficiently for the sprint.

[ASSUMPTION] The panel has access to the live repository and can verify that Wave‑1C’s regenerate path can accept a new premise without breaking the cascade.
[ASSUMPTION] The taste‑selector LLM call will fit within the same compute budget as the existing pipeline components.
[ASSUMPTION] The local model can generate pitches with enough diversity; the gate (SHOULD‑FIX #4) is the mitigation.
[ASSUMPTION] `use_exchange` has been unit‑tested in isolation and only needs integration validation.