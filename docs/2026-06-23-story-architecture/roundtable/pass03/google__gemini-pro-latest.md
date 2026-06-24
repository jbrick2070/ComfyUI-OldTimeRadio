<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Critical routing logic in the escalation module will swallow the new PREMISE scope because `premise_clarity` is already hardcoded to trigger an EPISODE reroll.

MUST-FIX BEFORE BUILD:
1. [Candidate 2] **Escalation Routing Collision.** The grounding code defines `STRUCTURAL_AXES = frozenset({"premise_clarity", ...})` and routes *all* structural hits to `EscalationScope.EPISODE`. Adding `EscalationScope.PREMISE` is useless unless you also remove `premise_clarity` from `STRUCTURAL_AXES`, create a new `PREMISE_AXES` set, and add a specific `if premise_hits:` routing block above `structural_hits` in `decide_escalation_scope()`.
2. [Candidate 0] **Circular Sequencing.** C0 is defined as the FIRST step to break circular dependencies, but its spec says "take the greenlit premise -> outline". Greenlighting happens in Candidate 1. C0 must either run its own mini-pitch or use the raw seed directly, otherwise it cannot run before C1.
3. [Candidate 1] **Immutable Dataclass Violation.** Plan says "map the winning PitchCandidate into the EXISTING OutlineRequest.script_brief". `OutlineRequest` is a frozen dataclass (per R2 facts). You cannot map into it mutably; you must use `dataclasses.replace(outline_req, script_brief=...)`.

SHOULD-FIX:
1. [Candidate 1] **External Call Resilience.** The `OTR_GREENLIGHT_MODEL` OpenRouter call specifies "fail-CLOSED to local" but lacks explicit timeout and retry parameters for the network boundary. Add a strict timeout (e.g., 10s) and max 1 retry to prevent pipeline stalls.
2. [Candidate 4] **Signature Mutation.** "feed as a numeric penalty INTO score_outline inputs". `score_outline` signature is `(outline, meta, roster)`. Pass the penalty via the `meta` dict rather than changing the pure function's signature, or explicitly add an optional `penalty: float = 0.0` kwarg.

CUT THESE:
1. [Candidate 0] **Grading out to the scene level for the local-ceiling probe.** Safe to cut. Grading the outline alone is sufficient to prove local LLM competence and avoids the heavy compute cost of composing a full scene just to test the prompt ceiling.