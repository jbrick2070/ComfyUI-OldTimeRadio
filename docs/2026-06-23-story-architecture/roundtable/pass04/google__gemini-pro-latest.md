<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is structurally sound and the integration seams are well-defined, but there are build-blocking ambiguities in tie-breaking, penalty math, and temporary enum routing that will cause divergent implementations or grader failures.

MUST-FIX BEFORE BUILD:
1. [C1] **Tie-breaker ambiguity**: "tie-break on numeric console_standoff_risk then id" does not specify direction. **Fix**: Explicitly state "tie-break on numeric console_standoff_risk (ASCENDING - lower risk wins), then id".
2. [C0] **Grader length bias**: "compose SHORT full episodes (reduced word budget) -> grade_story". Standard LLM graders heavily penalize short outputs for "lack of depth" or pacing issues. **Fix**: Either use the standard word budget for the probe, or explicitly inject a `[Note: This is a short-format test, do not penalize length]` directive into the `grade_story` prompt for C0.
3. [C2] **Enum temporary routing crash risk**: "Until Tier 2 ships, route PREMISE -> EPISODE (temporary) so no consumer crashes." If `EscalationScope.PREMISE` is added to the enum, returning it will break exhaustive match/switch statements downstream. **Fix**: Have `decide_escalation_scope` return `EscalationScope.EPISODE` when premise axes fail during the Tier 1 phase. Do not leak the `PREMISE` enum value out of the routing function until Tier 2 is actually built.
4. [C4] **Penalty math undefined**: "Add an explicit optional `penalty=None` kwarg to `score_outline`". It is not stated how this penalty is applied. **Fix**: Specify exactly how it applies (e.g., "subtract `penalty` directly from the final computed float score").

SHOULD-FIX:
1. [C1] **Fail-closed timeout**: "timeout 10s, 1 retry, fail-CLOSED to local". 10s is extremely aggressive for an OpenRouter LLM call doing structured JSON generation for a greenlight rationale. **Fix**: Increase timeout to 30s for the frontier greenlight call.

OPTIONAL / NICE-TO-HAVE:
- [C2] Log a distinct warning when `OTR_STORY_REPITCH_MAX` or `OTR_STORY_REPLAN_MAX` is exhausted so it's easily queryable in telemetry.

CUT THESE:
1. [C0] "Grade the best few TWICE; pass = both >= 75." **Why**: Waste of compute. This is a local-ceiling probe, not the final production output. A single grade >= 75 is sufficient signal to set the greenlight flag.

VERIFY-AT-BUILD:
- [ASSUMPTION] Verify exact `..._PALETTE` + `BEAT_ROLE` symbol names and their public visibility in `_otr_story_quality_l12`.
- [ASSUMPTION] Verify `OutlineRequest` is actually implemented as a `@dataclass(frozen=True)` before attempting `dataclasses.replace`.
- Verify `_refine_loop` body actually re-invokes outline+compose each pass and accepts a critique source swap without breaking keep-best logic.
- Verify exact JSON field name for `use_exchange` in the configuration schema (C3).
- Verify macro-prompt length tolerance for the injected richer `script_brief`.