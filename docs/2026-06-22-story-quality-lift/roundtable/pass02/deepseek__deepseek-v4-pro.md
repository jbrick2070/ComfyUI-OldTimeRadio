<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — the core algorithms for stage-direction detection/classification (Defect 1) and the stance-reversal repair mechanism (Defect 2) are explicitly left open (R2 deferral), making the plan not codeable as written.

MUST-FIX BEFORE BUILD:
1. [Section 2 / Defect 1 — detection/classification]
   The plan defers the detection primitive and exact classification rules to R2. The coder cannot implement Tier 2 (reroll detection of trailing, embedded, undelimited stage directions) or Tier 3 (deterministic strip of “HIGH-CONFIDENCE cases outside a matched quote pair”) without a concrete algorithm. The requirements (negative fixtures, well-formedness) do not substitute for a working classifier.
   Fix: Provide a specific, reviewable detection method (sentence-segment-then-classify, regex-based patterns, etc.) including rules for distinguishing action text from legitimate spoken content (b015’s malformed quoting, b017’s undelimited line). This must be defined before any coding begins.

2. [Section 3 / Defect 2 — repair mechanism]
   The plan states “Decide in R2: outline re-intent vs episode escalation, and the determinism/cost of a rerun on a weak model”. Without this decision, the coder cannot implement the repair path — there is no specification of where the coherence hint is injected, which cascade function is invoked, or how the rerun is triggered.
   Fix: Choose and document the repair flow (e.g., mark the episode for full rerun via `needs_full_rerun` and inject a coherence hint into the outline prompt; or re-intent the antagonist’s `_wants_must_oppose`). Also specify how determinism is maintained when a weak model is forced to rerun.

3. [Section 2 / Tier 3 — classification primitive]
   The plan requires the deterministic floor to “classify the span as action, not just locate it”. No primitive is provided (sentence segmentation, part-of-speech tagging, verb-set matching, etc.). The coder cannot implement a high-confidence strip without guidance on what an “action” clause looks like programmatically.
   Fix: Define a concrete heuristic (e.g., use a list of third-person present-tense action verbs, require that the span contains no first/second-person pronouns, enforce that it follows punctuation) with clear acceptance criteria and known false-positive rates.

4. [Section 4 / Defect 3 — cast-id check in repair guard] [ASSUMPTION]
   The plan says the role_mismatch repair (`_otr_ledger_reviewer.py:1054-1070`) should “reject `expected="announcer"` from the repair when char_id is a cast id”. The grounding does not show whether the cast mapping (the episode’s cast dict) is available in that repair function. If it is not, the coercion logic cannot be implemented without passing additional data.
   Fix: Verify that the cast dictionary is accessible in the repair function’s scope, and if not, adjust the design to inject it (e.g., pass as a parameter or pull from the episode state). Document the dependency.

5. [Section 4 / Defect 3 — additional write points] [ASSUMPTION]
   The plan lists only `init_lines_from_outline`, `set_lines`, and the role_mismatch repair as coercion points. Any other code path that sets `speaker_role` (e.g., other repairs in `_otr_ledger_reviewer`, or direct assignment during line composition) could still produce an inconsistent role for a cast char_id.
   Fix: Audit all code that writes `speaker_role` and ensure each incorporates the cast-id coercion. If none exist, document this as confirmed.

6. [Section 7 / open questions 1-4]
   All four open questions are deferred to R2. Since this document is the “coder kickoff,” these gaps block implementation. They must be answered before any developer can start.
   Fix: Resolve each question — (1) exact detection algorithm; (2) repair mechanism; (3) trace origin of stamp; (4) strong-model fixture ledger path and assert — and update the plan accordingly.

SHOULD-FIX:
- [Section 7 / strong-model NO-OP fixture] The plan references a known-good ledger for proving zero strips/rerolls but does not specify which file or the exact assert. For build, the coder needs the fixture path and the acceptance criteria (e.g., `pytest` test that runs the suite and asserts zero calls to strip/reroll functions). Clarify this early to avoid blocking integration tests.

OPTIONAL / NICE-TO-HAVE:
- None; the core gaps must be resolved for any build.

CUT THESE (over-engineering):
- None; the plan is not yet over-engineered — it lacks essential implementation detail.

[ASSUMPTION] points noted inline where the plan’s claims depend on code or state not visible in the grounding.