<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan introduces fatal schema contradictions and conflates `beat_role` with `arc_phase` in the budget validators, which will break outline generation.

MUST-FIX BEFORE BUILD:
1. **[L1/L2] Schema Contradiction:** The plan mandates "ZERO workflow-JSON change" (Hard Constraint #3), but L1 requires adding `conflict_object` + `conflict_type` "SLOT"s to the beat, and L2 requires carrying `beat_role` + `conflict_object` into the composer. The `LineRequest` grounding (`_otr_line_composer.py:1223`) only accepts `beat_objective`, `beat_obstacle`, `beat_turn`, `beat_subtext`, and `beat_tension`. You cannot pass new tags into the dramatic frame without modifying the `LineRequest` and `Beat` schemas. **Fix:** Explicitly authorize adding `conflict_object`, `conflict_type`, and `beat_role` to the `Beat` and `LineRequest` JSON schemas, or explicitly define how they pack into an existing `meta` dict on the beat level.
2. **[L2] Validator Conflation:** The plan states "adding required slots to `EpisodeBudget.arc_phases` MUST update the monotonic arc_phase validators". This conflates narrative phases (e.g., "Act 1") with dramatic functions (`beat_role`). If you replace `arc_phases` with `beat_roles`, you break `per_phase_words` and `per_phase_beats` zip logic (`_otr_outline.py:781`). If `beat_role` is a separate field, the existing `arc_phase` validator (`_otr_outline.py:853`) cannot check it. **Fix:** Do not overload `arc_phases`. Add a dedicated `beat_role` field to the `Beat` schema and write a *new* validator specifically to enforce the `personal_stake -> irreversible_choice` ordering contract.
3. **[L1b] Missing Domain Signal:** The plan relies on a "domain/category signal" to pick a conflict palette, noting "VERIFY a category field exists in meta (else classify from the logline)". The grounding for `meta` only shows `allowed_roster`. Classifying via LLM inline violates the deterministic/no-LLM-gate constraint. **Fix:** Define a deterministic fallback palette (e.g., a generic "institutional power" palette) to use when the domain cannot be strictly matched, rather than relying on an unverified field or an on-the-fly LLM classifier.

SHOULD-FIX:
1. **[L3] Delimiter Stripping Risk:** Stripping `[...]` via regex before TTS is standard, but weak models often use brackets for valid text (e.g., acronyms, redactions) or fail to close them. **Fix:** Ensure the regex only strips brackets if they contain known stage-direction keywords or span an entire standalone paragraph, to avoid swallowing valid dialogue.

OPTIONAL / NICE-TO-HAVE:
- **[L1a] Roster Filtering:** When anchoring beats on `allowed_roster`, filter out "ANNOUNCER" (seen in grounding) so the model doesn't try to use the narrator as a physical conflict object.

CUT THESE:
1. **[L6] Best-of-N line selection:** Safe to cut as noted in the plan. It scales poorly locally and doesn't solve structural sameness.

[ASSUMPTION] The plan assumes `meta.story_quality` and `objective_literal_retry` flags exist in the harness for L5a, which are not visible in the grounding.