<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan lacks concrete, validated designs for the two highest-impact defects (1 and 2); the "no-op on a good script" claim is unsubstantiated and the proposed freeze-floor change risks silently shifting the audio golden on strong-model output.

MUST-FIX BEFORE BUILD:
1. [Sec 2, Defect 1 / Sec 4, Q1-2] The detection algorithm for trailing/embedded stage directions is not specified. The plan says "add a detect_trailing_embedded_stage_direction … with a tight false-positive guard" but does not define the primitive (sentence-segmenter vs span regex), the classification rules, or how it distinguishes spoken narration from bare action. Without this, the fix cannot be implemented safely. MUST provide a detailed, testable specification and a validation plan against strong-model output before it enters the deterministic freeze floor.
2. [Sec 2, Defect 2 / Sec 4, Q3] The stance-coherence signal is a placeholder. The plan says "add a per-character STANCE-COHERENCE signal" but does not decide whether it is a critic axis, a deterministic tracker, or an outline-stage guard. The existing story critic has no such axis, so this requires a new subsystem. MUST choose the mechanism, define its interface, and specify how it integrates with the scoped reroll (e.g., a new `failed_dimension` value) before build.
3. [Sec 1, "Audio spine FROZEN" / Sec 2, Defect 1] Adding trailing/embedded stripping to the deterministic freeze floor will change the frozen text for any line that matches the pattern. If the strong model ever produces such a line (even if it does not in the current smoke), the audio golden would shift silently, violating the byte-identical invariant. The plan acknowledges this risk but only says "flag it." MUST add a concrete gate: either a strong-model regression test proving the new scrub never fires on a known-good script, or an operator-gated golden recapture process with explicit approval.
4. [Sec 1, Hard invariants / Sec 3] The plan assumes the new gates are no-ops on a strong model, but this is not verified. The acceptance criteria only require the 5 corpus lines to be stripped from the weak-model smoke. MUST include an acceptance test that the gates do not fire (or fire with an acceptable false-positive rate) on a representative strong-model script, to uphold the "no-op on a good script" invariant.

SHOULD-FIX:
1. [Sec 2, Defect 3] The fix location for the role<->char_id consistency assert is undecided. Resolve whether to place it at init, set_lines, or the role_mismatch repair guard before coding to avoid rework. The most robust is likely at the point where `speaker_role` is written (init/set_lines), preventing the inconsistency from entering the ledger.
2. [Sec 2, Defect 4] The plan is ambiguous: it proposes a measurement signal but also marks the gate as "candidate CUT." Decide explicitly whether to cut the proportion/setup gate or implement a minimal measurement-only signal. Leaving it unresolved risks scope creep.
3. [Sec 3] The acceptance criteria mention "freeze gate + reroll firing" but the plan says "Re-smoke WITHOUT OTR_BYPASS_FREEZE_HALT." Clarify that the re-smoke must run with the freeze gate enabled (bypass off) to validate the end-to-end fix, not just the reroll.

OPTIONAL / NICE-TO-HAVE:
- A false-positive heatmap for Defect 1 using a corpus of strong-model dialogue would de-risk the freeze-floor change.
- A decision record for Defect 2's stance-coherence approach (e.g., whether to use outline beat intent labels as a deterministic tracker) to guide the roundtable.

CUT THESE (scope / over-engineering):
- [Sec 2, Defect 4] The proportion/setup gate for UN escalation. The document itself suggests it may be a symptom of Defect 2 + weak model. If the panel agrees, cut the gate entirely; a measurement-only signal is low-cost and can be retained if desired, but a full gate is over-engineering for a defect that may not exist on the strong end.

[ASSUMPTION] The plan assumes the strong model never produces trailing/embedded stage directions that would be caught by the new detection. This is not validated and must be tested before the freeze-floor change is accepted.
[ASSUMPTION] The plan assumes the existing scoped reroll machinery can accept a new `failed_dimension` for stance coherence without breaking the convergence loop. Verify that `_otr_reroll.py` and `compose_line` can handle an additional dimension without altering the reroll budget or halting logic.