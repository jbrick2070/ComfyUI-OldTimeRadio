<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The diagnosis of structural sameness is sharp, but the proposed B3/B4 extraction parser is a massive, high-risk architectural rewrite disguised as a feature, and routing structural failures to the pitch room conflates bad plotting with a bad premise.

MUST-FIX BEFORE BUILD:
1. [Section 2 / 3.8] Routing structural critic failures to the pitch room abandons the premise entirely. If an episode has a "flat middle" or "climax off-stage" (structural failures), the premise isn't necessarily wrong; the *beat outline* is. Concrete fix: Route structural failures to a divergent *re-outline* step for the same premise, not all the way back to the pitch room. Only route back to the pitch room if the structural critic explicitly flags the premise as unsalvageable.
2. [Section 2A / 6] The B3/B4 "whole-episode free prose -> transcribe to ledger" extraction parser directly contradicts the mission statement that "schema/plumbing is done and good". Building a flawless prose-to-JSON speaker attribution parser is a massive undertaking that risks breaking the entire downstream A/V freeze cascade. Concrete fix: Cap the Axis B ladder at B2 (`use_exchange`). Remove B3/B4 from Sprint 3. Do not build a reverse-extraction parser until B2 is proven mathematically insufficient.

SHOULD-FIX:
1. [Section 2 / 3.3] The "showrunner taste pass" relies on the LLM having better taste than generation ability. If the local model ceiling is a "B", a local taste-selector will likely just pick the most average "B" premise. Concrete fix: Hardcode the "Greenlight / Taste Pass" node to default to the frontier OpenRouter lane, even if the draft generation remains on the local model. 

OPTIONAL / NICE-TO-HAVE:
- [Section 3.6] "require every beat to change the temperature" is a prompt-level instruction, but enforcing it mechanically requires a new structural critic pass on the outline *before* generation. Consider adding an outline-specific critic rather than waiting for the post-generation 5B critic to catch flat beats.

CUT THESE (scope / over-engineering):
1. [Section 3.5] Character interviews. Safe to cut because the triangulated root cause is *structural sameness* (plot/premise), not character voice. This is pure bloat that does not serve the stated primary goal.
2. [Section 3.4] Theme & ending first. Safe to cut because `_otr_outline.py` already handles the macro premise -> beat hierarchy. Forcing the planner to "write toward the last 20 seconds" requires a fundamental rewrite of the outline prompts and context windows, which violates the "do not rebuild" constraint.

[ASSUMPTION] The plan assumes the existing `_otr_outline.py` best-of-N deterministic selector can seamlessly ingest the output of the new "Pitch room" without schema mismatches.
[ASSUMPTION] The plan assumes the local model (gemma-12b / mistral-nemo) is capable of generating radically divergent genres and emotional cores from a single news seed in the pitch room, rather than just 3 slight variations of a console standoff.