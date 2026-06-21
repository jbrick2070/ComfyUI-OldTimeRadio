<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no -- critical gaps in F1 (undefined beat band, ambiguous token cap), F3 (no deterministic hedge prevention), F7 (missing first-person narration handling), and measurement harness (no seed control, undefined outro detection) would prevent meeting acceptance targets or make results unverifiable.

MUST-FIX BEFORE BUILD:
1. [T1.1] F1: Define how beat_lo/hi are computed from the episode target and beat allocation; clarify whether the token cap formula applies per-line or per-beat and adjust accordingly; specify the new 2-attempt ladder logic or confirm it remains unchanged. Without this, the length fix cannot be correctly implemented.
2. [T1.3] F3: Add a deterministic post-generation scrub for hedging phrases in the outro, or accept that 0 hedges may not be achievable with prompt-only. The current plan risks failing the acceptance target.
3. [T2.4] F7: Extend the narration hygiene regex to catch first-person narration (e.g., lines starting with "I" followed by action verbs) or explicitly exclude first-person from the 0-narration target. Also detail the self-address detection logic.
4. [Sprint 0] T0.1: Define the outro-vs-ending agreement detection method (e.g., list of hedging phrases) and implement it in the scan script before Sprint 1 exit can be measured.
5. [General] Specify a fixed random seed for all smoke runs to ensure before/after comparability; record the seed in the baseline.
6. [T1.2] F2: Define a fallback for when no character voiced beat has a must_turn contract, ensuring the chosen slot still yields a valid episode (e.g., force a contract on the fallback slot or select a beat that can have one). Otherwise the 90% valid target may be unachievable.

SHOULD-FIX:
1. [T1.1] F1: Add a test that verifies no line is truncated mid-sentence in the 864-word smoke.
2. [T2.1/T2.2] F4/F5: Create automated checks for gender/pronoun consistency and speech-register distinctiveness to make acceptance measurable.
3. [T2.3] F6: Include a length regression check in the smoke to ensure the unconditional rider does not push the ratio out of band.
4. [T3.2] F9: Before building, produce a detailed design for the pipeline reorder; otherwise defer F9 to a later sprint.
5. [T3.1] F8: Document the arc_shape templates and their integration to reduce implementation risk.

OPTIONAL / NICE-TO-HAVE:
- Add "length-pass not firing" detection to the scan script.
- Automate style-descriptor duplication metric in the scan script.
- Consider merging F6 into Sprint 1 alongside F1 to reduce merge conflicts.

CUT THESE (over-engineering):
1. [T3.2] F9: Cut the outline reorder entirely. F2 and F8 already address costly-choice binding and arc variety; the reorder adds significant risk and complexity without clear necessity. Safe to cut.
2. [T2.2] F5: Speech-register cue can be deferred; it's a nice-to-have that does not block core acceptance targets. Cutting reduces prompt engineering risk.

[ASSUMPTION] The per-beat target words are derived from the episode target_words divided by number of beats; the token cap formula is applied per line generation call; the "2-attempt ladder" exists as a retry mechanism; `_NARRATION_LEAK_REGEXES` include self-address patterns; the "length-pass" is a detectable truncation mechanism; the scan script can detect outro hedging via regex; the Bug Bible regression suite is maintained; WIRING_PLAN.md contains widget change instructions.