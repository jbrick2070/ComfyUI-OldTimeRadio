<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan correctly identifies the weak-model artifacts but proposes architectural mismatches and pipeline-crashing levers to fix them.

MUST-FIX BEFORE BUILD:
1. [DEFECT 2] Architectural mismatch on stance coherence. The plan proposes routing stance reversals through the line-level scoped reroll (`_otr_reroll.py`). Line-level rerolls cannot fix outline-level plot reversals; if the beat dictates Manfred leaks the research, the line writer must write it. Rerolling the line will just thrash and fail. 
   - Fix: Move the stance-coherence check entirely to the outline/beat critic stage. Do not add it to the line composer.
2. [DEFECT 1] Catastrophic false-positive risk in the deterministic scrub. The plan proposes putting a semantic detector (distinguishing 3rd-person physical action from spoken narration) into the deterministic freeze floor (`_otr_ledger_scrub._strip_stage_directions`). You cannot safely regex bare, unquoted trailing/embedded actions without NLP; it will inevitably strip legitimate dialogue and break the audio byte-identical invariant.
   - Fix: Restrict the deterministic freeze floor scrub strictly to structural quote-boundary anomalies (e.g., stripping text that falls entirely outside matched quotes). Relegate all semantic detection of bare actions to the LLM reroll hint (`_otr_line_composer.py`), where the backstop simply keeps the draft if it fails.
3. [DEFECT 3] Pipeline-crashing lever. The plan proposes adding a "role<->char_id CONSISTENCY assert" at the role source. An assert will halt the pipeline on a weak model, violating the goal of lifting the floor (a crash is worse than a mis-stamped line).
   - Fix: Implement a deterministic coercion, not a fatal assert. In `init_lines_from_outline` and the `role_mismatch` repair guard, force `speaker_role = "character"` if `char_id` resolves to a known cast ID. 

SHOULD-FIX:
1. [DEFECT 1] Sub-pattern 3 (embedded undelimited, NO quotes at all) is impossible to safely parse with heuristics because there are no structural boundaries to anchor a regex. 
   - Fix: Explicitly exclude sub-pattern 3 from any deterministic scrubbing. Rely entirely on the LLM critic/reroll loop to catch and fix unquoted bare actions.

OPTIONAL / NICE-TO-HAVE:
- [DEFECT 2] Rather than building a new deterministic stance-tracker, simply add a "Character Stance / Motivation" axis to the existing 5 craft dimensions in `_otr_story_critic.py` (253-335).

CUT THESE (scope / over-engineering):
1. [DEFECT 4] The UN escalation proportion/setup gate. 
   - Why it is safe to cut: This is classic scope bloat. Abrupt escalation is a symptom of weak-model pacing, not a discrete mechanical failure. Building a semantic "proportion gate" or a new measurement signal will flake constantly on the strong model. Add the soft prompt directive and drop the gate/measurement entirely.

[ASSUMPTION] The plan assumes the outline generation phase reliably outputs character IDs and explicit actions that a stance-tracker or critic can actually measure before line generation begins.
[ASSUMPTION] The plan assumes the weak local model (mistral-nemo) is capable of understanding and successfully executing a targeted reroll for "stance coherence" without rewriting the entire scene context.