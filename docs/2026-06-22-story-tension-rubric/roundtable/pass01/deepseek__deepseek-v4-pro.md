<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes. The vision of wiring a tension gradient and refining the critic’s flatness rubric is coherent, but the design lacks a concrete mechanism for exposing the new `beat_tension` target to the critic, and the proposed `failed_dimension` enum plus reroll parser change adds complexity that duplicates the critic’s existing actionable hint.

MUST-FIX BEFORE BUILD:
1. [STEP 5 & STEP 6 coupling / Missing piece] The document states the critic should judge whether a line “moves toward its target tension” (STEP 5 rubric), but it does not specify how the beat_tension target will be made visible to the critic. Today the critic sees ledger lines rendered by `_render_lines_for_doctor`, which does not include any tension target. The grounding confirms `beat_tension` is not present in the ledger line records. Fix: add a concrete plan to expose `beat_tension` to the critic – e.g., stamp a `beat_tension_target` field into the line’s `meta` and render it in the critic’s per-line view (as a simple text line such as “Tension target: 3/5”). Without this, the rubric’s reference to target tension cannot be applied reliably.

2. [STEP 5 scope / over‑engineering] Adding a `failed_dimension` enum to `FlatLine` and updating `_otr_reroll.py`’s hint consumer to generate a dimension‑specific REVISE instruction is unnecessary overhead. The critic already provides a concrete, actionable hint in `RerollTarget.hint`, and the reroll threads that string verbatim to the composer. Parsing a dimension to then re‑craft an instruction duplicates the critic’s judgement. Fix: cut the reroll consumer change. The critic may include the dimension as an optional annotation field for logging/metrics, but the reroll should continue to use the critic’s existing hint directly; no mapping logic should be added.

SHOULD-FIX:
3. [STEP 6 design] The document leaves the tension‑curve shape (phase‑band, smooth ramp, phase‑index+ramp) and peak‑placement open. To avoid implementation thrash, the design must settle on one deterministic rule before build (e.g., a simple phase‑index plus within‑phase linear ramp, peak at the start of the resolution phase, no easing on the last beats). Fix: choose a single shape and document the rationale; cut the alternative‑shape discussion from the final spec.

4. [STEP 5 rubric] The 5‑dimension rubric is described only informally. The final design should include the exact prompt language that the critic will use, showing how it integrates with the existing Section 3, so that the reviewer can verify consistency with the rest of the critic prompt. Fix: provide the concrete critic‑prompt text for the rubric, including how it references beat_tension (if adopted) and the expected output format.

OPTIONAL / NICE‑TO‑HAVE:
- Add a `tension_target` field to the ledger line meta (as allowed by the meta‑only rule) and render it in the critic’s view alongside the existing fields. This keeps the wire format frozen while making the target visible for judging.
- Include a short rationale for why a deterministic 1‑to‑5 tension signal is safe for a 13‑line episode without over‑constraining.

CUT THESE (scope / over‑engineering):
- The dimension‑to‑reroll‑hint parsing and mapping logic in `_otr_reroll.py`. Safe to cut because the critic’s hint string is already dimension‑specific; the reroll can use it directly.
- Any explicit SceneArcContext addition (not proposed in the doc, but the grounding confirms it’s not needed – keep it out).