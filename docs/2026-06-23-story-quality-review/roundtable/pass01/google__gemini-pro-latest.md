<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan explicitly violates its own primary constraint in the first proposed lever.

MUST-FIX BEFORE BUILD:
1. [L1 - Crisis-noun denylist] Contradicts Hard Constraint #1 ("NO new flag-and-reroll critic/QA gate"). L1 proposes: "on exceed, regenerate THAT beat's intent once". A regeneration loop is a reroll gate, shifting it to the outline phase does not change its nature. Fix: Make the denylist a hard generation constraint (e.g., logit bias/grammar) or strictly enforce it via prompt without a retry loop.
2. [L2 - Phase = dramatic FUNCTION] The core finding states models ignore soft prompts (e.g., "RAISE THE STAKE"). L2 proposes adding "required beat SLOTS" like "climax" to the Python skeleton. Naming a slot "climax" in Python does not force a weak model to write a climax if it already ignores explicit instructions. Fix: Define the mechanical enforcement for these slots (e.g., forcing terminal state variables or hard-coding the climax action from the brief).

SHOULD-FIX:
1. [L3 - Action/dialogue split] [ASSUMPTION] You assume a weak 12B local model can reliably output structured `{internal_action, spoken_dialogue}` without syntax/JSON errors. Weak models frequently hallucinate or break strict output schemas. Fix: Specify a dead-simple, robust fallback parser (e.g., regex stripping anything in brackets) rather than relying on strict key-value compliance.
2. [L5 - Writer default = gemma-12b] [ASSUMPTION] You assume Gemma-12b's success on 3 episodes generalizes, without addressing *why* it hit the `too_many_edits` abort in the first place (likely formatting instability or hallucinated tags). Fix: Investigate the root cause of the `too_many_edits` abort before blindly promoting the model.

OPTIONAL / NICE-TO-HAVE:
- [L4 - Deterministic transcript sanitizer] Ensure the regex for unbalanced quotes doesn't aggressively swallow valid dialogue containing apostrophes or measurements.

CUT THESE (scope / over-engineering):
1. [L6 - best-of-N line selection] Cut it. Multiplying local inference cost by N for a "polish" lever that explicitly does not fix the root structural sameness is a waste of compute and directly contradicts the need to run efficiently on local hardware.

[ASSUMPTION] [L1] Assumes `allowed_things` derived from the brief actually contains actionable dramatic conflict (e.g., "injunction") rather than just static background nouns (e.g., "desk", "paper"). If the extraction is weak, the palette will be weak.