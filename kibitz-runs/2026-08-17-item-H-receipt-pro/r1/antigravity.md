VERDICT: no. The document's core architectural dilemma rests on a demonstrably false premise that moving a telemetry label requires reordering cache and crypto logic.

MUST-FIX BEFORE BUILD:
1. [What the panel is asked to break] (Question 2) / [The two options, now asymmetric] False premise on reordering. The document claims making the label engine-aware is a "production reordering" that might not be safe amidst "cache keys, seeds and the banana transform." This is demonstrably false. `_neg_source` is a write-only telemetry variable bound at 1166 and only used in ledger dicts at 1413 and 1608. It is completely decoupled from `prompt_hash` and `_banana`. Concrete fix: Move the `_neg_source` assignment below `resolve_engine_for_role` (line 1225). You do not need to reorder any crypto or cache logic.
2. [The two options, now asymmetric] Missing concept: The Third Option. Option A accepts a blind ledger; Option B illegally alters a recipe. By realizing that `_neg_source` can be safely moved after engine resolution, a third option unlocks: let the dispatcher check the resolved `engine_id` (or query the engine) and accurately log `engine_hygiene` if the engine actually applies a floor (e.g., `z_image_turbo`), or `none` if it does not. Concrete fix: Implement this third option. Keep the recipes untouched, but make the ledger honest and engine-aware.

SHOULD-FIX:
1. [What the panel is asked to break] Question 1 asks if renaming the ledger value is a contract change that downstream consumers can feel. The document's own data in [Blast radius] proves there are zero historical on-disk ledgers carrying this field. Concrete fix: State definitively that Option A's rename is safe. Consumers reading fields cannot break on a value that has never existed in production.

OPTIONAL / NICE-TO-HAVE:
- Add a dedicated field `engine_effective_negative` to the ledger, clearly separating the dispatcher's composed prompt from the engine's final conditioning. [ASSUMPTION] Assuming engines can easily return or expose their applied negative post-generation.

CUT THESE (scope / over-engineering):
1. Option B (giving `lumina_image` a hygiene floor). Safe to cut because altering visual conditioning on a live engine to satisfy a broken upstream telemetry log is dangerous scope creep and directly violates the stated invariant ("The recipes are not on the table").
