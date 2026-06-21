<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core strategy (zero JSON changes for v1) is sound, but the widget math is wrong and the F10 state mechanism introduces race conditions.

MUST-FIX BEFORE BUILD:
1. [Node 1 current widget map] Off-by-one math error. Indices `[0]` through `[22]` constitute 23 entries, not 24. The next append point is index 23. Appending at index 24 will create an invalid sparse array or cause a ComfyUI positional mapping failure. Fix: Change the documented append point to 23.
2. [Per-fix wiring impact - F10] Concurrency/State risk. Using a "local JSON file" for the anti-repeat list violates ComfyUI's stateless execution model. Concurrent batch generations will clobber this file, causing race conditions and I/O locks. Fix: Store the anti-repeat window in memory (e.g., a class-level variable on `OTR_LedgerScriptWriter`) or use ComfyUI's standard execution caching. Do not use a raw local disk file.
3. [Open questions - Q5 / F8] Schema contract violation. You correctly identified the risk in Q5. If F8 (arc-shape variety) changes the number of acts/phases, it alters the `script_json` payload. [ASSUMPTION] `OTR_SceneSequencer` (Node 3) iterates over this JSON. If Node 3 hardcodes expectations for a 3-act structure or a fixed number of announcer beats, it will crash. Fix: Audit `OTR_SceneSequencer`'s parsing logic to ensure it dynamically handles `len(acts)` BEFORE allowing F8 to output non-standard shapes.

SHOULD-FIX:
1. [IF a widget append becomes necessary] Positional mapping safety. The document says to add the matching widget to "optional" in `INPUT_TYPES`. [ASSUMPTION] In ComfyUI, positional `widgets_values` in the JSON map strictly to the order of keys in the `"required"` dictionary of `INPUT_TYPES`. Adding it to `"optional"` may break the positional index mapping. Fix: Append new widgets to the very end of the `"required"` dictionary in `INPUT_TYPES`.
2. [Node 1 current widget map] Widget 7 is `act_count` ("auto"). If F8 dynamically picks an arc shape that dictates a specific act count, Node 1 must overwrite the "auto" value in the resulting `script_json` so downstream render nodes know the final resolved count.

OPTIONAL / NICE-TO-HAVE:
- Answer to Q1: Confirmed. F1-F7 and F9 are purely internal prompt/logic changes. As long as they do not remove existing keys from `script_json` that downstream nodes expect, they require zero wiring changes.
- Answer to Q2: Auto-pick (no widget) is highly recommended for v1 to avoid JSON schema drift. 
- Answer to Q4: Additive keys in `meta.*` are generally safe in Python/JSON, provided downstream nodes use `.get('key')` rather than strict schema validation.

CUT THESE (over-engineering):
1. [Per-fix wiring impact - F8/F10] Cut the entire concept of exposing these as widgets for v1. The document already recommends keeping them internal; commit to this. It completely eliminates the risk of widget positional drift (BUG-LOCAL-097) for this release cycle.