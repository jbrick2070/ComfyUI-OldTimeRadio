# QA: remove the deprecated `allow_auto_fallback` widget (clean-UI directive)

Operator directive 2026-07-03: clean json / clean workflow / NO deprecated cruft.
`allow_auto_fallback` on OTR_VideoDirector was deprecated (NO FALLBACKS, 2026-07-02):
ignored at runtime, always emitted False, kept only as a positional tombstone. Removed
it ENTIRELY (code + node schema + the production workflow JSON + tests).

## What was changed (verify each against the real files)
1. `nodes/otr_video_director.py`:
   - INPUT_TYPES: removed the `"allow_auto_fallback"` BOOLEAN widget block.
   - `direct()` signature: removed the `allow_auto_fallback` positional param.
   - Removed the `if bool(allow_auto_fallback): log.warning(...)` deprecation block.
   - Removed `"allow_auto_fallback": False` from the emitted policy dict.
   - Module docstring: dropped the stale "fallback policy" mention.
2. `nodes/_otr_video_engines/schemas.py`: removed `Policy.allow_auto_fallback` field
   (nothing set or read it; render_driver never touched it).
3. `workflows/otr_scifi_16gb_full.json` (node 87 OTR_VideoDirector):
   - widgets_values: removed slot 11 (the `false`) -> 15 -> 14 items.
   - inputs: removed the `allow_auto_fallback` widget-input SOCKET (was index 12,
     link=None). The only linked input is `gate_in` (slot 0, link 269) -> unaffected;
     the shifted inputs 13-15 (episode_duration_target/custom_models_json/
     character_video_model) all had link=None, so NO link dst_slot changed.
4. Tests updated: `test_video_platform_aseam.py` (required-keys vector 12->11 + two
   direct() calls), `test_route_a_14b_promotion.py`, `test_still_aspect_and_labels.py`
   (dropped the kwarg), `test_workflow_live_passes_validator.py` (pin 15->14, wv87[14]
   ->wv87[13]).

## Caught mid-change (already fixed)
The OTR_WorkflowValidator flagged a "rogue socket `allow_auto_fallback` not declared by
INPUT_TYPES()" -- because litegraph also lists each widget in the node `inputs` array.
Removing only the widgets_values slot was insufficient; the input SOCKET had to go too.
Both are now removed; validator passes.

## QA questions for the panel (ground against the real code)
1. Any REMAINING reference to `allow_auto_fallback` anywhere -- code, tests, other
   workflow JSONs that are loaded, docs that assert behavior, the b7 forbidden-symbol
   sweep, or a consumer that reads `policy["allow_auto_fallback"]`?
2. Is the node-87 JSON edit link-safe? Confirm no link has dst=87 with a dst_slot that
   shifted (only gate_in slot 0 is linked; inputs after the removed socket had no links).
   Confirm widgets_values (14) still positionally maps to the widget-inputs (BUG-LOCAL-097).
3. Does removing `Policy.allow_auto_fallback` (a `_Forbid` model) break parsing of any
   still-serialized request/fixture that carries the key? Is there any such fixture?
4. Does the production workflow still load + pass the full validator (widget-count audit,
   link referential integrity, positional-widget drift)?
5. Anything else that makes this NOT a clean, complete removal.

Invariants: NO fallbacks (this removal must not resurrect anything); workflow-JSON edited
in the SAME change as the code; audio spine untouched; UTF-8 no BOM; suite + Bug Bible + B7
green + push per green chunk.
