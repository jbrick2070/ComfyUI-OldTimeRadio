# r2 ANCHOR REVIEW (Claude, code-grounded) -- coding plan / implementability

VERDICT: implementable, but three coding-level gotchas must be in the plan before a coder starts.

## MUST-FIX (implementability)

1. S-E widget removal = POSITIONAL widget drift. `allow_auto_fallback` is an INPUT on
   `OTR_VideoDirector` (node 87, `otr_video_director.py:216`, serialized at :342). DELETING a widget
   mid-list shifts every later `widgets_values` entry in the saved node-87 graph (BUG-LOCAL-097). So
   either (a) DELETE it AND fix node-87 `widgets_values` in `otr_scifi_16gb_full.json` in the SAME
   change + re-run `OTR_WorkflowValidator`, or (b) DEPRECATE-in-place (keep the slot, ignore the value,
   hide/relabel) to avoid the shift. Recommend (b) for the widget, (a) only if a clean re-baseline is
   done. The plan must say which.

2. S-F injection = API-patch (A), NOT a new node (B). A new fixture-loader node is itself a positional-
   widget + JSON-wiring change (the thing we want to avoid in a TEST harness). The clean path: at the
   API layer, set node 91/92 inputs to the baked fixture values so the writer (node 1) + audio nodes
   are ORPHANED (nothing downstream consumes them -> ComfyUI does not execute them). Verify the cache
   key: the fixture values must hash differently per engine-under-test so a swap re-renders (ties to the
   IS_CHANGED profile rule). No production-JSON edit.

3. S-A fill belongs in the COMPOSITE, asserted by frame count. Put the loop/ping-pong where the hold is
   (`otr_silent_composite.py` ~395-411, replace `tpad=clone`), not per-engine (covers every engine at
   once). The legibility guard adds a per-clip ffmpeg/sharpness probe -- keep it on the SILENT video
   only (never touch the master audio path) so `test_audio_byte_identical` stays green. Acceptance reads
   the delivered frame count from the manifest.

## SHOULD-FIX
- S-D is a small localized LEVER-1 unwrap fix -- low risk; just add a gemma-shaped regression case.
- BUG-411 is ~done: FluxGuidance (flux_gen1), cinematic grade tail (`otr_meta_brief_image_prompt.py:535`),
  radio broadcast-distress tail (:805), portrait STYLE_ANCHOR (:92) all PRESENT. Only the bookend seed
  4242 is unverified. DEMOTE BUG-411 to a look-QA verify, not a coding sprint.
- S-E deletion blast radius (coding): `FLOOR_NAMES`/`UNIVERSAL_FLOOR`/`SYNTH_FALLBACKS` referenced at
  `render_driver.py:46-58,146,157-159,2282` + `eng_character_3d.py:55,257,326` + the fallback tests.
  Delete refs -> constants -> tests in one chunk so imports never dangle.

## GROUNDED LABELS
- CONFIRMED: BUG-411 levers present (meta_brief lines above); allow_auto_fallback is a node-87 widget;
  composite hold at otr_silent_composite ~395-411.
- VERIFY-AT-BUILD: seed 4242 location; exact node 91/92 input names in the JSON; whether deprecating vs
  deleting the widget is preferred by the operator's re-baseline appetite.
