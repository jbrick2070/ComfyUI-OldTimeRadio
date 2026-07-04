# Plan: 3 video slots together (announcer / music / CHARACTER), retire other_beats

**Operator directive 2026-07-03 (DECISIVE, overrides the rip-sfx-broll "keep
other_beats as migration fallback" + codex's cautious keep):** there are exactly THREE
video dropdowns -- announcer, music, character -- and they must sit TOGETHER in the node.
`character_video_model` currently sits alone at the BOTTOM (after custom_models_json) and
inherits from a legacy `other_beats_video_model` slot. Consolidate: character joins
announcer+music as the 3rd video slot (position 3); `other_beats_video_model` +
the `USE_OTHER_BEATS` inherit sentinel are REMOVED. NO back-compat shim (cleanbreak).

Panel: review the CODING + WIRING. This is a positional-widget reorder (BUG-LOCAL-097) +
a role-resolution change + a workflow-JSON same-change -- exactly where things silently
break. Ground every claim; cite file:line.

## Target UI (video models, in order)
`announcer_video_model`, `music_video_model`, `character_video_model` -- three siblings,
each the FULL video registry combo (no `(use Other Beats default)` sentinel). Image trio
(announcer/music/other_beats_image) is a SEPARATE structure -- see "Image side" below.

## Code changes (grounded)
1. `nodes/_otr_shared/role_slots.py`:
   - `ROLE_TO_VIDEO_SLOT` already maps character_video -> character_video_model (:45-49) -- KEEP.
   - DELETE `LEGACY_OTHER_BEATS_SLOT` (:33), `_OTHER_BEATS_ROLES` (:38-40), the
     `LEGACY_OTHER_BEATS_SLOT` row in `VIDEO_SLOT_ROLES` (:59), and the empty-slot fallback
     branch in `engine_id_for_role` (:121-122). character_video now resolves ONLY via
     character_video_model; empty -> "" (caller handles, no silent lane). Update the module
     docstring (:20-24) -- remove the migration-lane note.
2. `nodes/otr_video_director.py`:
   - INPUT_TYPES required: REMOVE `other_beats_video_model` (:215); INSERT
     `character_video_model` as the 3rd video slot (full `_video_model_combo()`), so the
     required order is announcer_video, music_video, character_video, announcer_image,
     music_image, other_beats_image, fps, canvas_w, canvas_h, seed_mode, request_seed.
   - REMOVE the optional `character_video_model` block (:258-262) + `USE_OTHER_BEATS` (:142),
     `_per_role_video_combo` (:162-166), and the sentinel handling in `direct()` (:304-308).
   - `direct()` signature: drop `other_beats_video_model`; make `character_video_model` a
     required positional in the video trio position; resolve it directly via
     `_engine_id_from_pick` (no USE_OTHER_BEATS branch).
   - `_role_aspects` / `_role_talking` already resolve via role_slots -- confirm they still
     work with the fallback gone.
3. `nodes/_otr_shared/role_slots.py` `PER_ROLE_VIDEO_SLOTS` (:65-69) already lists the 3
   correct slots -- KEEP; it becomes the single canonical widget order for the applier.
4. `nodes/_otr_workflow_apply.py:140`: the widget-order list has `other_beats_video_model`
   -> replace with `character_video_model` in position 3 (match the new INPUT_TYPES order).
5. `nodes/otr_shot_lock.py:704`: uses role_slots for the engine pick -- confirm no direct
   other_beats reference remains after role_slots changes.

## Workflow JSON (node 87, SAME change -- the risky part)
Current node 87 (post episode_duration_target removal) widgets_values (13):
`[announcer_v(viz_green), music_v(viz_green), other_beats_v(viz_green), announcer_i(flux_gen1),
music_i(flux_gen1), other_beats_i(flux_gen1), fps(25), canvas_w(832), canvas_h(480),
seed_mode(request_hash), request_seed(42), custom_models_json({}), character_v(humo_14B_169)]`
Target order (video trio together, character promoted, other_beats_video removed):
`[announcer_v(viz_green), music_v(viz_green), character_v(humo_14B_169), announcer_i(flux_gen1),
music_i(flux_gen1), other_beats_i(flux_gen1), fps(25), canvas_w(832), canvas_h(480),
seed_mode(request_hash), request_seed(42), custom_models_json({})]` -> 12 widgets.
Mirror the `inputs` array: remove the `other_beats_video_model` widget-socket, move the
`character_video_model` widget-socket into position 3, keep `gate_in`(slot0, link 269) intact.
Re-validate: OTR_WorkflowValidator (rogue-socket + widget-count audit), JSON round-trip,
link referential integrity (no dst_slot on node 87 except gate_in has a live link).

## Tests to update
- `tests/test_rip_sfx_broll_guard.py`: it currently ASSERTS `other_beats_video_model in req`
  (:149-150) and pins the widget vector -- REVERSE to assert other_beats_video_model NOT in
  req + character_video_model IN req; update the widget-count (13->12) + character index.
- `tests/test_video_platform_aseam.py`: required-keys vector (drops other_beats_video_model,
  adds character_video_model at pos 3).
- `tests/test_route_a_14b_promotion.py` + `test_still_aspect_and_labels.py`: they pass
  other_beats_video_model + character_video_model kwargs to direct() -- update to the new
  signature.
- `tests/test_workflow_live_passes_validator.py`: widget-count pin 13->12 + character index.
- role_slots tests + any migration-fallback test that asserts the legacy fallback -- update
  to the no-fallback reality.

## OPEN for the panel
1. **Image side symmetry:** the image trio is announcer_image / music_image /
   `other_beats_image_model`, and the image dispatcher (`otr_image_gen_dispatcher.py:162,172`)
   uses other_beats_image_model as the character-image fallback. Should this pass ALSO rename
   the image "other_beats" slot to `character_image_model` for a fully consistent node, or is
   that a separate follow-up (more dispatcher surface)? Recommend: do it in the SAME pass for
   symmetry IF low-risk; else defer with a clear ticket.
2. **default value:** character_video_model's shipped default is humo_14B_169 (the Route-A
   14B lane). Keep that as the position-3 default? (yes -- it is the current live pin.)
3. Any consumer of the removed empty-slot fallback in real/old ledgers that would now resolve
   to "" and fail LOUD -- is that the desired cleanbreak behavior (old graphs must re-pick)?

## Invariants
NO back-compat shim; node + JSON edited in the SAME change; positional-widget audit after;
validator + widget-count + link integrity pass; audio spine untouched; suite + Bug Bible + B7
green; push per green chunk; character default stays humo_14B_169.
