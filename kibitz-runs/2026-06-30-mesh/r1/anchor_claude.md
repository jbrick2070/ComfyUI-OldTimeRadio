# Claude anchor -- r1 (mesh_stage MIN-ACCEPT, arc)

Grounded vs eng_mesh_stage.py.

## CONFIRMED
- Blender turntable is driven by `build_blender_cmd(blender_exe, glb, out_dir, frames, width, height, ...)`
  -> a matcap orbit, camera-motion-only (L293). The vertical framing lever is EITHER the Blender stage
  script's camera distance/target (the "turntable-orbit camera preset") OR the compositor fit -- there IS a
  `"crop": "center"` hint at L468. So "more headroom / subject centered not full-height" has two candidate
  seams: (a) pull the Blender camera back / raise target, or (b) change the composite fit-mode/scale.
- E-4 frame-dir contract is strict (exact N frames, canvas dims, straight-alpha RGBA) -- any framing change
  must keep the SAME delivered frame count + canvas (don't break validate_frame_dir).

## MUST-FIX (arc)
1. **Headroom:** ground which seam actually controls vertical fill BEFORE coding -- the Blender camera
   preset is the RIGHT layer (a compositor crop just zooms/pads a full-height render and loses detail).
   Recommend: adjust the Blender orbit camera distance/elevation/target for centered-with-headroom framing;
   use the composite fit only as a secondary nudge.
2. **Music-bookend = a 3D RADIO subject, not a body.** Grounding TODO: the mesh_fodder/subject is resolved
   per beat in `render_driver.build_request_from_shot` (`_requires_fodder`/`_mesh_subject_id`); the
   music-open beat (char_id "") currently meshes a generic story object. Route it to a radio subject
   (radio prompt or a canned radio mesh) -- shared "radio IS the host" theme with HuMo + viz. Keep the
   LOUD missing-fodder guard (no silent env-mesh).

## SCOPE (operator: r1 only, don't over-spend)
- Fold ONLY the clear wins. The optional deeper quality items (mesh cleanliness / texture / lighting /
  turntable speed / background plate) stay r1-level notes unless something big surfaces -- do NOT spin r2-r4.
- Trellis-as-alt-backend + the WorldMirror-vs-object-mesh 3D-path decision stay operator-gated; these
  tweaks improve the SHIPPED hy3d path regardless.
