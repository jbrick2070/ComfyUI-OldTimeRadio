# mesh_stage improvements -- DRAFT (operator 2026-06-30, soak eyeball: "meshy pretty good, one must-have")

mesh_stage (hy3d mesh -> Blender turntable) is a KEEPER. "Getting real close." Two MINIMUM acceptance
criteria before it's promotable, plus an optional light kibitz (r1 only -- operator doesn't want to
over-spend). Build with the post-soak code batch.

## MINIMUM ACCEPTANCE CRITERIA (must-haves)
1. **Opening / music bookend = a 3D RADIO, not a character body.** The music_open (and pure-music close)
   beat should mesh a VINTAGE RADIO subject, not a person. Route the music-bookend mesh_fodder to a radio
   subject (radio prompt / a canned radio mesh), consistent with the "the radio IS the host" aesthetic
   (same theme as the HuMo-improve + viz plans). Grounding: mesh_fodder is resolved per beat in
   `render_driver.build_request_from_shot` (the `_requires_fodder` / `_mesh_subject_id` path); the
   music-open beat (char_id "") currently meshes a generic story object -> point it at a radio.
2. **More HEADROOM -- 3D subject CENTERED, not full top-to-bottom body.** The turntable currently frames
   the mesh full-height (body fills the frame). Pull the Blender camera back / adjust the framing (or the
   composite fit/scale) so the subject sits CENTERED with headroom above. Likely levers: the Blender stage
   camera distance / target, or the compositor's fit-mode for the mesh clip. Confirm which in
   `eng_mesh_stage.py` + the Blender stage script.

## OPTIONAL -- r1-only kibitz (light, don't over-spend)
Run ONE kibitz round (r1, high-level) on further mesh_stage quality: mesh cleanliness (the "plaster blob"
risk from a low-feature init), texture/material, lighting, turntable speed/arc, background plate. Fold only
the clear wins; skip r2-r4 unless r1 surfaces something big. (Operator: "maybe I just do an r1.")

## CONSTRAINTS
Stay in the current VRAM budget (mesh + Blender run AFTER the reclaim barrier, never co-resident); 100%
local; LOUD on missing fodder (no silent env-mesh -- the existing missing-fodder guard); content-only where
possible, workflow-JSON only if a node/widget changes. Suite + Bug Bible + B7 green; push per chunk.

## RELATED / DEFERRED
Trellis (object-mesh) is a candidate ALTERNATIVE 3D backend the operator is musing about; the GO_FORWARD
3D-path decision (WorldMirror multi-view vs single-image object-mesh: TripoSG / Hunyuan3D / Trellis) stays
operator-gated. These mesh_stage tweaks improve the SHIPPED hy3d path regardless of that decision.
