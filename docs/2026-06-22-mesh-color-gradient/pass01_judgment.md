# Mesh color gradient -- pass01 judgment (Claude = judge)

Panel: Gemini-3.1-pro + Grok-4.3 (GPT-5.5 FAILED -- empty, reasoning-budget burn). Grounded vs
`scripts/otr_mesh_stage_blender.py`. ~$0.05.

## The split + my call
- Grok: implement FLAT + saturated ramp + cavity (my proposed baseline).
- Gemini: FLAT is WRONG -- it ignores surface normals -> ZERO diffuse shading -> the 3D form collapses
  to a flat 2D colored silhouette (only cavity lines remain). KEEP STUDIO for the diffuse 3D form, but
  DISABLE the SPECULAR that blew v1.2 white.
- JUDGE -> **Gemini.** We WANT the mesh to read as a 3D sculpt (just colored, not white); FLAT throws
  the 3D away. The blowout was the SPECULAR, not STUDIO itself.

## CONVERGED FIX (both agree on saturation; Gemini's technique)
1. **Keep `light=STUDIO`** (diffuse directional shading -> the 3D form reads). NOT FLAT.
2. **`shading.show_specular_highlight = False`** (try/except like `show_cavity`, headless-safe) ->
   kills the white blowout that made the lit side plaster.
3. **Keep `show_cavity = True`** (subtle form definition).
4. **SATURATE the ramp** -- v1.2's (0.58,0.64,0.80)/(0.09,0.11,0.20) blue-GRAY is the other half of the
   "still white" problem. Use a saturated duotone. Default = teal -> navy (on-brand for SIGNAL LOST);
   operator can swap to amber/green/bronze by changing the two constants.
5. World-Z ramp stays (deterministic, after `_normalize_meshes`). EMIT the light/color_type/colors to
   stdout so operator QA can confirm which path rendered.

## CUT (panel)
- Custom MATCAP gradient (Gemini): a matcap maps by VIEW normal -> the gradient SWIMS as the turntable
  rotates (up-facing-to-camera always gets the top colour, regardless of real height); also a new
  asset + load path outside the geometry-only/otr_proj seam. CUT.
- FLAT (Gemini): flattens the form. CUT as the default.
- Rim/fresnel/3-stop ramp, OBJECT/SINGLE comparison (Grok): need shader nodes / add nothing; the
  "simple colored ramp, never a busy texture" goal forbids view-dependent tints. CUT.

## Build (one small stage-script chunk)
`_configure_render` WORKBENCH block: STUDIO + `show_specular_highlight=False` + `show_cavity=True`;
`GRADIENT_TOP/BOTTOM` -> saturated teal/navy; stdout the chosen light/colors. CPU-Blender verify the
render is COLORED (not white) AND the form reads (top brighter than bottom, sides shaded). Suite +
Bug Bible. Applies to the next mesh render (the stage script is re-read each Blender spawn -> no server
restart).
