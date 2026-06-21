<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Plan leaves Q1-Q6 unresolved, hardcodes incompatible STUDIO+desat gradient in _configure_render + gradient_color, and never implements the claimed FLAT path.

MUST-FIX BEFORE BUILD:
1. [My proposed direction] FLAT + saturated ramp + cavity is stated as baseline but _configure_render unconditionally does `shading.light = "STUDIO"` (and `color_type = "VERTEX"` only when _has_vertex_colors); replace with `if args.surface == "gradient": shading.light = "FLAT"` (or equivalent) else the existing STUDIO block, and add the saturated ramp constants.
2. [What we tried v1.2] GRADIENT_TOP=(0.58,0.64,0.80) / GRADIENT_BOTTOM=(0.09,0.11,0.20) are the exact desaturated values called out as failing; update gradient_color (and its callers) to saturated equivalents before any FLAT render path can satisfy the goal.
3. [Hard constraints] "color is a PER-RENDER attribute/setting (the existing `otr_proj` vertex-color seam + `_configure_render`)" is violated by any Q2 custom MATCAP (requires shipping + loading a .exr/png studio_light outside the otr_proj path and outside the geometry-only GLB rule); remove or explicitly scope that option to a later tier.
4. [Q3] "Can WORKBENCH STUDIO specular be disabled (e.g. `shading.show_specular`, `studiolight_intensity`)" has no supporting code in _configure_render; either delete the question or add the exact attribute sets (with try/except for API variance) and verify they are headless-safe.
5. [Hard constraints + _configure_render] All surface modes must remain `--background` safe with zero bpy.ops after import; any Q2/Q5 rim/fresnel or 3-stop change that would require a ShaderNode tree is banned and must be rejected in the plan.

SHOULD-FIX:
1. [Q1 + _paint_gradient_onto_meshes] Plan never states whether the vertical ramp stays world-Z after _normalize_meshes or switches to view-normal/rim; add the exact choice and the corresponding co.z vs. normal math so the dense Hunyuan case can be tested.
2. [parse_stage_args + _configure_render] `--surface gradient` already exists but forces the broken STUDIO path and ignores show_cavity under FLAT; make surface drive both the paint function and the shading block so the state machine documented in main() is consistent.
3. [_has_vertex_colors] Returns True for any color_attributes (including legacy ones); after switching to FLAT this must be narrowed to only otr_proj or the SINGLE fallback will be unreachable.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line `shading.show_cavity = True` guard inside the new FLAT branch (already present for STUDIO).
- Emit the exact light/color_type values to stdout on render so operator QA can confirm which path ran.

CUT THESE (over-engineering):
1. Q2 custom MATCAP image: requires new asset, load path, determinism proof, and extra file in the portable Blender tree; safe to cut because FLAT+vertex already satisfies "pure albedo, no specular" with zero new files.
2. Q4 OBJECT/SINGLE vs VERTEX comparison: already answered by the otr_proj seam and _activate_render_color; adds no new capability.
3. Q5 "subtle rim/fresnel tint" or 3-stop ramp: directly contradicts the "simple, deterministic, good-looking colored ramp" and "never a per-vertex photo/noise texture" constraints; any extra stops or view-dependent tint would need per-frame normal math not present in gradient_color.