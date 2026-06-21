# 3D mesh look: a clean COLOR GRADIENT (not white plaster, not a meshy texture)

## Goal (operator)
The mesh_stage 3D meshes should read as a clean, pleasing COLOR GRADIENT over the silhouette --
NOT washed-out white plaster, and NOT a busy "meshy 3D texture"/photo-decal. A simple, deterministic,
good-looking colored ramp on the sculpted form. Operator wants 1-2 OTHER ideas before we pick.

## What we tried (real, grounded vs `scripts/otr_mesh_stage_blender.py`)
- v1.0: portrait PROJECTION (the flat photo decal) -> looked like a 2D photo floating on gray -> CUT.
- v1.1: WORKBENCH `light=MATCAP` + per-vertex gradient -> MATCAP renders its own baked clay material and
  IGNORES the vertex albedo -> uniform WHITE MARBLE.
- v1.2: WORKBENCH `light=STUDIO` + `color_type=VERTEX` + per-vertex vertical gradient
  (`gradient_color`: top ~(0.58,0.64,0.80), bottom ~(0.09,0.11,0.20)) + `show_cavity=True`.
  RESULT (operator look-QA on "The Green Rail", leg_0001 mesh bookends): STILL reads white/plaster --
  (a) the ramp colors are DESATURATED blue-GRAY, and (b) STUDIO's specular highlights BLOW OUT the lit
  (top) side toward white. So neither the color nor the gradient reads.

## Hard constraints
- WORKBENCH only (Cycles = the deferred v1.5 tier). Render is the headless Blender stage; the GLB stays
  GEOMETRY-ONLY on disk -- color is a PER-RENDER attribute/setting (the existing `otr_proj` vertex-color
  seam + `_configure_render`). Deterministic (no per-render variance). Headless `--background` safe (no
  `bpy.ops` needing UI context -- set on data). UTF-8 no BOM, ASCII, SFW. Cheap (no new model/dep).
- Keep the smooth form + the bounded turntable arc; per-poly `use_smooth` already on.

## My proposed direction (the baseline to beat)
WORKBENCH `light=FLAT` (shows the PURE vertex-color albedo, no specular wash) + a SATURATED two-color
vertical ramp + `show_cavity=True` (subtle form definition from cavity AO, no specular). FLAT means the
exact colors show -> a clean duotone gradient over the silhouette.

## Questions for the panel (1-2 OTHER ideas, each implementable in the WORKBENCH stage)
- Q1: Is FLAT + saturated vertex ramp + cavity the most reliable clean-color-gradient on a DENSE
  Hunyuan mesh, or is there a better WORKBENCH lever?
- Q2: A CUSTOM MATCAP that IS a gradient -- generate/ship a small 2-tone vertical-gradient matcap image
  and set `studio_light` to it (so MATCAP gives the gradient + soft shading in one, no specular blowout).
  Is this cleaner than FLAT+vertex? (it shades by VIEW normal -> a "studio gradient" look.)
- Q3: NORMAL/HEIGHT-mapped color: map the gradient to the surface NORMAL (rim->core) or world-Z with a
  controlled, low-specular STUDIO (dial `studiolight_intensity` down / specular off) so the form reads
  WITHOUT the white blowout. Can WORKBENCH STUDIO specular be disabled (e.g. `shading.show_specular`,
  `studiolight_intensity`)?
- Q4: WORKBENCH `color_type='OBJECT'` or `'SINGLE'` + a gradient-baked vertex color vs `'VERTEX'` --
  which is least likely to wash out?
- Q5: Any cheap deterministic touch that lifts it from "flat ramp" to "pleasing" without becoming a busy
  texture (a subtle rim/fresnel tint, a 3-stop ramp, a duotone + cavity)?
- Q6 [anti-goal]: keep it a COLOR GRADIENT, never a per-vertex photo/noise texture (no "meshy" look).
