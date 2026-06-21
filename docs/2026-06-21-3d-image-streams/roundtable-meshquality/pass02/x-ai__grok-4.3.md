<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The Chunk B surface/gradient/smooth implementation described in the plan is absent from the grounded otr_mesh_stage_blender.py + eng_mesh_stage.py (no argparse, no _paint_gradient_onto_meshes, no _smooth_mesh_normals, no state machine in main(), no surface handling in build_blender_cmd or render_clip); plan and code contradict on defaults and paths.

MUST-FIX BEFORE BUILD:
1. [Chunk B + otr_mesh_stage_blender.py:parse_stage_args] argparse lacks --surface entirely (and the LOUD error for mode=render+surface=portrait+no-portrait); add the argument + validation exactly as specified before any render path can reach gradient/portrait.
2. [Chunk B + otr_mesh_stage_blender.py:main + _configure_render] main() and _has_vertex_colors only ever set VERTEX for portrait/selftest; gradient path never creates otr_proj or calls any activation, so WORKBENCH never sees the attribute -- add _paint_gradient_onto_meshes (and shared _activate_render_color helper) + the full surface state machine.
3. [Chunk B + eng_mesh_stage.py:render_clip + build_blender_cmd] render_clip unconditionally passes portrait=still and build_blender_cmd has no surface= param or --surface append logic; change default to surface="gradient" (no portrait), gate portrait behind OTR_MESH_PROJECT_PORTRAIT=1, and wire the arg through.
4. [Chunk B + otr_mesh_stage_blender.py] _smooth_mesh_normals is missing; add the poly.use_smooth=True loop (no bpy.ops) and call it for every render-mode surface after paint.
5. [Invariants + eng_mesh_stage.py:render_clip] plan states GLB stays geometry-only and MESHER_VERSION unchanged, but current render_clip always forces the portrait path; the gradient default must be the non-portrait case to preserve the contract.

SHOULD-FIX:
1. [otr_mesh_stage_blender.py:Q4] _smooth_mesh_normals (once added) iterates mesh.polygons but never calls mesh.update() or calc_normals_split(); verify headless WORKBENCH actually shades smooth without it (or add the minimal update).
2. [otr_mesh_stage_blender.py:Q3 + _project_portrait_onto_meshes] color_attributes.new + active/render index setting re-uses the same name on every render; cache-hit GLB re-import is per-process so no collision today, but add an explicit remove of any pre-existing PROJ_ATTR_NAME before new() to guard future re-entrancy.
3. [eng_mesh_stage.py:render_clip + Q2] straight-alpha + film_transparent is set, but validate_frame_dir only checks mode=="RGBA"; add an explicit premultiplied-alpha probe on first frame for the directory-clip validator path.

OPTIONAL / NICE-TO-HAVE:
- Add a unit test for gradient_color clamping/lerp edge cases at exactly +/-0.5 (pure, already CPU-testable).
- Expose surface= as an explicit render_clip kwarg (still default gradient) for future opt-in without env var.

CUT THESE (over-engineering):
- [Chunk A question on NEG absorption] No change needed; the POS scaffold is already the only channel that reaches the engine, so the checked-in NEG is dead weight for this path and can be dropped.
- [Q6 matcap alternative] Per-vertex gradient + poly smooth is heavier than a WORKBENCH STUDIO+matcap ramp would be; safe to cut the entire gradient painting block if the visual target is only "basic ramp" (keeps selftest + portrait paths untouched).