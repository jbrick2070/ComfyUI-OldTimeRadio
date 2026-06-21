# OTR 3D Mesh-Quality v1.1 -- pass02 REVIEW FOCUS: coding + wiring + ComfyUI/Blender quirks

This is the CONVERGED pass01 plan (C cut). pass01 already hardened the architecture (cache trap,
headless context, world-space coords, surface state machine). pass02 asks the panel to make the
SPRINTS CODE-READY along three axes ONLY -- be concrete, cite the grounding files:

1. CODING CORRECTNESS of the exact functions below (signatures, edge cases, determinism, test gaps).
2. WIRING: does ANYTHING here require a change to the ComfyUI workflow JSON
   (`workflows/otr_scifi_16gb_full.json`) or a node's INPUT_TYPES/widgets? Our claim is NO -- this
   is script + content + adapter-internal only (no new node, no new widget). Confirm or refute with
   the grounding.
3. COMFYUI / HEADLESS-BLENDER QUIRKS that bite at runtime (the "final finish"): in-process
   `wrapper_bridge` execution, WORKBENCH VERTEX + per-poly smooth, color-attribute API variance,
   `--background --factory-startup` context, straight-alpha `film_transparent` handoff, the cube
   selftest gate, the GLB-stays-geometry-only contract.

## The exact implementation (review THIS)

### Chunk A (SHIPPED-pending: content-only, otr_meta_brief_image_prompt.py)
`MESH_FODDER_POS_SCAFFOLD` now forces: "simple clean unbroken silhouette, smooth solid form, plain
matte solid-colour clothing, short tight neat hair, neutral symmetrical forward stance, ...". NEG
adds: "loose flowing hair, hair wisps, fine surface detail, intricate texture, frills, thin
protrusions, jewellery". No per-object negative channel exists in the dispatcher, so NEG is
checked-in canon only (the image engine renders from POSITIVE). Question: is that an acceptable
limitation, or should the fodder POSITIVE itself absorb the key "no loose hair / smooth" cues
(since only POSITIVE reaches the engine)?

### Chunk B (Blender stage + eng_mesh_stage)
PURE (CPU-tested):
- `gradient_color(co_z, top=(0.86,0.87,0.92), bottom=(0.30,0.31,0.38))`: clamp co_z to [-0.5,0.5],
  s=co_z+0.5, lerp bottom->top, return (r,g,b) in [0,1].

BLENDER (real-Blender; exercised by the selftest + the GPU smoke, NOT pytest):
- `_paint_gradient_onto_meshes(bpy, meshes)`: new per-VERTEX `otr_proj` FLOAT_COLOR POINT-domain
  attribute; for each vert `co = obj.matrix_world @ v.co`; colour = `gradient_color(co.z)`; set it
  active + render colour (the same selection code the portrait projection uses -> extract a shared
  `_activate_render_color(mesh, ca)` helper). Returns distinct-colour count.
- `_smooth_mesh_normals(meshes)`: `poly.use_smooth = True` on every polygon of each mesh's DATA --
  NO `bpy.ops` (avoids the headless active-object/selection context crash pass01 flagged).
- `--surface {flat,gradient,portrait}` argparse, DEFAULT `flat`. `parse_stage_args` LOUD-errors if
  `mode==render and surface==portrait and not portrait`.
- `main()` render-mode state machine:
  - selftest mode: UNCHANGED (projects the in-memory portrait, non-uniformity gate intact).
  - `surface==gradient`: paint gradient; if `--portrait` also passed, WARN + ignore it.
  - `surface==portrait` OR (`surface==flat` AND `--portrait` set): project the still. The second
    arm PRESERVES the legacy omitted-`--surface` + `--portrait` behavior (project), so an old
    invocation is unchanged.
  - `surface==flat` with no portrait: paint nothing (gray matcap).
  - `_smooth_mesh_normals(meshes)` runs for ALL render-mode surfaces (after paint, before render).
- `build_blender_cmd(..., surface="")`: append `--surface <surface>` ONLY when set (so the
  selftest call + any legacy call are byte-identical).
- `eng_mesh_stage.render_clip`: default `surface="gradient"`, do NOT pass the fodder photo;
  `OTR_MESH_PROJECT_PORTRAIT=1` -> `surface="portrait"` + `portrait=still` (opt-in decal).

## Invariants (reject any "fix" that breaks one)
100% local/offline; deterministic seed-keyed; audio byte-identical (mesh/image-only); GLB stays
geometry-only on disk (MESHER_VERSION NOT bumped); WORKBENCH only (EEVEE banned, Cycles v1.5);
selftest stays green; UTF-8 no BOM; SFW; NO workflow-JSON change unless a node/widget actually
changes.

## Specific questions for the panel (code-ready gate)
- Q1 [wiring] Confirm NO `otr_scifi_16gb_full.json` change is needed (mesh_stage is already a
  registered VALIDATED engine selectable in OTR_VideoDirector; this only changes the adapter's
  Blender invocation + the image-prompt text). Any node/widget/INPUT_TYPES touch we are missing?
- Q2 [comfy] WORKBENCH `shading.color_type='VERTEX'` + `film_transparent=True`: does the per-vertex
  gradient + per-poly smooth render with correct STRAIGHT alpha (RGBA), so the downstream
  directory-clip validator (`validate_frame_dir`: exactly N RGBA frames, dims == canvas) still
  passes? Any premultiplied-alpha trap?
- Q3 [comfy] `mesh.color_attributes.new(... domain="POINT", type="FLOAT_COLOR")` + setting
  active_color / *_color_index: the projection already does this and works on the box. Any Blender
  4.5 API caveat for a SECOND attribute name reuse across cache-hit re-renders (same GLB re-imported
  fresh each render, so attrs are per-process -- confirm no stale-attr collision)?
- Q4 [coding] `_smooth_mesh_normals` via `poly.use_smooth` -- is iterating `mesh.polygons` and
  setting the flag sufficient in `--background` (no `update()`/`free_normals` needed) for WORKBENCH
  to shade smooth? Or is a `mesh.update()` / `mesh.calc_normals_split()` call required?
- Q5 [coding] Determinism: `gradient_color` is pure + seed-free; the gradient depends only on mesh
  geometry (deterministic per cached GLB). Confirm no per-render nondeterminism is introduced.
- Q6 [finish] Any cheaper/cleaner "basic gradient" that the operator means -- e.g. a WORKBENCH
  STUDIO light + a matcap that already ramps, vs per-vertex colours? Trade-offs for a turntable arc.
