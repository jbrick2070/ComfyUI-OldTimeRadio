# OTR 3D Mesh-Quality v1.1 -- plan to harden (pass00)

## Context (what shipped, what the operator now wants)
3D image streams v1.0 just shipped + GPU-verified: `mesh_stage` no longer meshes the
per-beat cinematic scene still -- it meshes a CLEAN isolated `mesh_fodder` subject over a
generated `scene_background_plate`, composited opaque source-over. The all-3D 30w smoke
rendered end-to-end to `otr/obs`.

Operator look-QA on the obs final: **background plates = excellent; meshes are now isolated
subjects (not the clay blob) -- a real win.** Two remaining quality gaps:
1. **Lumpy mesh artifacts.** Hunyuan3D reads busy detail in the fodder image (loose hair,
   intricate clothing, fine texture) as geometry -> a coral-like growth fused onto a head,
   elongated blobs. The operator wants a cleaner mesh via (a) a tighter 2D fodder prompt and
   (b) "perhaps some 3D-gen parameters".
3. **Flat photo decal + flat gray surface.** The Blender stage projects the fodder still onto
   the front vertices (single-view orthographic) and renders WORKBENCH matcap. The result reads
   as a flat 2D photo floating on a gray form (a "double-exposure" look). The operator wants a
   **simpler, better mesh texture -- "basic gradients"** instead of the photo decal + flat gray.

This is the deferred 3D v1.5 "lit/textured" tier, scoped DOWN to a cheap, deterministic,
CPU-local win: better fodder + a gradient sculpt look. NO Cycles, NO lighting rig, NO
multi-view texture bake (those stay v1.5).

## Invariants (HARD -- reject any change that breaks one)
- 100% local / offline; no new model, no paid service at render time.
- Deterministic, seed-keyed; same inputs -> same frames.
- Audio byte-identical (this is an image/mesh-only change).
- Ledger schema `l3-2026-05-14` additive-only; capability-gated routing (no engine-name checks).
- The mesh GLB stays GEOMETRY-ONLY on disk (the mesher never writes color; `MESHER_VERSION` not
  bumped) -- surface look is a per-render Blender attribute, exactly like the existing projection.
- The cube self-test (E-6) must stay green -- it gates the first real Blender use each process.
- UTF-8 no BOM; SFW; WORKBENCH only (EEVEE banned, Cycles is v1.5).
- Workflow JSON change ONLY if a node/widget actually changes; otherwise content/script-only.

## Grounded current state (verified against the real files)
- **Fodder prompt** (`nodes/otr_meta_brief_image_prompt.py`): `MESH_FODDER_POS_SCAFFOLD` =
  "single centered subject, full unoccluded three-quarter view, entire head and body clearly
  visible, plain seamless neutral mid-grey studio backdrop, even soft diffuse frontal lighting,
  no hard shadows, no props, sharp focus, full natural color". `MESH_FODDER_NEG_SCAFFOLD` =
  "busy background, multiple subjects, occlusion, hands over face, hood, dramatic shadow, cast
  shadow, cropped, scene, environment, props, text, watermark". NOTE: there is NO per-object
  negative channel in `otr_image_gen_dispatcher.py` today (it builds its row from `prompt` only);
  the NEG scaffold is checked-in canon but not consumed by the image engine. Subject built by
  `_mesh_fodder_subject` (appearance for chars / announcer figure / story object).
- **Mesh gen** (`nodes/_otr_video_engines/eng_mesh_stage.py::_build_mesh_graph`): Hunyuan3D-2mv
  native graph -- `KSampler` steps=30 (env `OTR_HY3D_STEPS`), cfg=5.0 (`OTR_HY3D_CFG`),
  sampler=euler, scheduler=normal; `EmptyLatentHunyuan3Dv2` resolution=3072; `VAEDecodeHunyuan3D`
  num_chunks=8000, octree_resolution=256; `VoxelToMesh` algorithm="surface net", threshold=0.6.
  All passed EXPLICITLY (V3 nodes backfill nothing).
- **Blender stage** (`scripts/otr_mesh_stage_blender.py`): imports GLB, bbox-normalizes, projects
  `--portrait` onto a per-VERTEX `otr_proj` color attribute via single-view (Y,Z) orthographic
  `project_uv` + `sample_image`, renders WORKBENCH matcap with `shading.color_type='VERTEX'` when
  vertex colors exist else `'SINGLE'` gray (0.78). `film_transparent=True` -> straight-alpha RGBA.
  Pure CPU-tested helpers: `clamp_arc_degrees`, `arc_keyframes`, `project_uv`, `sample_image`.
  `eng_mesh_stage.render_clip` ALWAYS passes `portrait=still` today.

## Proposed changes (the plan to harden)

### A. Tighter fodder prompt (content-only, `otr_meta_brief_image_prompt.py`)
Rewrite `MESH_FODDER_POS_SCAFFOLD` to force a clean, mesh-friendly subject: simple smooth
unbroken silhouette, plain matte solid-colour clothing, short tight neat hair, neutral
symmetrical forward A-pose, no fine surface detail. Extend `MESH_FODDER_NEG_SCAFFOLD` with the
artifact drivers: loose flowing hair, wisps, intricate texture, frills, thin protrusions.
Rationale: the lumps come from the SUBJECT, not the background (the bg is already neutral). The
deterministic subject phrase from `_mesh_fodder_subject` is preserved; only the scaffold changes.

### B. Gradient sculpt surface (`scripts/otr_mesh_stage_blender.py` + `eng_mesh_stage.py`)
Replace the flat photo decal as the DEFAULT with a basic vertical gradient vertex-colour sculpt:
- New pure `gradient_color(co_z)` -> a vertical lerp (lighter top, darker base) over normalized
  z in [-0.5, 0.5]; CPU-tested for monotonicity + [0,1] range + determinism.
- New `_paint_gradient_onto_meshes(bpy, meshes)` writing `otr_proj` from `gradient_color(co.z)`,
  active+render, reusing the EXACT vertex-colour mechanism the projection uses (so WORKBENCH
  `color_type='VERTEX'` draws it).
- New `--surface {gradient,portrait,flat}` arg (default `gradient`); render-mode branch in
  `main()`: gradient -> paint gradient; portrait (and `--portrait` set) -> project the still;
  flat -> no vertex colours (gray matcap). `build_blender_cmd` gains `surface=""` appended only
  when set (legacy invocation byte-identical).
- `eng_mesh_stage.render_clip`: default `surface="gradient"` and STOP passing the fodder photo as
  `--portrait`; keep the projection behind opt-in `OTR_MESH_PROJECT_PORTRAIT=1`.
- Add `bpy.ops.object.shade_smooth()` (smooth normals) before render -> kills the faceted look
  with ZERO geometry change. Selftest still uses projection (its non-uniformity gate is intact).

### C. 3D-gen parameters for a cleaner mesh (`eng_mesh_stage._build_mesh_graph`)
The operator's "perhaps some 3D-gen parameters". Candidate levers (need the panel's read on which
actually reduce the lumps without GPU iteration risk): `VoxelToMesh.threshold` (0.6 -> higher =
tighter iso-surface, fewer floaters, but can erode thin features); `octree_resolution` (256;
lower = smoother/blockier, higher = finer + noisier); `OTR_HY3D_STEPS`. PROPOSAL: keep gen params
as-is for v1.1 (they are GPU-proven) and lean on A+B for the visible win; expose threshold via an
env knob (`OTR_HY3D_VOXEL_THRESHOLD`, default 0.6) so it can be A/B'd on the box without a code
change. OPEN QUESTION for the panel: is a small threshold bump (0.6 -> ~0.65) a safe default, or
does it risk eroding faces/hands -> worse silhouette? Ground against the surface-net semantics.

## Build order (each its own green chunk: full suite + Bug Bible -> commit AND push to v2.0-alpha)
1. A: fodder prompt scaffolds (+ update the chunk-3 prompt-content test assertions).
2. B: gradient surface + shade-smooth + `--surface` arg + eng default flip (+ pure
   `gradient_color` CPU test + `build_blender_cmd --surface` test + real-Blender selftest stays
   green).
3. C: `OTR_HY3D_VOXEL_THRESHOLD` env knob (default 0.6 = byte-identical) (+ test).
Then ONE GPU re-smoke (all-slots mesh_stage 30w, real `otr_scifi_16gb_full.json`) -> obs final ->
operator look-QA.

## Verify-at-build / open questions for the panel
- V1: Does WORKBENCH `color_type='VERTEX'` with `shade_smooth` interpolate the per-vertex gradient
  smoothly across faces (so the gradient reads as a soft ramp, not faceted bands)?
- V2: `bpy.ops.object.shade_smooth()` needs an active object / selection context in headless
  `--background` Blender -- is the right call `mesh.shade_smooth()` on the data, or per-polygon
  `use_smooth=True`, to avoid a context error? (operator pain point: headless context ops fail.)
- V3: Dropping `--portrait` by default loses the character likeness on the mesh surface -- is the
  gradient-only sculpt the right call, or should the gradient TINT toward the fodder's dominant
  colour for a hint of identity while staying "simple"? (operator said "simpler" + "basic
  gradients" -> lean gradient-only, but confirm.)
- V4: Is there a cheaper artifact win than gen-param tuning -- e.g. a Blender `decimate` or
  `smooth`/`corrective_smooth` modifier pass to melt the lumps post-mesh, deterministically, on
  CPU? Trade-off vs eroding real features.
- V5: Gradient direction/colours: vertical (Z) light-over-dark is the safe "museum sculpt"; does a
  subtle front-to-back (X) or normal-based shade read better for a turntable with a bounded arc?
