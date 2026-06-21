# OTR 3D Mesh-Quality v1.1 -- CONVERGED build plan (pass01)

Hardened by the panel (GPT-5.5 + Gemini-3.1-pro), grounded + judged by Claude. C is CUT.
Two chunks, each: full suite + Bug Bible -> commit AND push to v2.0-alpha -> verify HEAD==origin /
no 0-byte / no BOM / AST-parse. Then ONE GPU re-smoke. Invariants unchanged (local, deterministic,
audio byte-identical, GLB geometry-only, WORKBENCH-only, UTF-8 no BOM, SFW).

## Chunk 1 -- A: tighter fodder prompt (content-only, `nodes/otr_meta_brief_image_prompt.py`)
- `MESH_FODDER_POS_SCAFFOLD` -> force a clean, mesh-friendly subject: "simple clean unbroken
  silhouette, smooth solid form, plain matte solid-colour clothing, short tight neat hair, neutral
  symmetrical forward stance, full unoccluded three-quarter view, entire head and body clearly
  visible, plain seamless neutral mid-grey studio backdrop, even soft diffuse frontal lighting, no
  hard shadows, no props, sharp focus".
- `MESH_FODDER_NEG_SCAFFOLD` -> add the artifact drivers: "loose flowing hair, hair wisps, fine
  surface detail, intricate texture, frills, thin protrusions, jewellery" (keep existing terms).
- Update the chunk-3 prompt-content test assertions (`tests/test_3d_image_streams.py`:
  `test_fork_mints_fodder_and_plate_not_scene_still` asserts `"neutral mid-grey" in prompt` -- keep
  a stable substring; assert a new one like `"smooth solid form"`).

## Chunk 2 -- B: gradient sculpt surface (`scripts/otr_mesh_stage_blender.py` + `eng_mesh_stage.py`)
PURE (CPU-tested):
- `gradient_color(co_z, top=(0.86,0.87,0.92), bottom=(0.30,0.31,0.38))` -> CLAMP `co_z` to
  [-0.5,0.5], then `s=(co_z+0.5)` in [0,1], lerp bottom->top, return (r,g,b) in [0,1]. Tests:
  range, monotonic-in-z, clamp out-of-range, determinism.

BLENDER (real-Blender, exercised by the selftest + GPU smoke):
- `_paint_gradient_onto_meshes(bpy, meshes)`: new per-VERTEX `otr_proj` FLOAT_COLOR (POINT domain)
  exactly like the projection; for each vert `co = obj.matrix_world @ v.co`; color =
  `gradient_color(co.z)`; set active+render color attr (reuse the projection's active/render
  selection code). Returns distinct-colour count.
- `_smooth_mesh_normals(meshes)`: set `poly.use_smooth = True` on every polygon of each mesh's
  DATA (NO `bpy.ops`, no context needed). Called after import/normalize, before render.
- `--surface {flat,gradient,portrait}` arg, **default `flat`** (so an omitted arg keeps legacy
  flat-gray render). State machine in `main()` (render mode):
  - `flat`  -> paint nothing (gray matcap; `_has_vertex_colors` False -> SINGLE).
  - `gradient` -> `_paint_gradient_onto_meshes`; IGNORE `--portrait` if also passed (LOUD warn).
  - `portrait` -> require `--portrait` (else `p.error`/raise LOUD); project the still (existing path).
  - selftest mode is UNCHANGED (still projects its in-memory portrait + non-uniformity gate).
  - `_smooth_mesh_normals` runs for ALL render-mode surfaces (gradient + portrait + flat).
- `build_blender_cmd(..., surface="")` -> append `--surface <surface>` ONLY when set (omitted-arg
  legacy invocation byte-identical; the legacy `--portrait`-only call path is preserved).
- `eng_mesh_stage.render_clip`: default `surface="gradient"`, do NOT pass the fodder photo;
  `OTR_MESH_PROJECT_PORTRAIT=1` -> `surface="portrait"` + `portrait=still` (opt-in decal).
- Tests: `build_blender_cmd` emits `--surface gradient` (and omits it when unset = legacy);
  surface state-machine arg validation (portrait-without-image errors).

## C -- CUT
No mesher-gen-param change in v1.1. `OTR_HY3D_VOXEL_THRESHOLD` would be a cache trap
(`mesh_cache_key` excludes gen params). The lump fix is A (clean fodder); the faceted-look fix is
B (per-poly smooth). A cache-aware mesher-tuning sprint is the place for gen params if ever needed.

## GPU re-smoke (after both chunks green + pushed)
Reset box; boot FLOOR + `OTR_ENABLE_MESH_STAGE=1`; all-slots mesh_stage 30w on the REAL
`otr_scifi_16gb_full.json`; confirm `obs_publish OK` + the obs mp4; operator look-QA on the gradient
sculpt + cleaner meshes. The mesh GLB cache from the prior smoke is per-subject -- a fresh cast/seed
(no OTR_C7) re-meshes; if reusing the same subjects, purge `episodes/_shared/mesh_cache` so the
tighter fodder actually re-generates (fodder content hash changes -> new key, so a new prompt
auto-misses; portrait-content-hash keying makes this automatic).

## Verify-at-build (GPU)
- WORKBENCH VERTEX + per-poly smooth renders the gradient as a soft ramp (not faceted bands).
- `_smooth_mesh_normals` raises no headless context error (data-level, should be safe).
- The gradient reads as a sculpted form over the background plate; no flat photo decal by default.
