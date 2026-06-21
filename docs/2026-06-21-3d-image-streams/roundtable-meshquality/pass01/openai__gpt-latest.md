<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still has unresolved build/runtime choices in B, and C’s proposed env knob is incompatible with the current mesh cache key.

MUST-FIX BEFORE BUILD:
1. [C] `OTR_HY3D_VOXEL_THRESHOLD` will not be A/B-able with the current cache. Grounding: `render_clip()` keys the GLB only by `mesh_cache_key(subject_id, portrait_content_hash(still))`; `_build_mesh_graph()` already reads gen env (`OTR_HY3D_STEPS`, `OTR_HY3D_CFG`) but none of those parameters are in the cache key or manifest key. A new threshold env would be ignored on cache hits and would create geometry under the same key as default on cache misses. Concrete fix: either cut C for v1.1, or add a cache strategy before adding the knob: include all geometry-affecting HY3D params in the cache identity/manifest, or require/document a separate `OTR_MESH_CACHE_DIR`/cache purge for A/B and make the code log the effective threshold and cache key policy. Do not claim “A/B without code change” while cache hits bypass the graph.

2. [B] The `--surface` default contradicts “legacy invocation byte-identical.” Current `build_blender_cmd()` omits optional args unless set; current stage with no `--portrait` renders flat gray. The plan says stage `--surface` defaults to `gradient`, while also saying `build_blender_cmd surface=""` is appended only when set so legacy invocation is byte-identical. The command bytes may be identical, but the render behavior is not. Concrete fix: choose one contract. Smallest safe fix: make Blender parser default `surface="flat"` to preserve omitted-arg legacy behavior, and have `eng_mesh_stage.render_clip()` explicitly pass `--surface gradient` for the new default. If you intentionally want all direct script invocations to change to gradient, remove the byte-identical claim and update tests/docs accordingly.

3. [B / V2] Do not implement `bpy.ops.object.shade_smooth()` as planned without resolving the headless context issue. Grounding: current `_import_glb()` returns mesh objects but does not set active object/selection; `main()` does not manage selection before render. Blender ops are context-sensitive in `--background`. Concrete fix: avoid the operator and set smoothing directly on imported mesh data, e.g. for each mesh object set each polygon’s `use_smooth = True` after import/normalization. Add a real-Blender selftest path that exercises this.

4. [B] The surface-mode state machine is underspecified and can silently produce the wrong look. The plan says `portrait (and --portrait set) -> project the still`, but does not define behavior for `--surface portrait` without `--portrait`, `--portrait` supplied with `--surface gradient`, or `--surface flat` with existing color attributes. Concrete fix: define and enforce:
   - `surface=gradient`: paint gradient and ignore/forbid `--portrait`.
   - `surface=portrait`: require `--portrait` in render mode and project it.
   - `surface=flat`: do not paint; optionally remove/ignore color attributes before `_configure_render`.
   Add argparse/main errors for invalid combinations so failures are loud.

5. [B] Gradient coordinates must use the same normalized world-space coordinates as projection, not raw local `v.co.z`. Grounding: `_normalize_meshes()` changes object location/scale; `_project_portrait_onto_meshes()` correctly samples `co = obj.matrix_world @ v.co`. The plan says `gradient_color(co.z)` but does not specify world vs local. Concrete fix: in `_paint_gradient_onto_meshes()`, compute `co = obj.matrix_world @ v.co` after `_normalize_meshes()` and pass `co.z` to `gradient_color()`.

6. [B] `gradient_color(co_z)` must clamp its input. The normalized object coordinate assumption is “~[-0.5, 0.5]” in `project_uv()`, not a hard guarantee for every mesh/object after global longest-dimension normalization. Concrete fix: clamp `co_z` to `[-0.5, 0.5]` inside `gradient_color()` before lerp, and test out-of-range inputs as well as monotonicity/range.

SHOULD-FIX:
1. [A] Extending `